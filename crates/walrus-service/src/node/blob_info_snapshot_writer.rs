// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

//! Epoch-boundary in-process serialization of blob info snapshots.
//!
//! At the epoch boundary, directly after garbage-collection phase 1 and before any further
//! events are processed, the blob info tables are identical across all honest nodes. When
//! enabled, this module serializes the three blob-info column families in-process at exactly
//! that point and removes the previous epoch's snapshot, so that at most one snapshot exists at
//! a time. The serialized size and content digest are reported through metrics and a log line;
//! operators compare digests for the same epoch across nodes.

use std::{
    fs,
    io::{BufWriter, Write as _},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use walrus_core::{BlobId, Epoch};

use super::{
    StorageNodeInner,
    storage::blob_info_snapshot::{SnapshotHeader, SnapshotStats},
};
use crate::event::events::EventStreamCursor;

/// Configuration for the blob info snapshot writer.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BlobInfoSnapshotWriterConfig {
    /// Whether to serialize a blob info snapshot at each epoch boundary.
    ///
    /// When enabled, the node serializes the three blob-info column families in-process at the
    /// post-GC-phase-1 boundary and reports the serialization duration, size, and digest. Note
    /// that disabling this flag leaves the last snapshot file on disk until it is removed
    /// manually.
    pub enabled: bool,
}

/// Returns the directory under which the writer keeps its snapshots.
pub fn snapshot_base_dir(storage_path: &Path) -> PathBuf {
    storage_path.join("blob_info_snapshots")
}

fn snapshot_file_path(base_dir: &Path, epoch: Epoch) -> PathBuf {
    base_dir.join(format!("snapshot_epoch_{epoch}.bin"))
}

/// Parses the epoch out of a (possibly temporary) snapshot file name.
fn snapshot_file_epoch(file_name: &str) -> Option<Epoch> {
    file_name
        .strip_suffix(".tmp")
        .unwrap_or(file_name)
        .strip_prefix("snapshot_epoch_")?
        .strip_suffix(".bin")?
        .parse()
        .ok()
}

fn saturating_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Serializes the blob info tables in-process at the epoch boundary, reports the serialization
/// duration, size, and digest, and removes older snapshot files.
///
/// Must be called at the deterministic post-GC-phase-1 point, while event processing is blocked,
/// so the serialized snapshot is identical across honest nodes for the same epoch. It touches
/// only the three blob-info column families (read through a single RocksDB engine snapshot), not
/// the whole database. The header is built exactly as the offline `db-tool
/// bench-blob-info-snapshot` builds it (epoch embedded, default cursor, zero previous blob ID),
/// so the reported digest matches the db-tool digest for the same epoch.
pub(super) async fn serialize_snapshot_at_epoch_boundary(
    node: Arc<StorageNodeInner>,
    epoch: Epoch,
) -> Result<()> {
    let base_dir = node.blob_info_snapshot_dir.clone();
    fs::create_dir_all(&base_dir)?;
    remove_snapshot_files_matching(&base_dir, |snapshot_epoch| snapshot_epoch != epoch);

    let final_path = snapshot_file_path(&base_dir, epoch);
    if final_path.exists() {
        // Already created, e.g., because the epoch change event is being reprocessed after a
        // restart.
        tracing::debug!(walrus.epoch = epoch, "blob info snapshot already exists");
        return Ok(());
    }
    let tmp_path = base_dir.join(format!("snapshot_epoch_{epoch}.bin.tmp"));
    if tmp_path.exists() {
        fs::remove_file(&tmp_path)?;
    }

    let start = Instant::now();
    let storage_node = node.clone();
    let serialize_tmp_path = tmp_path.clone();
    let stats = tokio::task::spawn_blocking(move || -> Result<SnapshotStats> {
        let header = SnapshotHeader::new(epoch, EventStreamCursor::default(), BlobId::ZERO);
        let file = fs::File::create(&serialize_tmp_path)?;
        let mut writer = BufWriter::with_capacity(1 << 20, file);
        let stats = storage_node
            .storage
            .write_blob_info_snapshot(&header, &mut writer)?;
        writer.flush()?;
        writer
            .into_inner()
            .context("failed to flush the snapshot file")?
            .sync_all()?;
        Ok(stats)
    })
    .await
    .context("snapshot serialization task panicked")??;
    fs::rename(&tmp_path, &final_path)?;
    let elapsed = start.elapsed();

    node.metrics
        .blob_info_snapshot_serialize_duration_seconds
        .set(elapsed.as_secs_f64());
    node.metrics
        .blob_info_snapshot_size_bytes
        .set(saturating_i64(stats.bytes_written));
    let digest = format!("{:016x}", stats.checksum);
    tracing::info!(
        walrus.epoch = epoch,
        ?elapsed,
        size_bytes = stats.bytes_written,
        per_object = stats.per_object_count,
        per_object_pooled = stats.per_object_pooled_count,
        storage_pool = stats.storage_pool_count,
        digest = %digest,
        path = %final_path.display(),
        "serialized blob info snapshot in-process"
    );
    Ok(())
}

/// Removes all snapshot files (including temporary ones) whose epoch matches `should_remove`.
fn remove_snapshot_files_matching(base_dir: &Path, should_remove: impl Fn(Epoch) -> bool) {
    let Ok(entries) = fs::read_dir(base_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if snapshot_file_epoch(&name).is_some_and(&should_remove)
            && let Err(error) = fs::remove_file(entry.path())
        {
            tracing::warn!(
                ?error,
                path = %entry.path().display(),
                "failed to remove blob info snapshot file"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn config_default_is_disabled() {
        assert!(!BlobInfoSnapshotWriterConfig::default().enabled);
        let parsed: BlobInfoSnapshotWriterConfig =
            serde_yaml::from_str("enabled: true\n").expect("config should deserialize");
        assert!(parsed.enabled);
    }

    #[test]
    fn snapshot_file_epoch_parses_file_names() {
        assert_eq!(snapshot_file_epoch("snapshot_epoch_7.bin"), Some(7));
        assert_eq!(snapshot_file_epoch("snapshot_epoch_7.bin.tmp"), Some(7));
        assert_eq!(snapshot_file_epoch("unrelated"), None);
        assert_eq!(snapshot_file_epoch("snapshot_epoch_x.bin"), None);
    }

    #[test]
    fn keep_latest_removes_other_epochs() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path();
        fs::write(snapshot_file_path(base, 3), b"old")?;
        fs::write(snapshot_file_path(base, 4), b"keep")?;
        fs::write(base.join("snapshot_epoch_4.bin.tmp"), b"tmp")?;
        fs::write(base.join("unrelated.file"), b"keep me")?;

        remove_snapshot_files_matching(base, |epoch| epoch != 4);

        assert!(!snapshot_file_path(base, 3).exists());
        assert!(snapshot_file_path(base, 4).exists());
        // The temporary file for epoch 4 is kept by this filter; the serialization's atomic
        // rename replaces it otherwise.
        assert!(base.join("snapshot_epoch_4.bin.tmp").exists());
        assert!(base.join("unrelated.file").exists());
        Ok(())
    }
}
