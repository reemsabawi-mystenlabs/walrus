// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

//! Serialization format for blob info snapshots.
//!
//! A blob info snapshot is a deterministic serialization of the blob info tables that are
//! identical across all honest nodes at the epoch boundary (after garbage-collection phase 1):
//! `per_object_blob_info`, `per_object_pooled_blob_info`, and `storage_pool_info`. The
//! `aggregate_blob_info` table is deliberately excluded: it is a materialized view that contains
//! node-local state (`is_metadata_stored`) and entries whose deletion timing depends on the
//! background GC phase 2, so it is not deterministic across nodes; it is reconstructed from the
//! per-object tables during recovery.
//!
//! This module contains only the writer: the storage node produces snapshots, but never reads
//! them back during normal operation. Deserialization belongs to the (not yet implemented) node
//! recovery workflow and is exercised offline through `db-tool bench-blob-info-snapshot`.
//!
//! The format is versioned and self-delimiting:
//!
//! ```text
//! +----------------------+
//! |    Magic (4 B, BE)   |
//! +----------------------+
//! |  Version (4 B, BE)   |
//! +----------------------+
//! | Header len (4 B, BE) |
//! +----------------------+
//! |  Header (BCS bytes)  |
//! +----------------------+
//! |  Section (tag = 1)   |  per_object_blob_info
//! +----------------------+
//! |  Section (tag = 2)   |  per_object_pooled_blob_info
//! +----------------------+
//! |  Section (tag = 3)   |  storage_pool_info
//! +----------------------+
//! |  Checksum (8 B, BE)  |  xxhash64 of all preceding bytes
//! +----------------------+
//!
//! Section := tag (1 B)
//!            { 0x01 | key len (4 B, BE) | key BCS | value len (4 B, BE) | value BCS }*
//!            0x00 | entry count (8 B, BE)
//! ```
//!
//! Entries within a section are required to be in strictly increasing key order (the natural
//! RocksDB iteration order), which makes the serialization a pure function of the table contents.
//! Any change that affects the serialized bytes (entry types, section layout, future compression)
//! MUST bump [`SNAPSHOT_FORMAT_VERSION`]: the snapshot bytes are consensus-critical, since all
//! nodes must produce bit-identical snapshots for the same epoch.

use std::{hash::Hasher as _, io::Write};

use byteorder::{BigEndian, WriteBytesExt};
use serde::{Deserialize, Serialize};
use sui_types::base_types::ObjectID;
use twox_hash::XxHash64;
use typed_store::TypedStoreError;
use walrus_core::{BlobId, Epoch};

use super::blob_info::{PerObjectBlobInfo, PerObjectPooledBlobInfo, StoragePoolInfo};
use crate::event::events::EventStreamCursor;

/// The magic bytes at the start of a blob info snapshot.
pub(crate) const SNAPSHOT_MAGIC: u32 = 0xB10B1F05;
/// The current format version of the blob info snapshot.
pub(crate) const SNAPSHOT_FORMAT_VERSION: u32 = 1;

const SECTION_TAG_PER_OBJECT: u8 = 1;
const SECTION_TAG_PER_OBJECT_POOLED: u8 = 2;
const SECTION_TAG_STORAGE_POOL: u8 = 3;

const ENTRY_MARKER: u8 = 0x01;
const SECTION_END_MARKER: u8 = 0x00;

const CHECKSUM_SEED: u64 = 0;

/// Errors occurring during blob info snapshot serialization.
#[derive(Debug, thiserror::Error)]
pub(crate) enum SnapshotError {
    /// An I/O error occurred while writing the snapshot.
    #[error("I/O error during snapshot serialization")]
    Io(#[from] std::io::Error),
    /// Reading from the underlying database failed.
    #[error("database error during snapshot serialization")]
    Storage(#[from] TypedStoreError),
    /// BCS serialization of a header or entry failed.
    #[error("BCS serialization error")]
    Encoding(#[from] bcs::Error),
    /// The snapshot is structurally invalid.
    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
    /// The entries of a section are not in strictly increasing key order.
    #[error("keys are not in strictly increasing order in section with tag {0}")]
    UnsortedKeys(u8),
}

/// The header of a blob info snapshot.
///
/// The header pins the exact event-stream position the snapshot corresponds to: the snapshot
/// contains the table state after applying all events up to and including `event_cursor`, with
/// the inline GC phase 1 for `epoch` applied. The chunk fields are reserved for splitting large
/// snapshots across multiple blobs; the current writer always produces a single chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SnapshotHeader {
    /// The epoch whose boundary this snapshot was taken at.
    pub epoch: Epoch,
    /// The position of the last event included in the snapshot.
    pub event_cursor: EventStreamCursor,
    /// The blob ID of the previous epoch's snapshot, or [`BlobId::ZERO`] if unknown.
    pub prev_snapshot_blob_id: BlobId,
    /// The index of this chunk; always 0 until chunking is implemented.
    pub chunk_index: u32,
    /// The total number of chunks; always 1 until chunking is implemented.
    pub chunk_count: u32,
}

impl SnapshotHeader {
    /// Creates a single-chunk snapshot header.
    pub fn new(
        epoch: Epoch,
        event_cursor: EventStreamCursor,
        prev_snapshot_blob_id: BlobId,
    ) -> Self {
        Self {
            epoch,
            event_cursor,
            prev_snapshot_blob_id,
            chunk_index: 0,
            chunk_count: 1,
        }
    }
}

/// Statistics about a written snapshot, for logging and metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct SnapshotStats {
    /// The total number of bytes written, including the checksum.
    pub bytes_written: u64,
    /// The number of entries serialized from `per_object_blob_info`.
    pub per_object_count: u64,
    /// The number of entries serialized from `per_object_pooled_blob_info`.
    pub per_object_pooled_count: u64,
    /// The number of entries serialized from `storage_pool_info`.
    pub storage_pool_count: u64,
    /// The xxhash64 checksum of the snapshot contents (also stored in the snapshot trailer).
    ///
    /// Since the serialization is deterministic, this is a fingerprint of the snapshotted
    /// table contents: nodes with identical tables produce identical checksums.
    pub checksum: u64,
}

/// A writer wrapper that maintains a running xxhash64 and byte count of everything written.
struct HashingWriter<W> {
    inner: W,
    hasher: XxHash64,
    bytes_written: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: XxHash64::with_seed(CHECKSUM_SEED),
            bytes_written: 0,
        }
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.hasher.write(&buf[..written]);
        self.bytes_written += u64::try_from(written).expect("usize fits in u64");
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Serializes a blob info snapshot to `writer`.
///
/// The entry iterators must yield entries in strictly increasing key order, as produced by
/// RocksDB iteration; this is checked and [`SnapshotError::UnsortedKeys`] is returned otherwise.
pub(crate) fn write_snapshot<W: Write>(
    writer: W,
    header: &SnapshotHeader,
    per_object: impl IntoIterator<Item = Result<(ObjectID, PerObjectBlobInfo), TypedStoreError>>,
    per_object_pooled: impl IntoIterator<
        Item = Result<(ObjectID, PerObjectPooledBlobInfo), TypedStoreError>,
    >,
    storage_pools: impl IntoIterator<Item = Result<(ObjectID, StoragePoolInfo), TypedStoreError>>,
) -> Result<SnapshotStats, SnapshotError> {
    let mut writer = HashingWriter::new(writer);

    writer.write_u32::<BigEndian>(SNAPSHOT_MAGIC)?;
    writer.write_u32::<BigEndian>(SNAPSHOT_FORMAT_VERSION)?;
    let header_bytes = bcs::to_bytes(header)?;
    writer.write_u32::<BigEndian>(checked_len(header_bytes.len())?)?;
    writer.write_all(&header_bytes)?;

    let per_object_count = write_section(&mut writer, SECTION_TAG_PER_OBJECT, per_object)?;
    let per_object_pooled_count = write_section(
        &mut writer,
        SECTION_TAG_PER_OBJECT_POOLED,
        per_object_pooled,
    )?;
    let storage_pool_count = write_section(&mut writer, SECTION_TAG_STORAGE_POOL, storage_pools)?;

    let checksum = writer.hasher.finish();
    writer.write_u64::<BigEndian>(checksum)?;
    writer.flush()?;

    Ok(SnapshotStats {
        bytes_written: writer.bytes_written,
        per_object_count,
        per_object_pooled_count,
        storage_pool_count,
        checksum,
    })
}

fn write_section<W: Write, V: Serialize>(
    writer: &mut W,
    tag: u8,
    entries: impl IntoIterator<Item = Result<(ObjectID, V), TypedStoreError>>,
) -> Result<u64, SnapshotError> {
    writer.write_u8(tag)?;
    let mut count: u64 = 0;
    let mut previous_key: Option<ObjectID> = None;

    for entry in entries {
        let (key, value) = entry?;
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(SnapshotError::UnsortedKeys(tag));
        }
        previous_key = Some(key);

        writer.write_u8(ENTRY_MARKER)?;
        let key_bytes = bcs::to_bytes(&key)?;
        writer.write_u32::<BigEndian>(checked_len(key_bytes.len())?)?;
        writer.write_all(&key_bytes)?;
        let value_bytes = bcs::to_bytes(&value)?;
        writer.write_u32::<BigEndian>(checked_len(value_bytes.len())?)?;
        writer.write_all(&value_bytes)?;
        count += 1;
    }

    writer.write_u8(SECTION_END_MARKER)?;
    writer.write_u64::<BigEndian>(count)?;
    Ok(count)
}

fn checked_len(len: usize) -> Result<u32, SnapshotError> {
    u32::try_from(len)
        .map_err(|_| SnapshotError::Corrupt(format!("entry of {len} bytes exceeds the u32 limit")))
}
