// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

//! Utilities to publish the walrus contracts and deploy a system object for testing.

use std::{
    num::NonZeroU16,
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use move_core_types::language_storage::StructTag;
use move_package_alt::{
    RootPackage,
    schema::{
        Environment,
        EnvironmentID,
        EnvironmentName,
        OriginalID,
        Publication,
        PublishAddresses,
        PublishedID,
    },
};
use move_package_alt_compilation::build_config::BuildConfig as MoveBuildConfig;
use sui_move_build::{CompiledPackage, PackageDependencies};
use sui_package_alt::{BuildParams, SuiFlavor};
use sui_package_management::LockCommand;
use sui_rpc_api::client::ExecutedTransaction;
use sui_sdk::types::{
    Identifier,
    base_types::ObjectID,
    programmable_transaction_builder::ProgrammableTransactionBuilder,
    transaction::TransactionData,
};
use sui_types::{
    SUI_CLOCK_OBJECT_ID,
    SUI_CLOCK_OBJECT_SHARED_VERSION,
    effects::TransactionEffectsAPI,
    execution_status::ExecutionStatus,
    transaction::{ObjectArg, SharedObjectMutability, TransactionKind},
};
use walkdir::WalkDir;
use walrus_core::{EpochCount, ensure};

#[cfg(any(test, feature = "test-utils"))]
use crate::test_utils::system_setup;
use crate::{
    client::retry_client::{
        RetriableSuiClient,
        retriable_sui_client::{GasBudgetAndPrice, LazySuiClientBuilder, MAX_GAS_PAYMENT_OBJECTS},
    },
    coin::Coin,
    contracts,
    wallet::Wallet,
};

/// Gets the objects of the given type that were created in an [`ExecutedTransaction`].
fn get_created_object_ids_by_type(
    response: &ExecutedTransaction,
    struct_tag: &StructTag,
) -> Result<Vec<ObjectID>> {
    use std::collections::HashSet;

    use sui_types::effects::TransactionEffectsAPI;

    let created_ids: HashSet<_> = response
        .effects
        .created()
        .iter()
        .map(|(obj_ref, _)| obj_ref.0)
        .collect();

    let struct_tag_str = struct_tag.to_canonical_string(true);

    Ok(response
        .changed_objects
        .iter()
        .filter_map(|o| {
            let id: ObjectID = o.object_id().parse().ok()?;
            if created_ids.contains(&id) && o.object_type() == struct_tag_str {
                Some(id)
            } else {
                None
            }
        })
        .collect())
}

#[cfg(any(test, feature = "test-utils"))]
fn get_pkg_id_from_tx_response(tx_response: &ExecutedTransaction) -> Result<ObjectID> {
    tx_response
        .get_new_package_obj()
        .map(|(id, _, _)| id)
        .ok_or_else(|| anyhow!("no package object was created"))
}

#[cfg(any(test, feature = "test-utils"))]
pub(crate) async fn publish_package_with_default_build_config(
    wallet: &mut Wallet,
    package_path: PathBuf,
    gas_budget: Option<u64>,
) -> Result<ExecutedTransaction> {
    publish_package(wallet, package_path, Default::default(), gas_budget).await
}

/// Selects the package-system environment to compile and publish under.
///
/// Mirrors `sui_package_alt::find_environment`: it prefers an environment the package manifest
/// declares (the `[environments]` table plus `SuiFlavor`'s `testnet`/`mainnet` defaults) whose
/// chain id matches the wallet's active chain id. When such an environment exists, the env *name*
/// and the chain id are consistent, so the env name and chain id agree and each dependency's
/// published addresses resolve under the matching `[published.<env>]` key. Two scenarios reach
/// this honest path:
///  - production `testnet`/`mainnet`, where the wallet alias and chain id match a default env;
///  - a custom or local network the operator has registered by adding an
///    `[environments] <alias> = "<chain id>"` entry to each package's `Move.toml`.
///
/// When no declared environment matches the wallet's chain id, it falls back to `testnet`. This
/// keeps the previous behavior for unregistered networks — addresses are written and read under the
/// `testnet` env key — and is the only case that pairs an env name with a possibly-different chain
/// id. Automated test clusters mint a fresh chain id per run that no manifest declares, so they
/// take this fallback; that is acceptable because they publish to throwaway local networks.
fn select_environment(
    package_path: &Path,
    chain_id: Option<String>,
    wallet: &Wallet,
) -> Result<Environment> {
    let alias = wallet.get_active_env().alias.clone();
    let chain_id = chain_id.unwrap_or_default();
    let manifest_envs: Vec<(EnvironmentName, EnvironmentID)> =
        RootPackage::<SuiFlavor>::environments(package_path, &wallet.sui_flavor())
            .context("failed to read the package environments from the manifest")?
            .into_iter()
            .collect();
    Ok(resolve_environment(&manifest_envs, alias, chain_id))
}

/// Pure environment-selection logic for `select_environment`; see its docs for the rationale.
fn resolve_environment(
    manifest_envs: &[(EnvironmentName, EnvironmentID)],
    alias: String,
    chain_id: String,
) -> Environment {
    // 1. Exact match: the manifest declares the active alias with the active chain id.
    if manifest_envs
        .iter()
        .any(|(name, id)| *name == alias && *id == chain_id)
    {
        return Environment::new(alias, chain_id);
    }

    // 2. Unique match by chain id: exactly one declared environment uses this chain id.
    let mut by_chain_id = manifest_envs.iter().filter(|(_, id)| id == &chain_id);
    if let Some((name, id)) = by_chain_id.next()
        && by_chain_id.next().is_none()
    {
        return Environment::new(name.clone(), id.clone());
    }

    // 3. Fall back to `testnet` for an unregistered network.
    Environment::new("testnet".to_string(), chain_id)
}

/// Compiles a package and returns the compiled package, and build config.
///
/// Loads the root package in persistent mode under the environment chosen by `select_environment`
/// from the wallet's active alias and chain id. Production `testnet`/`mainnet` resolve to a default
/// environment whose name and chain id agree; an unregistered network (such as a test cluster's
/// `localnet`) falls back to the `testnet` env name. The selected env name keys the
/// `[published.<env>]` lookups, so package addresses propagate between sibling packages via each
/// one's per-directory `Published.toml`.
pub async fn compile_package(
    package_path: PathBuf,
    build_config: MoveBuildConfig,
    chain_id: Option<String>,
    wallet: &Wallet,
) -> Result<(CompiledPackage, MoveBuildConfig, RootPackage<SuiFlavor>)> {
    let env = select_environment(&package_path, chain_id.clone(), wallet)?;

    let build_config_clone = build_config.clone();
    let package_path_clone = package_path.clone();
    let root_pkg: RootPackage<SuiFlavor> = build_config_clone
        .package_loader(&package_path_clone, &env, wallet.sui_flavor())
        // TODO(WAL-1125): under simtest the package system can't fetch external git deps, but
        // `Move.lock`'s `[pinned.testnet]` still points its `std`/`sui` entries at git sources.
        // The default loader fetches each lockfile pin before checking digests, which triggers a
        // real `git clone` that hangs under msim. Forcing a repin makes the loader read the
        // (already-rewritten) manifest with local paths instead. Drop once simtest can pull
        // external git deps, together with `update_contract_sui_dependency_to_local_copy`.
        .force_repin(cfg!(msim))
        .load()
        .await?;

    tokio::task::spawn_blocking(|| {
        sui_macros::nondeterministic!(compile_package_inner_blocking(
            package_path,
            build_config,
            chain_id,
            root_pkg
        ))
    })
    .await?
}

/// Synchronous method to compile the package. Should only be called from an async context
/// using `tokio::task::spawn_blocking` or similar methods.
fn compile_package_inner_blocking(
    package_path: PathBuf,
    build_config: MoveBuildConfig,
    chain_id: Option<String>,
    root_pkg: RootPackage<SuiFlavor>,
) -> Result<(CompiledPackage, MoveBuildConfig, RootPackage<SuiFlavor>)> {
    let mut stdout = std::io::stdout();
    let package = move_package_alt_compilation::compile_from_root_package::<
        std::io::Stdout,
        SuiFlavor,
    >(&root_pkg, &build_config, &mut stdout)
    .expect("Compilation should succeed");

    let package_dependencies = PackageDependencies::new(&root_pkg)?;
    tracing::info!(
        "package path {:?}, chain_id {:?}, package_dependencies {:?}",
        package_path,
        chain_id,
        package_dependencies
    );

    // Check that all dependencies are published.
    if !package_dependencies.unpublished.is_empty() {
        bail!(
            "Walrus packages must not have unpublished dependencies. Unpublished dependencies: {}
        ",
            package_dependencies
                .unpublished
                .values()
                .map(|dep| dep.name.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let published_at = root_pkg
        .publication()
        .map(|p| ObjectID::from_address(p.addresses.published_at.0));

    let compiled_package = CompiledPackage {
        package,
        published_at,
        dependency_ids: package_dependencies,
    };

    ensure!(
        compiled_package.published_root_module().is_none(),
        "package was already published, modules must all have 0x0 as their addresses."
    );

    Ok((compiled_package, build_config, root_pkg))
}

// TODO(WAL-1127): this is a complete copy of the function from
// https://github.com/MystenLabs/sui/blob/a5ee2b9b736e712dfc917c5226ae1d835c59abd9/
// crates/sui/src/client_commands.rs#L3548
// Make that function in sui publicly accessible and use it here.
/// Return the update publication data, without writing it to lockfile
pub fn update_publication(
    chain_id: &str,
    command: LockCommand,
    response: &ExecutedTransaction,
    _build_config: &MoveBuildConfig,
    publication: Option<&mut Publication<SuiFlavor>>,
) -> Result<Publication<SuiFlavor>, anyhow::Error> {
    // Get the published package ID and version from the response
    let (published_id, version, _) = response.get_new_package_obj().ok_or_else(|| {
        anyhow!(
            "Expected a valid published package response but didn't see \
        one when attempting to update the `Move.lock`."
        )
    })?;

    match command {
        LockCommand::Publish => {
            let (upgrade_cap, _, _) = response
                .get_new_package_upgrade_cap()
                .ok_or_else(|| anyhow!("Expected a valid published package with a upgrade cap"))?;
            Ok(Publication::<SuiFlavor> {
                chain_id: chain_id.to_string(),
                metadata: sui_package_alt::PublishedMetadata {
                    toolchain_version: Some(env!("CARGO_PKG_VERSION").into()),
                    build_config: Some(sui_package_alt::BuildParams::default()),
                    upgrade_capability: Some(upgrade_cap),
                },
                addresses: PublishAddresses {
                    published_at: PublishedID(*published_id),
                    original_id: OriginalID(*published_id),
                },
                version: version.value(),
            })
        }
        LockCommand::Upgrade => {
            let publication =
                publication.expect("for upgrade there should already exist publication info");
            publication.addresses.published_at = PublishedID(*published_id);
            publication.version = version.value();
            // TODO: fix build config data
            publication.metadata.build_config = Some(BuildParams::default());
            publication.metadata.toolchain_version = Some(env!("CARGO_PKG_VERSION").into());
            // TODO: fix this, we should return a mut publication instead of creating a new one in
            // the Publish case
            Ok(publication.clone())
        }
    }
}

// This function is used to publish walrus packages in a local environment such as integration
// test or local testbed. This cannot be used in production environments.
#[cfg(any(test, feature = "test-utils"))]
#[tracing::instrument(err, skip(wallet, build_config))]
pub(crate) async fn publish_package(
    wallet: &mut Wallet,
    package_path: PathBuf,
    build_config: MoveBuildConfig,
    gas_budget: Option<u64>,
) -> Result<ExecutedTransaction> {
    let sender = wallet.active_address();
    let retry_client = RetriableSuiClient::new(
        vec![LazySuiClientBuilder::new(wallet.get_rpc_url(), None)],
        Default::default(),
    )?;

    let chain_id = retry_client.get_chain_identifier().await?;

    if cfg!(msim) {
        // TODO(WAL-1125): before the new sui package management system introduced in 1.63 can
        // support external dependencies, in simtest, we have to update all the implicit
        // dependencies to sui using a local copy of the sui repository.
        // The local copy should be pointed to by the SUI_REPO environment variable, and it should
        // match the sui version used by the walrus. The pulling logic is implemented in the
        // cargo-simtest script.
        //
        // Note that this must be done after the localnet environment is added to the Move.toml
        // file.
        system_setup::update_contract_sui_dependency_to_local_copy(package_path.clone())?;
    }

    let (compiled_package, final_build_config, mut root_package) =
        compile_package(package_path, build_config, Some(chain_id.clone()), wallet).await?;

    let compiled_modules = compiled_package.get_package_bytes(false);
    let transaction_kind = {
        let mut pt_builder = ProgrammableTransactionBuilder::new();
        let upgrade_cap = pt_builder.publish_upgradeable(
            compiled_modules,
            compiled_package
                .dependency_ids
                .published
                .into_values()
                .map(|dep| dep.published_at)
                .collect(),
        );
        pt_builder.transfer_arg(sender, upgrade_cap);
        TransactionKind::programmable(pt_builder.finish())
    };

    let GasBudgetAndPrice {
        gas_budget,
        gas_price,
    } = retry_client
        .gas_budget_and_price(gas_budget, sender, transaction_kind.clone())
        .await?;

    let gas_coins = retry_client
        .select_coins(
            sender,
            Coin::SUI,
            u128::from(gas_budget),
            vec![],
            MAX_GAS_PAYMENT_OBJECTS,
        )
        .await?
        .into_iter()
        .map(|coin| coin.object_ref())
        .collect::<Vec<_>>();

    let tx_data = TransactionData::new_with_gas_coins_allow_sponsor(
        transaction_kind,
        sender,
        gas_coins,
        gas_budget,
        gas_price,
        sender,
    );

    #[allow(deprecated)]
    let response = wallet
        .execute_transaction_may_fail(wallet.sign_transaction(&tx_data).await)
        .await?;

    // Write published data.
    let publish_data = update_publication(
        chain_id.as_str(),
        LockCommand::Publish,
        &response,
        &final_build_config,
        None,
    )?;

    root_package.write_publish_data(publish_data)?;

    Ok(response)
}

#[cfg(any(test, feature = "test-utils"))]
pub(crate) struct PublishSystemPackageResult {
    pub walrus_pkg_id: ObjectID,
    pub wal_exchange_pkg_id: Option<ObjectID>,
    pub credits_pkg_id: Option<ObjectID>,
    pub walrus_subsidies_pkg_id: Option<ObjectID>,
    pub init_cap_id: ObjectID,
    pub upgrade_cap_id: ObjectID,
    pub treasury_object_id: Option<ObjectID>,
}

/// Copy files from the `source` directory to the `destination` directory recursively.
#[tracing::instrument(err, skip(source, destination))]
pub async fn copy_recursively(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<()> {
    let source = source.as_ref().to_owned();
    let destination = destination.as_ref().to_owned();
    tokio::task::spawn_blocking(|| copy_recursively_inner_blocking(source, destination)).await?
}

/// Synchronous method to copy directories recursively. Should only be called from an async context
/// using `tokio::task::spawn_blocking` or similar methods.
fn copy_recursively_inner_blocking(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<()> {
    std::fs::create_dir_all(destination.as_ref())?;
    for entry in WalkDir::new(source.as_ref()) {
        let entry = entry?;
        let filetype = entry.file_type();
        let dest_path = entry.path().strip_prefix(source.as_ref())?;
        if filetype.is_dir() {
            std::fs::create_dir_all(destination.as_ref().join(dest_path))?;
        } else {
            std::fs::copy(entry.path(), destination.as_ref().join(dest_path))?;
        }
    }
    Ok(())
}

/// Publishes the `wal`, `wal_exchange`, `subsidies`, and `walrus` packages.
///
/// Returns the IDs of the packages, the `InitCap`, and the `UpgradeCap`.
///
/// If `deploy_directory` is provided, the contracts will be copied to this directory and published
/// from there to keep the `Move.toml` in the original directory unchanged.
///
/// If `use_existing_wal_token` is set, skips the deployment of the `wal` package. This requires
/// the package address to be set in the `wal/Move.lock` file for the current network.
#[cfg(any(test, feature = "test-utils"))]
#[tracing::instrument(err, skip(wallet))]
pub(crate) async fn publish_coin_and_system_package(
    wallet: &mut Wallet,
    InitSystemParams {
        contract_dir,
        deploy_directory,
        with_wal_exchange,
        use_existing_wal_token,
        with_credits,
        with_walrus_subsidies,
        ..
    }: InitSystemParams,
    gas_budget: Option<u64>,
) -> Result<PublishSystemPackageResult> {
    use sui_types::SUI_FRAMEWORK_ADDRESS;

    use crate::contracts::StructTag;

    const INIT_MODULE: &str = "init";
    const INIT_CAP_TAG: StructTag<'_> = StructTag {
        name: "InitCap",
        module: INIT_MODULE,
    };
    const UPGRADE_CAP_TAG: StructTag<'_> = StructTag {
        name: "UpgradeCap",
        module: "package",
    };

    let walrus_contract_directory = if let Some(deploy_directory) = deploy_directory {
        copy_recursively(&contract_dir, &deploy_directory).await?;

        // We're about to publish each of these packages *fresh* on the test cluster. The source
        // `testnet-contracts/<pkg>/Published.toml` files (carried over by `copy_recursively`)
        // record the real testnet publication, which under env name `"testnet"` would make the
        // compile think the package is already published and trip the
        // `published_root_module().is_none()` assertion. Drop each one so the fresh publishes
        // write clean entries.
        for entry in std::fs::read_dir(&deploy_directory)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let published_toml = entry.path().join("Published.toml");
                if published_toml.exists() {
                    std::fs::remove_file(&published_toml)?;
                }
            }
        }

        deploy_directory
    } else {
        contract_dir
    };

    let treasury_object_id = if !use_existing_wal_token {
        // Publish `wal` package.
        let wal_response = publish_package_with_default_build_config(
            wallet,
            walrus_contract_directory.join("wal"),
            gas_budget,
        )
        .await?;
        let wal_pkg_id = get_pkg_id_from_tx_response(&wal_response)?;
        let [treasury_id] = get_created_object_ids_by_type(
            &wal_response,
            &contracts::wal::ProtectedTreasury.to_move_struct_tag_with_package(wal_pkg_id, &[])?,
        )?[..] else {
            bail!("unexpected number of ProtectedTreasury objects created");
        };
        Some(treasury_id)
    } else {
        None
    };

    let wal_exchange_pkg_id = if with_wal_exchange {
        // Publish `wal_exchange` package.
        let transaction_response = publish_package_with_default_build_config(
            wallet,
            walrus_contract_directory.join("wal_exchange"),
            gas_budget,
        )
        .await?;
        Some(get_pkg_id_from_tx_response(&transaction_response)?)
    } else {
        None
    };

    // Publish `walrus` package.
    let transaction_response = publish_package_with_default_build_config(
        wallet,
        walrus_contract_directory.join("walrus"),
        gas_budget,
    )
    .await?;
    let walrus_pkg_id = get_pkg_id_from_tx_response(&transaction_response)?;

    let [init_cap_id] = get_created_object_ids_by_type(
        &transaction_response,
        &INIT_CAP_TAG.to_move_struct_tag_with_package(walrus_pkg_id, &[])?,
    )?[..] else {
        bail!("unexpected number of InitCap objects created");
    };

    let [upgrade_cap_id] = get_created_object_ids_by_type(
        &transaction_response,
        &UPGRADE_CAP_TAG.to_move_struct_tag_with_package(SUI_FRAMEWORK_ADDRESS.into(), &[])?,
    )?[..] else {
        bail!("unexpected number of UpgradeCap objects created");
    };

    let credits_pkg_id = if with_credits {
        // Publish `subsidies` package.
        let transaction_response = publish_package_with_default_build_config(
            wallet,
            walrus_contract_directory.join("subsidies"),
            gas_budget,
        )
        .await?;
        Some(get_pkg_id_from_tx_response(&transaction_response)?)
    } else {
        None
    };

    let walrus_subsidies_pkg_id = if with_walrus_subsidies {
        // Publish `walrus_subsidies` package.
        let transaction_response = publish_package_with_default_build_config(
            wallet,
            walrus_contract_directory.join("walrus_subsidies"),
            gas_budget,
        )
        .await?;
        Some(get_pkg_id_from_tx_response(&transaction_response)?)
    } else {
        None
    };

    Ok(PublishSystemPackageResult {
        walrus_pkg_id,
        wal_exchange_pkg_id,
        credits_pkg_id,
        init_cap_id,
        upgrade_cap_id,
        walrus_subsidies_pkg_id,
        treasury_object_id,
    })
}

/// Parameters used to call the `init_walrus` function in the Walrus contracts.
#[derive(Debug, Clone)]
pub struct InitSystemParams {
    /// Number of shards in the system.
    pub n_shards: NonZeroU16,
    /// Duration of the initial epoch in milliseconds.
    pub epoch_zero_duration: Duration,
    /// Duration of an epoch in milliseconds.
    pub epoch_duration: Duration,
    /// The maximum number of epochs ahead for which storage can be obtained.
    pub max_epochs_ahead: EpochCount,
    /// The directory containing the contract source code.
    pub contract_dir: PathBuf,
    /// The directory to deploy the contracts to.
    pub deploy_directory: Option<PathBuf>,
    /// Whether to publish the `wal_exchange` package.
    pub with_wal_exchange: bool,
    /// Whether to use an existing WAL token.
    pub use_existing_wal_token: bool,
    /// Whether to publish the `subsidies` package for client-side credits.
    pub with_credits: bool,
    /// Whether to publish the `walrus_subsidies` package for system subsidies.
    pub with_walrus_subsidies: bool,
}

/// Initialize the system and staking objects on chain.
///
/// Returns the IDs of the system, staking, and upgrade manager objects.
pub async fn create_system_and_staking_objects(
    wallet: &mut Wallet,
    contract_pkg_id: ObjectID,
    init_cap: ObjectID,
    upgrade_cap: ObjectID,
    system_params: InitSystemParams,
    gas_budget: Option<u64>,
) -> Result<(ObjectID, ObjectID, ObjectID)> {
    let mut pt_builder = ProgrammableTransactionBuilder::new();

    let epoch_duration_millis: u64 = system_params
        .epoch_duration
        .as_millis()
        .try_into()
        .context("epoch duration is too long")?;
    let epoch_zero_duration_millis: u64 = system_params
        .epoch_zero_duration
        .as_millis()
        .try_into()
        .context("genesis epoch duration is too long")?;

    // prepare the arguments
    #[allow(deprecated)]
    let init_cap_ref = wallet.get_object_ref(init_cap).await?;
    let init_cap_arg = pt_builder.input(init_cap_ref.into())?;

    #[allow(deprecated)]
    let upgrade_cap_ref = wallet.get_object_ref(upgrade_cap).await?;
    let upgrade_cap_arg = pt_builder.input(upgrade_cap_ref.into())?;

    let epoch_zero_duration_arg = pt_builder.pure(epoch_zero_duration_millis)?;
    let epoch_duration_arg = pt_builder.pure(epoch_duration_millis)?;
    let n_shards_arg = pt_builder.pure(system_params.n_shards.get())?;
    let max_epochs_ahead_arg = pt_builder.pure(system_params.max_epochs_ahead)?;
    let clock_arg = pt_builder.obj(ObjectArg::SharedObject {
        id: SUI_CLOCK_OBJECT_ID,
        initial_shared_version: SUI_CLOCK_OBJECT_SHARED_VERSION,
        mutability: SharedObjectMutability::Immutable,
    })?;

    // Create the system and staking objects
    let result = pt_builder.programmable_move_call(
        contract_pkg_id,
        Identifier::from_str(contracts::init::initialize_walrus.module)?,
        Identifier::from_str(contracts::init::initialize_walrus.name)?,
        vec![],
        vec![
            init_cap_arg,
            upgrade_cap_arg,
            epoch_zero_duration_arg,
            epoch_duration_arg,
            n_shards_arg,
            max_epochs_ahead_arg,
            clock_arg,
        ],
    );

    pt_builder.transfer_arg(wallet.active_address(), result);

    // finalize transaction
    let ptb = pt_builder.finish();
    let address = wallet.active_address();

    let retry_client = RetriableSuiClient::new(
        vec![LazySuiClientBuilder::new(wallet.get_rpc_url(), None)],
        Default::default(),
    )?;

    let GasBudgetAndPrice {
        gas_budget,
        gas_price,
    } = retry_client
        .gas_budget_and_price(
            gas_budget,
            address,
            TransactionKind::ProgrammableTransaction(ptb.clone()),
        )
        .await?;

    let gas_coins = retry_client
        .select_coins(
            address,
            Coin::SUI,
            u128::from(gas_budget),
            vec![],
            MAX_GAS_PAYMENT_OBJECTS,
        )
        .await?
        .into_iter()
        .map(|coin| coin.object_ref())
        .collect::<Vec<_>>();

    let transaction =
        TransactionData::new_programmable(address, gas_coins, ptb, gas_budget, gas_price);

    // sign and send transaction
    let signed_transaction = wallet.sign_transaction(&transaction).await;
    #[allow(deprecated)]
    let response = wallet
        .execute_transaction_may_fail(signed_transaction)
        .await?;

    if let ExecutionStatus::Failure(failure) = response.effects.status() {
        bail!(
            "Error during execution (command {:?}): {}",
            failure.command,
            failure.error
        );
    }

    let [staking_object_id] = get_created_object_ids_by_type(
        &response,
        &contracts::staking::Staking.to_move_struct_tag_with_package(contract_pkg_id, &[])?,
    )?[..] else {
        bail!("unexpected number of staking objects created");
    };

    let [system_object_id] = get_created_object_ids_by_type(
        &response,
        &contracts::system::System.to_move_struct_tag_with_package(contract_pkg_id, &[])?,
    )?[..] else {
        bail!("unexpected number of system objects created");
    };

    let [upgrade_manager_object_id] = get_created_object_ids_by_type(
        &response,
        &contracts::upgrade::UpgradeManager
            .to_move_struct_tag_with_package(contract_pkg_id, &[])?,
    )?[..] else {
        bail!("unexpected number of upgrade manager objects created");
    };

    Ok((
        system_object_id,
        staking_object_id,
        upgrade_manager_object_id,
    ))
}

#[cfg(test)]
mod tests {
    use super::resolve_environment;

    /// `SuiFlavor`'s default environments, as they appear in `manifest_envs` when a package
    /// declares no `[environments]` table of its own.
    fn defaults() -> Vec<(String, String)> {
        vec![
            ("testnet".to_string(), "4c78adac".to_string()),
            ("mainnet".to_string(), "35834a8a".to_string()),
        ]
    }

    #[test]
    fn production_env_matches_by_alias_and_chain_id() {
        // testnet/mainnet wallets hit the exact-match path: the alias is a declared env and the
        // chain id agrees, so the env name and chain id stay consistent.
        let env = resolve_environment(&defaults(), "testnet".to_string(), "4c78adac".to_string());
        assert_eq!(env.name(), "testnet");
        assert_eq!(env.id(), "4c78adac");

        let env = resolve_environment(&defaults(), "mainnet".to_string(), "35834a8a".to_string());
        assert_eq!(env.name(), "mainnet");
        assert_eq!(env.id(), "35834a8a");
    }

    #[test]
    fn registered_localnet_is_selected_honestly() {
        // When an operator registers `[environments] localnet = "<chain id>"`, the wallet alias and
        // chain id match it exactly, so we build under `localnet` rather than masquerading as
        // `testnet`.
        let mut envs = defaults();
        envs.push(("localnet".to_string(), "deadbeef".to_string()));
        let env = resolve_environment(&envs, "localnet".to_string(), "deadbeef".to_string());
        assert_eq!(env.name(), "localnet");
        assert_eq!(env.id(), "deadbeef");
    }

    #[test]
    fn registered_env_matches_by_chain_id_when_alias_differs() {
        // A declared env whose chain id matches is picked even when the wallet alias differs, as
        // long as the match is unique.
        let mut envs = defaults();
        envs.push(("localnet".to_string(), "deadbeef".to_string()));
        let env = resolve_environment(&envs, "custom-alias".to_string(), "deadbeef".to_string());
        assert_eq!(env.name(), "localnet");
        assert_eq!(env.id(), "deadbeef");
    }

    #[test]
    fn unregistered_network_falls_back_to_testnet() {
        // A fresh local network whose chain id isn't declared falls back to the `testnet` env key,
        // preserving the previous behavior. This is the only case where the env name and chain id
        // can disagree.
        let env = resolve_environment(&defaults(), "localnet".to_string(), "deadbeef".to_string());
        assert_eq!(env.name(), "testnet");
        assert_eq!(env.id(), "deadbeef");
    }
}
