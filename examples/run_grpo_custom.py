import importlib
import logging
import os
import pprint


class SuppressDtypeMismatchWarning(logging.Filter):
    def filter(self, record) -> bool:
        return "Dtype mismatch between HuggingFace weights and Megatron module".lower() not in record.getMessage().lower()


logging.getLogger().addFilter(SuppressDtypeMismatchWarning())

from omegaconf import OmegaConf

from examples.run_grpo import (
    _select_trainer,
    parse_args,
)
from nemo_rl.algorithms.grpo import MasterConfig, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def main() -> None:
    """Main entry point."""
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    # ============== BEGIN CUSTOM IMPORT ============== #
    # Import custom modules specified in config.
    custom_imports = config.get("custom_imports", [])
    if isinstance(custom_imports, str):
        custom_imports = [custom_imports]
    for module_name in custom_imports:
        print(f"Importing custom module: '{module_name}'...", end="")
        importlib.import_module(module_name)
        print("Done.")
    # ==============  END CUSTOM IMPORT  ============== #

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
    )

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_response_data(tokenizer, config.data, config.env)

    # Pick the policy factory at the launcher level so the legacy trainer
    # stays data-plane-agnostic (architectural invariant — see
    # tests/data_plane/unit/test_architecture_invariants.py).
    _dp_cfg = config.data_plane or {}
    if _dp_cfg.get("enabled", False):
        from nemo_rl.models.policy.tq_policy import TQPolicy

        def _make_policy(**kwargs):
            return TQPolicy(**kwargs, dp_cfg=_dp_cfg)

        _policy_factory = _make_policy
    else:
        _policy_factory = None  # setup() defaults to plain Policy

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        teacher_worker_groups,
        alias_to_group_alias,
    ) = setup(
        config,
        tokenizer,
        dataset,
        val_dataset,
        policy_factory=_policy_factory,
    )

    # Check if async mode is enabled
    if "async_grpo" in config.grpo and config.grpo["async_grpo"]["enabled"]:
        # Async GRPO does not support dynamic sampling (implementation issue with async) and reward scaling
        unsupported_features = ["use_dynamic_sampling", "reward_scaling"]

        for feature in unsupported_features:
            if feature not in config.grpo:
                continue

            if feature == "use_dynamic_sampling":
                if config.grpo[feature]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async GRPO"
                    )
            else:
                if config.grpo[feature]["enabled"]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async GRPO"
                    )

        # Async GRPO does not support multiple dataloaders
        if config.data["use_multiple_dataloader"]:
            raise NotImplementedError(
                "use_multiple_dataloader is not supported with async GRPO"
            )

        from nemo_rl.algorithms.grpo import async_grpo_train

        print("🚀 Running async GRPO training")

        async_config = config.grpo["async_grpo"]
        # Run async GRPO training
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
            teacher_worker_groups=teacher_worker_groups,
            alias_to_group_alias=alias_to_group_alias,
        )
    else:
        # Two parallel synchronous trainers (verl-style — main_ppo.py vs
        # main_ppo_sync.py). data_plane.enabled selects which one runs.
        trainer = _select_trainer(master_config)
        trainer(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
