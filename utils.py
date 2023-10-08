import io
import json
from copy import deepcopy
from transformers.utils import logging
from transformers.utils.versions import require_version

logger = logging.get_logger(__name__)



def init_deepspeed(trainer, num_training_steps, resume_from_checkpoint=None):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If ``resume_from_checkpoint`` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load

    Returns: model, optimizer, lr_scheduler

    """
    import deepspeed

    require_version("deepspeed>0.3.12")

    args = trainer.args
    ds_config_file = args.deepspeed
    model = trainer.model

    if isinstance(args.deepspeed, dict):
        # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
        # modified it, it will not be accepted here again, since some config params must be not set by users
        config = deepcopy(args.deepspeed)
    elif isinstance(args.deepspeed, str):
        with io.open(ds_config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ValueError("expecting either a path to a config file or a pre-populated dict")

    # The following code translates relevant trainer's cl args into the DS config

    # First to ensure that there is no mismatch between cl args values and presets in the config
    # file, ask to not set in ds config file:
    # - "train_batch_size",
    # - "train_micro_batch_size_per_gpu",
    # - "gradient_accumulation_steps"
    bs_keys = ["train_batch_size", "train_micro_batch_size_per_gpu"]
    if len([x for x in bs_keys if x in config.keys()]):
        raise ValueError(
            f"Do not include {bs_keys} entries in the ds config file, as they will be set via --per_device_train_batch_size or its default"
        )
    if "gradient_accumulation_steps" in config.keys():
        raise ValueError(
            "Do not include gradient_accumulation_steps entries in the ds config file, as they will be set via --gradient_accumulation_steps or its default"
        )

    # DeepSpeed does:
    #   train_batch_size = n_gpus * train_micro_batch_size_per_gpu * gradient_accumulation_steps
    # therefore we just need to set:
    config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    if "gradient_clipping" in config:
        logger.info(
            f"Keeping the `gradient_clipping` config from {ds_config_file} intact, ignoring any gradient clipping-specific cl args"
        )
    else:  # override only if the ds config doesn't already have this section
        config["gradient_clipping"] = args.max_grad_norm

    # Optimizer + Scheduler
    # Currently support combos:
    # 1. DS scheduler + DS optimizer: Yes
    # 2. HF scheduler + HF optimizer: Yes
    # 3. DS scheduler + HF optimizer: Yes
    # 4. HF scheduler + DS optimizer: No
    # Unless Offload is enabled in which case it's:
    # 1. DS scheduler + DS optimizer: Yes
    # 2. HF scheduler + HF optimizer: No
    # 3. DS scheduler + HF optimizer: No
    # 4. HF scheduler + DS optimizer: No

    optimizer = None
    if "optimizer" in config:
        logger.info(f"Updating the `scheduler` config from {ds_config_file} with other command line arguments")

        # to avoid inconsistent values of lr and warm up steps the command line args override config
        params = dict(
            lr=args.learning_rate,
            betas=[args.adam_beta1, args.adam_beta2],
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        for k, v in params.items():
            if k in config["optimizer"]["params"]:
                logger.info(f"setting optimizer.params.{k} to {v}")
                config["optimizer"]["params"][k] = v

    else:  # override only if the ds config doesn't already have this section
        if (
            "zero_optimization" in config
            and "cpu_offload" in config["zero_optimization"]
            and config["zero_optimization"]["cpu_offload"] is True
        ):
            raise ValueError("ZeRO Offload can only work with DeepSpeed optimizers")
        else:
            # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
            # But trainer uses AdamW by default.
            # To use other optimizers so using a different scheduler requires voiding warranty with: `zero_allow_untested_optimizer`
            trainer.create_optimizer()
            optimizer = trainer.optimizer
            # flag that this is non-native optimizer
            config["zero_allow_untested_optimizer"] = True

    # DS schedulers (deepspeed/runtime/lr_schedules.py):
    #
    # DS name      | --lr_scheduler_type  | HF func                           | Notes
    # -------------| ---------------------|-----------------------------------|--------------------
    # LRRangeTest  | na                   | na                                | LRRT
    # OneCycle     | na                   | na                                | 1CLR
    # WarmupLR     | constant_with_warmup | get_constant_schedule_with_warmup | w/ warmup_min_lr=0
    # WarmupDecayLR| linear               | get_linear_schedule_with_warmup   |
    lr_scheduler = None
    if "scheduler" in config:
        logger.info(f"Updating the `scheduler` config from {ds_config_file} with other command line arguments")
        # the user won't easily know the correct num_training_steps should they use WarmupDecayLR,
        # so let's set it to the correct value
        if config["scheduler"]["type"] == "WarmupDecayLR":
            logger.info(f"setting scheduler.params.total_num_steps to {num_training_steps}")
            config["scheduler"]["params"]["total_num_steps"] = num_training_steps

        # to avoid inconsistent values of lr and warmup steps the command line args override config
        params = dict(
            warmup_max_lr=args.learning_rate,
            warmup_num_steps=args.warmup_steps,
        )
        for k, v in params.items():
            if k in config["scheduler"]["params"]:
                logger.info(f"setting scheduler.params.{k} to {v}")
                config["scheduler"]["params"][k] = v

    else:  # override only if the ds config doesn't already have this section
        if "optimizer" in config:
            # to make this option work, we need to init DS optimizer first, then init HS scheduler,
            # then pass the HS scheduler to DS init, which is not possible at the moment
            raise ValueError("At the moment HF scheduler + DeepSpeed optimizer combination is not possible")
        else:
            trainer.create_scheduler(num_training_steps=num_training_steps)
            lr_scheduler = trainer.lr_scheduler

    # fp16
    if trainer.fp16_backend is not None:
        # Deepspeed has 2 possible fp16 config entries:
        # - `fp16`: for the native amp - it has a bunch of optional params but we won't set any here unless the user did the work
        # - `amp`: which delegates amp work to apex (which needs to be available), but it cannot be used with any ZeRO features, so probably best to be avoided.
        if trainer.fp16_backend == "apex":
            if "amp" in config:
                logger.info(
                    f"Keeping the `amp` config from {ds_config_file} intact, ignoring any amp-specific cl args"
                )
            else:
                config["amp"] = {
                    "enabled": True,
                    "opt_level": args.fp16_opt_level,
                }
        elif trainer.fp16_backend == "amp":
            if "fp16" in config:
                logger.info(
                    f"Keeping the `fp16` config from {ds_config_file} intact, ignoring any fp16-specific cl args"
                )
            else:
                config["fp16"] = {
                    "enabled": True,
                }

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    # init that takes part of the config via `args`, and the bulk of it via `config_params`
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config_params=config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    if resume_from_checkpoint is not None:  # and os.path.isdir(resume_from_checkpoint):
        logger.info(f"Attempting to resume from {resume_from_checkpoint}")
        # this magically updates self.optimizer and self.lr_scheduler
        load_path, _ = model.load_checkpoint(
            resume_from_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
        )
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {resume_from_checkpoint}")

    return model, optimizer, lr_scheduler