#!/usr/bin/env python3
"""PyTorch profiler script for analyzing training performance bottlenecks.

This script profiles the training loop to identify where time is being spent.
It runs training using lerobot-train under the hood to ensure proper config.

Usage:
    python profile_training.py --policy-type act_relative_rtc_2 --output outputs/tmp/profile_rtc2
    python profile_training.py --policy-type act --output outputs/tmp/profile_act
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def profile_training(
    policy_type: str,
    output_dir: Path,
    num_warmup_steps: int = 5,
    num_profile_steps: int = 15,
):
    """Profile training loop by importing and running lerobot training with profiler."""
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.utils import cycle
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    logger.info(f"Profiling {policy_type} training...")
    logger.info(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Enable CUDA optimizations (same as lerobot training)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = "cuda"
    batch_size = 8

    # Build CLI args for lerobot config parser
    base_args = [
        f"--policy.type={policy_type}",
        "--dataset.repo_id=giacomoran/so101_data_collection_cube_hand_guided_1x224x8",
        "--dataset.episodes=[0]",
        "--steps=50",
        f"--batch_size={batch_size}",
        "--policy.optimizer_lr=3e-5",
        "--policy.optimizer_lr_backbone=3e-5",
        "--policy.chunk_size=8",
        "--policy.n_action_steps=8",
        "--policy.use_vae=false",
        "--policy.n_decoder_layers=4",
        "--policy.vision_backbone=resnet34",
        "--policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1",
        "--policy.pre_norm=true",
        f"--policy.device={device}",
        '--policy.input_features={"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 224, 224], "type": "VISUAL"}}',
        "--wandb.enable=false",
        f"--output_dir={output_dir / 'train_output'}",
        "--policy.push_to_hub=false",
    ]

    if policy_type == "act_relative_rtc_2":
        base_args.extend(
            [
                "--policy.downscale_img_square=224",
                "--policy.obs_state_delta_frames=1",
                "--policy.rtc_max_delay=3",
            ]
        )

    # Register third-party plugins (for custom policy types)
    from lerobot.scripts.lerobot_train import register_third_party_plugins

    register_third_party_plugins()

    # Parse config using draccus (what lerobot uses internally)
    import draccus

    old_argv = sys.argv
    sys.argv = ["profile_training.py"] + base_args
    try:
        cfg = draccus.parse(TrainPipelineConfig)
    finally:
        sys.argv = old_argv

    # Create dataset
    logger.info("Creating dataset...")
    dataset = make_dataset(cfg)

    # Create policy
    logger.info("Creating policy...")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    policy = policy.to(device)
    policy.train()

    # Create preprocessor
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=None,
        dataset_stats=dataset.meta.stats,
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Create optimizer
    if hasattr(policy, "get_optim_params"):
        optimizer = torch.optim.AdamW(policy.get_optim_params(), weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-5, weight_decay=1e-4)

    # Create data iterator
    dl_iter = cycle(dataloader)

    def train_step():
        batch = next(dl_iter)
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    # Warmup
    logger.info(f"Running {num_warmup_steps} warmup steps...")
    for i in range(num_warmup_steps):
        loss = train_step()
        if i == 0:
            logger.info(f"  Step {i + 1}: loss = {loss:.4f}")

    # Synchronize before profiling
    torch.cuda.synchronize()

    # Profile with PyTorch profiler
    logger.info(f"Profiling {num_profile_steps} steps...")

    wait_steps = 1
    warmup_prof_steps = 2
    active_steps = num_profile_steps

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait_steps, warmup=warmup_prof_steps, active=active_steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step in range(wait_steps + warmup_prof_steps + active_steps):
            train_step()
            prof.step()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info(f"PROFILER SUMMARY - {policy_type}")
    logger.info("=" * 80)

    # Self CUDA time (actual GPU work, excludes child ops)
    logger.info("\n--- Top 25 operations by Self CUDA time ---")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=25,
        )
    )

    # CPU time sorted (includes Python overhead)
    logger.info("\n--- Top 25 operations by CPU time ---")
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=25,
        )
    )

    # Note: tensorboard_trace_handler already saves the trace
    logger.info(f"\nTrace files saved to: {output_dir}")

    logger.info(f"\nTensorBoard traces saved to: {output_dir}")
    logger.info(f"View with: tensorboard --logdir={output_dir}")

    return prof


def main():
    parser_arg = argparse.ArgumentParser(description="Profile LeRobot training")
    parser_arg.add_argument(
        "--policy-type",
        type=str,
        required=True,
        choices=["act", "act_relative_rtc_2"],
        help="Policy type to profile",
    )
    parser_arg.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for profiler results",
    )
    parser_arg.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps before profiling",
    )
    parser_arg.add_argument(
        "--profile-steps",
        type=int,
        default=15,
        help="Number of steps to profile",
    )
    args = parser_arg.parse_args()

    profile_training(
        policy_type=args.policy_type,
        output_dir=args.output,
        num_warmup_steps=args.warmup_steps,
        num_profile_steps=args.profile_steps,
    )


if __name__ == "__main__":
    main()
