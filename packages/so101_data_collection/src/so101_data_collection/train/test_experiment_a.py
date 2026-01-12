#!/usr/bin/env python
"""Test script to verify Experiment A setup before training.

This script performs sanity checks:
1. Load pretrained model
2. Initialize prefix-conditioned model
3. Create a dummy batch
4. Test forward pass with different prefix modes
5. Verify gradients flow correctly
6. Test loss masking

Usage:
    python test_experiment_a.py
"""

import sys
import traceback
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.utils.constants import ACTION, OBS_STATE

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from so101_data_collection.train.run_experiment_a import (
    ExperimentAConfig,
    freeze_model,
    load_pretrained_policy,
)


def test_model_loading():
    """Test loading of pretrained model."""
    print("=" * 60)
    print("TEST 1: MODEL LOADING")
    print("=" * 60)

    config = ExperimentAConfig(
        prefix_mode="encoder_input",
        pretrained_path="outputs/cube_hand_guided_act_umi_wrist_10_30k/pretrained_model",
    )

    try:
        policy = load_pretrained_policy(
            config.pretrained_path, config.prefix_mode, device="cpu"
        )
        print("✓ Model loaded successfully")
        print(f"  Device: {next(policy.parameters()).device}")
        print(f"  Prefix mode: {policy.config.prefix_mode}")

        return policy
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return None


def test_parameter_counting(policy):
    """Test that parameters are frozen correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: PARAMETER FREEZING")
    print("=" * 60)

    freeze_model(policy, freeze_backbone=True, freeze_encoder=True, freeze_decoder=True)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    action_prefix_proj_trainable = any(
        p.requires_grad for p in policy.model.action_prefix_proj.parameters()
    )
    print(f"✓ action_prefix_proj trainable: {action_prefix_proj_trainable}")

    backbone_trainable = any(
        p.requires_grad for p in policy.model.backbone.parameters()
    )
    print(f"✓ backbone frozen: {not backbone_trainable}")

    encoder_trainable = any(p.requires_grad for p in policy.model.encoder.parameters())
    print(f"✓ encoder frozen: {not encoder_trainable}")

    decoder_trainable = sum(
        p.numel() for p in policy.model.decoder.parameters() if p.requires_grad
    )
    print(f"Decoder trainable params: {decoder_trainable:,}")

    some_frozen = trainable_params < total_params
    assert some_frozen, "Expected some parameters to be frozen!"
    return some_frozen


def test_forward_pass(policy):
    """Test forward pass with prefix conditioning."""
    print("\n" + "=" * 60)
    print("TEST 3: FORWARD PASS")
    print("=" * 60)

    action_dim = policy.config.action_feature.shape[0]
    state_dim = policy.config.robot_state_feature.shape[0]
    chunk_size = policy.config.chunk_size
    batch_size = 2

    print(f"Action dim: {action_dim}")
    print(f"State dim: {state_dim}")
    print(f"Chunk size: {chunk_size}")

    batch = {
        OBS_STATE: torch.randn(batch_size, 2, state_dim),
        ACTION: torch.randn(batch_size, chunk_size, action_dim),
        "action_is_pad": torch.zeros(batch_size, chunk_size, dtype=torch.bool),
    }

    if policy.config.image_features:
        for key, feat in policy.config.input_features.items():
            if feat.type == FeatureType.VISUAL:
                img_shape = feat.shape
                batch[key] = torch.randn(batch_size, *img_shape)

    for delay in [0, 1, 2, 3]:
        try:
            loss, loss_dict = policy.forward(batch, delay=delay)
            print(f"\n✓ Forward pass with delay={delay}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Postfix error: {loss_dict.get('l1_loss_postfix', 0.0):.6f}")
            print(f"  Prefix error: {loss_dict.get('l1_loss_prefix', 0.0):.6f}")
            print(f"  Boundary diff: {loss_dict.get('boundary_diff', 0.0):.6f}")

            assert not torch.isnan(loss), f"Loss is NaN for delay={delay}"
            assert not torch.isinf(loss), f"Loss is inf for delay={delay}"

        except Exception as e:
            print(f"✗ Forward pass failed for delay={delay}: {e}")
            traceback.print_exc()
            return False

    return True


def test_backward_pass(policy):
    """Test backward pass and gradient flow."""
    print("\n" + "=" * 60)
    print("TEST 4: BACKWARD PASS")
    print("=" * 60)

    action_dim = policy.config.action_feature.shape[0]
    state_dim = policy.config.robot_state_feature.shape[0]
    chunk_size = policy.config.chunk_size
    batch_size = 2

    batch = {
        OBS_STATE: torch.randn(batch_size, 2, state_dim),
        ACTION: torch.randn(batch_size, chunk_size, action_dim),
        "action_is_pad": torch.zeros(batch_size, chunk_size, dtype=torch.bool),
    }

    if policy.config.image_features:
        for key, feat in policy.config.input_features.items():
            if feat.type == FeatureType.VISUAL:
                img_shape = feat.shape
                batch[key] = torch.randn(batch_size, *img_shape)

    freeze_model(policy, freeze_backbone=True, freeze_encoder=True, freeze_decoder=True)
    policy.train()

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-4
    )

    delay = 2
    try:
        loss, loss_dict = policy.forward(batch, delay=delay)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("✓ Backward pass successful")
        print(f"  Loss: {loss.item():.6f}")

        action_prefix_proj_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.model.action_prefix_proj.parameters()
        )
        print(f"✓ action_prefix_proj has gradients: {action_prefix_proj_has_grad}")

        frozen_params_no_grad = all(
            p.grad is None or p.grad.abs().sum() == 0
            for p in policy.model.backbone.parameters()
        )
        print(f"✓ backbone has no gradients: {frozen_params_no_grad}")

        return True

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        traceback.print_exc()
        return False


def test_all_prefix_modes():
    """Test all three prefix modes and vanilla mode."""
    print("\n" + "=" * 60)
    print("TEST 5: ALL PREFIX MODES")
    print("=" * 60)

    for prefix_mode in ["vanilla", "decoder_pos", "encoder_input", "encoder_output"]:
        print(f"\nTesting {prefix_mode}...")

        config = ExperimentAConfig(
            prefix_mode=prefix_mode,
            pretrained_path="outputs/cube_hand_guided_act_umi_wrist_10_30k/pretrained_model",
        )

        try:
            policy = load_pretrained_policy(
                config.pretrained_path, prefix_mode, device="cpu"
            )
            freeze_model(
                policy, freeze_backbone=True, freeze_encoder=True, freeze_decoder=True
            )

            if policy.config.action_feature is None:
                print(f"✗ {prefix_mode} failed: action_feature is None in config")
                return False
            if policy.config.robot_state_feature is None:
                print(f"✗ {prefix_mode} failed: robot_state_feature is None in config")
                return False

            action_dim = policy.config.action_feature.shape[0]
            state_dim = policy.config.robot_state_feature.shape[0]
            chunk_size = policy.config.chunk_size
            batch_size = 2

            batch = {
                OBS_STATE: torch.randn(batch_size, 2, state_dim),
                ACTION: torch.randn(batch_size, chunk_size, action_dim),
                "action_is_pad": torch.zeros(batch_size, chunk_size, dtype=torch.bool),
            }

            if policy.config.image_features:
                for key, feat in policy.config.input_features.items():
                    if feat.type == FeatureType.VISUAL:
                        img_shape = feat.shape
                        batch[key] = torch.randn(batch_size, *img_shape)

            delay = 0 if prefix_mode == "vanilla" else 2
            loss, loss_dict = policy.forward(batch, delay=delay)
            print(f"✓ {prefix_mode} works")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Postfix error: {loss_dict.get('l1_loss_postfix', 0.0):.6f}")

        except Exception as e:
            print(f"✗ {prefix_mode} failed: {e}")
            traceback.print_exc()
            return False

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EXPERIMENT A: SETUP VERIFICATION")
    print("=" * 60)

    model = test_model_loading()
    if model is None:
        print("\n✗ TEST 1 FAILED: Cannot continue")
        return False

    if not test_parameter_counting(model):
        print("\n✗ TEST 2 FAILED")
        return False

    if not test_forward_pass(model):
        print("\n✗ TEST 3 FAILED")
        return False

    if not test_backward_pass(model):
        print("\n✗ TEST 4 FAILED")
        return False

    if not test_all_prefix_modes():
        print("\n✗ TEST 5 FAILED")
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nSetup is ready for training!")
    print("\nTo start training with prefix conditioning, run:")
    print("  python run_experiment_a.py --prefix_mode encoder_input")
    print("\nTo train vanilla baseline, run:")
    print("  python run_experiment_a.py --prefix_mode vanilla")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
