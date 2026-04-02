#!/bin/bash
#
# Train ACTSmooth policy on the cube direct manipulation dataset.
#
# Dataset: giacomoran/so101_cube_direct_manipulation (30fps, 100 episodes)
# Model:   ACTSmooth p4f2 @ 30fps
#
# Intended to run on a remote GPU machine.

set -e

DIR_BASE="/workspace"
# Wrist camera only, no state — the dataset also has observation.images.top and
# observation.state, but LeRobot's policy forward pass only uses features listed
# here, so omitting them suffices. State is excluded because in direct manipulation
# it's identical to the action.
INPUT_FEATURES='{"observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}}'

STEPS=30000
SAVE_FREQ=10000

echo "=== Training cube direct manipulation ==="

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id=giacomoran/so101_cube_direct_manipulation \
    --policy.repo_id=giacomoran/so101_cube_dm_act_smooth_p4f2 \
    --output_dir="${DIR_BASE}/so101_cube_dm_act_smooth_p4f2" \
    --policy.input_features="${INPUT_FEATURES}" \
    --steps=${STEPS} \
    --save_freq=${SAVE_FREQ} \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_vae=false \
    --policy.length_prefix_past=4 \
    --policy.length_prefix_future=2 \
    --policy.use_action_relative=true \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

echo "=== Resuming for extra checkpoints ==="

lerobot-train \
    --resume=true \
    --config_path="${DIR_BASE}/so101_cube_dm_act_smooth_p4f2/checkpoints/030000/pretrained_model/train_config.json" \
    --output_dir="${DIR_BASE}/so101_cube_dm_act_smooth_p4f2" \
    --steps=30003 \
    --save_freq=1

echo "=== Training complete ==="

# --- Compress outputs ---
echo "=== Compressing outputs ==="
cd "${DIR_BASE}"

DIR="so101_cube_dm_act_smooth_p4f2"
ARGS_TAR=""

if [ -d "${DIR}/checkpoints" ]; then
    for CKPT in $(ls -1 "${DIR}/checkpoints" | sort -n); do
        if [ -d "${DIR}/checkpoints/${CKPT}/pretrained_model" ]; then
            ARGS_TAR="${ARGS_TAR} ${DIR}/checkpoints/${CKPT}/pretrained_model"
        fi
    done
    CKPT_LATEST=$(ls -1 "${DIR}/checkpoints" | sort -n | tail -1)
    if [ -n "${CKPT_LATEST}" ] && [ -d "${DIR}/checkpoints/${CKPT_LATEST}/training_state" ]; then
        ARGS_TAR="${ARGS_TAR} ${DIR}/checkpoints/${CKPT_LATEST}/training_state"
    fi
fi

if [ -n "${ARGS_TAR}" ]; then
    tar -czvf so101_cube_dm_act_smooth_p4f2.tar.gz ${ARGS_TAR}
    echo "Done! Archive: ${DIR_BASE}/so101_cube_dm_act_smooth_p4f2.tar.gz"
else
    echo "ERROR: No checkpoints found to compress!"
    exit 1
fi

# To transfer and uncompress on local machine:
#   Remote:  croc send so101_cube_dm_act_smooth_p4f2.tar.gz
#   Local:   croc <code>
#            tar -xzvf so101_cube_dm_act_smooth_p4f2.tar.gz -C /path/to/destination
