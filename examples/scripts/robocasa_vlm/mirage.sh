#!/usr/bin/env bash
set -euo pipefail
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 2 --offload_optimizer_device cpu \
  examples/scripts/robocasa_vlm/sft_vlm_robocasa.py \
  --model_name_or_path "$QWEN_BASE" --dtype bfloat16 --attn_implementation flash_attention_2 \
  --pretrain_root "$PRETRAIN_ROOT" --task OpenDrawer --num_eval_episodes 8 --output_dir "outputs/opendrawer-${ROBOCRITIC_MIRAGE:-0}" \
  --eval_strategy steps --logging_steps 1 --eval_steps 15 --save_steps 15 --save_total_limit 2 \
  --max_steps 334 --learning_rate 1.5e-5 --lr_scheduler_type linear --warmup_steps 15 \
  --weight_decay 0 --max_grad_norm 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 12 --gradient_checkpointing true \
  --compare_interval 4,8,12,16 --train_sample_interval 4 --failure_last_frac 0.95 \
  --failure_min_frames 8 --max_succ_per_fail 1 --subsample 1 --eval_max_pairs 100 \
  --max_pixels 848x480 --balance_fail_vs_succ True --seed 42 --data_seed 42 --report_to none
