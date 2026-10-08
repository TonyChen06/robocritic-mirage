# Five-frame mirage

`ROBOCRITIC_MIRAGE=1` blends frames at offsets `16,12,8,4,0` with weights
`.04,.08,.12,.16,.60`. Early history repeats frame zero. Default `0` is control.
The shared loader applies this during training and ranking.

From the repo root, using your existing critic environment and 250 OpenDrawer rollouts
(50 initial states, policy seeds 0-4):

```bash
export PRETRAIN_ROOT=/path/to/open_drawer/evals/pretrain
export QWEN_BASE=/path/to/Qwen2.5-VL-3B-Instruct
ROBOCRITIC_MIRAGE=0 bash examples/scripts/robocasa_vlm/mirage.sh
ROBOCRITIC_MIRAGE=1 bash examples/scripts/robocasa_vlm/mirage.sh
```

Evaluate each saved best checkpoint on all held-out pairs:

```bash
for mode in 0 1; do
  CUDA_VISIBLE_DEVICES=0 ROBOCRITIC_MIRAGE=$mode python examples/scripts/robocasa_vlm/sft_vlm_robocasa.py \
    --model_name_or_path "outputs/opendrawer-$mode" --output_dir "results/opendrawer-$mode" \
    --pretrain_root "$PRETRAIN_ROOT" --task OpenDrawer --num_eval_episodes 8 \
    --compare_interval 4,8,12,16 --train_sample_interval 4 --failure_last_frac 0.95 \
    --failure_min_frames 8 --max_succ_per_fail 1 --subsample 1 --balance_fail_vs_succ True \
    --max_pixels 848x480 --eval_only true
done
```

Results go to `results/opendrawer-{0,1}/accuracy.json`. Use the matching
`ROBOCRITIC_MIRAGE` value when running `rank_serve_robocasa.py`.
