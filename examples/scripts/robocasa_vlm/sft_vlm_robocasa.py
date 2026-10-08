"""
Generalizable trainer for robocasa policy-rollout pairwise progress comparison.

Expected layout at --pretrain_root (one subdir per rollout):
    <TASK>_<...>_seed<seed>/eval_log.json   # per-episode success + metadata
    <TASK>_<...>_seed<seed>/media/seed1000XX.mp4

eval_log.json fields we use:
    eval_args.task                       # task name, matched against --task
    test/sim_max_reward_1000XX  = 1.0/0.0  # episode 1000XX success / failure
The mp4 path is reconstructed as <rollout_dir>/media/seed1000XX.mp4 (the
test/sim_video_* paths stored in the json are stale).

Initial conditions: the env reset seed is fixed across rollouts, so episode index
E (100000..100049) denotes the SAME initial condition in every rollout. The same E
succeeds in some rollouts and fails in others, which gives failure<->success pairs
that share an initial condition.

Pairing (mirrors sft_vlm_stacking, single camera):
- success-vs-success: same success video, later sub-frame = more progress.
- failure-vs-success: LAST K sub-frames of a failure video at episode E vs the
  temporally aligned sub-frames of up to --max_succ_per_fail success videos at the
  SAME episode E (same initial condition).

Frames are decoded on the fly from the mp4 via decord (extract_frame), so there is
no on-disk conversion step. 
"""

import gc
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt, gridspec
from typing import List

import torch
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

# Sibling imports work regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def visualize_dataset(combined_data, output_dir, split_name="train", num_examples=20, max_pixels=None):
    """
    Create statistics plots and sample overlay images for the dataset.

    Saves to {output_dir}/dataset_viz/:
      - dataset_statistics_{split}.png: answer distribution, demo type, per-demo counts, per-task counts
      - sample_overlays_{split}.png: grid of example side-by-side overlays with metadata
      - vlm_view_{split}.png: same samples rendered at the post-`smart_resize` resolution the VLM actually sees (only if `max_pixels` is provided).
      - dataset_stats_{split}.json: machine-readable statistics

    Args:
        combined_data: List of metadata dicts (with video_path_1/2, frame_idx_1/2, etc.)
        output_dir: Directory to save outputs.
        split_name: Label for the split (e.g., "train", "val").
        num_examples: Number of sample overlay images to render.
        max_pixels: Qwen2.5-VL processor pixel budget per image (H*W). If set,
            an additional `vlm_view_{split}.png` is written showing each sampled
            frame after `smart_resize` (28-aligned, aspect-ratio-preserving).
    """
    if len(combined_data) == 0:
        logger.warning("No data to visualize")
        return

    viz_dir = Path(output_dir) / "dataset_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Extract metadata
    correct_answers = [item.get("correct_answer", 0) for item in combined_data]
    demo_types = [item.get("demo_success", "unknown") for item in combined_data]
    demo_ids = [item.get("demo_id", "unknown") for item in combined_data]
    job_names = [item.get("job_name", "unknown") for item in combined_data]
    task_tokens = [item.get("task_token", "unknown") for item in combined_data]

    # ---- Figure 1: Dataset Statistics ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Dataset Statistics ({split_name}, n={len(combined_data)})", fontsize=14, fontweight="bold")

    # 1a. Answer distribution
    ax = axes[0, 0]
    unique_answers = sorted(set(correct_answers))
    answer_counts = [correct_answers.count(a) for a in unique_answers]
    colors = ["#2ecc71" if a > 0 else "#e74c3c" if a < 0 else "#95a5a6" for a in unique_answers]
    ax.bar([str(a) for a in unique_answers], answer_counts, color=colors)
    ax.set_xlabel("Correct Answer")
    ax.set_ylabel("Count")
    ax.set_title("Answer Distribution")

    # 1b. Success vs Failure
    ax = axes[0, 1]
    type_counts = {t: demo_types.count(t) for t in set(demo_types)}
    colors_pie = ["#2ecc71" if "success" in t.lower() else "#e74c3c" for t in type_counts.keys()]
    ax.pie(type_counts.values(), labels=type_counts.keys(), autopct="%1.1f%%", colors=colors_pie)
    ax.set_title("Demo Type Distribution")
    # 1c. Samples per demo_id
    ax = axes[1, 0]
    demo_id_counts = defaultdict(int)
    for d in demo_ids:
        demo_id_counts[d] += 1
    sorted_ids = sorted(demo_id_counts.keys())
    ax.bar(range(len(sorted_ids)), [demo_id_counts[d] for d in sorted_ids], color="#3498db")
    ax.set_xlabel("Demo ID")
    ax.set_ylabel("Sample Count")
    ax.set_title(f"Samples per Demo ({len(sorted_ids)} demos)")
    if len(sorted_ids) > 20:
        ax.set_xticks([])
    else:
        ax.set_xticks(range(len(sorted_ids)))
        ax.set_xticklabels(sorted_ids, rotation=45)

    # 1d. Samples per task
    ax = axes[1, 1]
    task_counts = defaultdict(int)
    for t in task_tokens:
        task_counts[t] += 1
    sorted_tasks = sorted(task_counts.keys())
    ax.barh(range(len(sorted_tasks)), [task_counts[t] for t in sorted_tasks], color="#9b59b6")
    ax.set_yticks(range(len(sorted_tasks)))
    ax.set_yticklabels(sorted_tasks, fontsize=8)
    ax.set_xlabel("Sample Count")
    ax.set_title("Samples per Task")

    plt.tight_layout()
    stats_path = viz_dir / f"dataset_statistics_{split_name}.png"
    plt.savefig(stats_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved statistics plot to {stats_path}")

    # ---- Figure 2: Sample Overlay Images ----
    num_to_show = min(num_examples, len(combined_data))

    # Sample diverse examples (mix success/failure and positive/negative answers)
    indices_by_type = defaultdict(list)
    for i, item in enumerate(combined_data):
        key = (item.get("demo_success", "unknown"), item.get("correct_answer", 0) > 0)
        indices_by_type[key].append(i)

    sampled_indices = []
    rng = random.Random(42)
    for key, indices in indices_by_type.items():
        n_sample = max(1, num_to_show // len(indices_by_type))
        sampled_indices.extend(rng.sample(indices, min(n_sample, len(indices))))
    sampled_indices = sampled_indices[:num_to_show]
    if len(sampled_indices) == 0:
        sampled_indices = list(range(num_to_show))

    n_cols = min(3, num_to_show)
    n_rows = (num_to_show + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    for plot_idx, data_idx in enumerate(sampled_indices):
        item = combined_data[data_idx]
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)

        try:
            frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
            frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
            overlay = create_side_by_side(frame1, frame2)
            ax.imshow(overlay)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load:\n{e}", ha="center", va="center", transform=ax.transAxes)

        ax.axis("off")
        answer = item.get("correct_answer", "?")
        demo_type = item.get("demo_success", "?")
        task = item.get("task_token", "?")
        title_color = "#2ecc71" if answer > 0 else "#e74c3c" if answer < 0 else "#333333"
        ax.set_title(f"Answer: {answer} | {demo_type} | {task}", fontsize=9, color=title_color, fontweight="bold")

    plt.suptitle(f"Sample Overlays ({split_name})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    samples_path = viz_dir / f"sample_overlays_{split_name}.png"
    plt.savefig(samples_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sample overlays to {samples_path}")

    # ---- Figure 3: Detailed view with original frame pairs ----
    num_detailed = min(4, len(sampled_indices))
    fig = plt.figure(figsize=(15, 4 * num_detailed))
    gs = gridspec.GridSpec(num_detailed, 3, width_ratios=[1, 1, 2], hspace=0.3, wspace=0.1)

    for row_idx in range(num_detailed):
        item = combined_data[sampled_indices[row_idx]]

        try:
            frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
            frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
            overlay = create_side_by_side(frame1, frame2)
        except Exception:
            frame1 = frame2 = overlay = None

        ax1 = fig.add_subplot(gs[row_idx, 0])
        if frame1:
            ax1.imshow(frame1)
        ax1.axis("off")
        ax1.set_title("Left Frame", fontsize=9)

        ax2 = fig.add_subplot(gs[row_idx, 1])
        if frame2:
            ax2.imshow(frame2)
        ax2.axis("off")
        ax2.set_title("Right Frame", fontsize=9)

        ax3 = fig.add_subplot(gs[row_idx, 2])
        if overlay:
            ax3.imshow(overlay)
        ax3.axis("off")
        answer = item.get("correct_answer", "?")
        demo_type = item.get("demo_success", "?")
        task = item.get("task_token", "?")
        title_color = "#2ecc71" if answer > 0 else "#e74c3c" if answer < 0 else "#333333"
        ax3.set_title(
            f"Overlay | Answer: {answer} | {demo_type} | {task}",
            fontsize=10, color=title_color, fontweight="bold",
        )

    plt.suptitle(f"Detailed: Original Frames + Overlay ({split_name})", fontsize=12, fontweight="bold")
    detailed_path = viz_dir / f"detailed_examples_{split_name}.png"
    plt.savefig(detailed_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved detailed examples to {detailed_path}")

    # ---- Figure 4: What the VLM actually sees (post smart_resize) ----
    # Mirror Qwen2.5-VL's image processor: smart_resize snaps H,W to multiples
    # of 28 (patch_size*merge_size) while preserving aspect ratio, bounded by
    # max_pixels. The result is the exact pixel buffer the model is fed.
    if max_pixels is not None:
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        except Exception as e:
            logger.warning(f"Could not import smart_resize for VLM view: {e}")
        else:
            n_vlm = min(num_examples, len(sampled_indices))
            n_cols_v = 2  # frame1 | frame2
            fig = plt.figure(figsize=(6 * n_cols_v, 3 * n_vlm))
            for row_idx in range(n_vlm):
                item = combined_data[sampled_indices[row_idx]]
                try:
                    f1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
                    f2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
                    h1_new, w1_new = smart_resize(f1.height, f1.width, max_pixels=max_pixels)
                    h2_new, w2_new = smart_resize(f2.height, f2.width, max_pixels=max_pixels)
                    v1 = f1.resize((w1_new, h1_new), Image.BICUBIC)
                    v2 = f2.resize((w2_new, h2_new), Image.BICUBIC)
                except Exception as e:
                    v1 = v2 = None
                    h1_new = w1_new = h2_new = w2_new = 0
                    orig1 = orig2 = ""
                    err = str(e)
                else:
                    orig1 = f"{f1.width}x{f1.height}"
                    orig2 = f"{f2.width}x{f2.height}"
                    err = ""

                ax1 = fig.add_subplot(n_vlm, 2, row_idx * 2 + 1)
                if v1 is not None:
                    ax1.imshow(v1, interpolation="nearest")
                elif err:
                    ax1.text(0.5, 0.5, err, ha="center", va="center", transform=ax1.transAxes)
                ax1.axis("off")
                ax1.set_title(f"frame_1: {orig1} -> {w1_new}x{h1_new}", fontsize=9)

                ax2 = fig.add_subplot(n_vlm, 2, row_idx * 2 + 2)
                if v2 is not None:
                    ax2.imshow(v2, interpolation="nearest")
                ax2.axis("off")
                answer = item.get("correct_answer", "?")
                ax2.set_title(
                    f"frame_2: {orig2} -> {w2_new}x{h2_new} | ans={answer}",
                    fontsize=9,
                )

            plt.suptitle(
                f"VLM view ({split_name}) - smart_resize @ max_pixels={max_pixels}",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            vlm_path = viz_dir / f"vlm_view_{split_name}.png"
            plt.savefig(vlm_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved VLM view to {vlm_path}")

    # ---- Save JSON stats ----
    stats_summary = {
        "split": split_name,
        "total_samples": len(combined_data),
        "answer_distribution": {str(k): v for k, v in zip(unique_answers, answer_counts)},
        "demo_type_distribution": dict(type_counts),
        "num_unique_demo_ids": len(set(demo_ids)),
        "task_distribution": dict(task_counts),
    }
    stats_json_path = viz_dir / f"dataset_stats_{split_name}.json"
    with open(stats_json_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    logger.info(f"Saved stats JSON to {stats_json_path}")

    return viz_dir

TASK_TOKENS = {
    "PnPCounterToCab": "[COUNTER_TO_CAB]",
    "PnPCabToCounter": "[CAB_TO_COUNTER]",
    "PnPCounterToMicrowave": "[COUNTER_TO_MICROWAVE]",
    "PnPMicrowaveToCounter": "[MICROWAVE_TO_COUNTER]",
    "PnPStoveToCounter": "[STOVE_TO_COUNTER]",
    "PnPCounterToStove": "[COUNTER_TO_STOVE]",
    "PnPCounterToSink": "[COUNTER_TO_SINK]",
    "PnPSinkToCounter": "[SINK_TO_COUNTER]",
    "PnPCoffeeServeMug": "[COFFEE_SERVE_MUG]",
    "PnPRedLegoToBrownBowl": "[PNP_RED_LEGO_TO_BROWN_BOWL]",
    "CoffeeServeMug": "[COFFEE_SERVE_MUG]",
    "PnPCloseDrawer": "[CLOSE_DRAWER]",
    "CoffeeSetupMug": "[COFFEE_SETUP_MUG]",
    "CoffeePressButton": "[COFFEE_PRESS_BUTTON]",
    "CloseDrawer": "[CLOSE_DRAWER]",
    "OpenDrawer": "[OPEN_DRAWER]",
    "CloseSingleDoor": "[CLOSE_SINGLE_DOOR]",
    "CloseDoubleDoor": "[CLOSE_DOUBLE_DOOR]",
    "OpenDoubleDoor": "[OPEN_DOUBLE_DOOR]",
    "OpenSingleDoor": "[OPEN_SINGLE_DOOR]",
    "TurnSinkSpout": "[TURN_SINK_SPOUT]",
    # Realworld dataset tasks
    "PutKiwiInCenterOfTable": "[PUT_KIWI_IN_CENTER_OF_TABLE]",
    "PushCoasterToMug": "[PUSH_COASTER_TO_MUG]",
    "TurnMugRightsideUp": "[TURN_MUG_RIGHTSIDE_UP]",
    "BimanualBikeRotorInstall": "[BIKE_ROTOR_INSTALL]",
    "BimanualClearKitchenCounter": "[CLEAR_KITCHEN_COUNTER]",
    "BimanualSetUpBreakfastTable": "[SETUP_BREAKFAST_TABLE]",
    "CleanLitterBox": "[CLEAN_LITTERBOX]",
    "CutAppleIntoSlices": "[CUT_APPLE]",
    "UprightBottle": "[UPRIGHT_BOTTLE]",
    "BagPlate": "[BAG_PLATE]",
    "PushBowl": "[PUSH_BOWL]",
    "Stacking": "[STACKING]",
}
@dataclass
class TwoImageCollator(DataCollatorForVisionLanguageModeling):
    """DataCollatorForVisionLanguageModeling that resolves two video frames per
    example and provides them as a list of two PIL images. The base class's
    prepare_multimodal_messages will inject both image placeholders into the
    first user message automatically.
    """

    def _collate_language_modeling(self, examples):
        for example in examples:
            f1 = extract_frame(example["video_path_1"], example["frame_idx_1"])
            f2 = extract_frame(example["video_path_2"], example["frame_idx_2"])
            example["images"] = [f1, f2]
        return super()._collate_language_modeling(examples)

    def _collate_prompt_completion(self, examples):
        for example in examples:
            f1 = extract_frame(example["video_path_1"], example["frame_idx_1"])
            f2 = extract_frame(example["video_path_2"], example["frame_idx_2"])
            example["images"] = [f1, f2]
        output = super()._collate_prompt_completion(examples)
        # TRL's _collate_prompt_completion concatenates prompt+completion
        # input_ids/attention_mask but leaves Qwen2.5-VL's per-token
        # `mm_token_type_ids` at the prompt length, which crashes
        # get_rope_index. Append zeros (text type) for the completion.
        if "mm_token_type_ids" in output:
            mm = output["mm_token_type_ids"]
            seq_len = output["input_ids"].shape[1]
            if mm.shape[1] < seq_len:
                pad = torch.zeros(
                    (mm.shape[0], seq_len - mm.shape[1]),
                    dtype=mm.dtype, device=mm.device,
                )
                output["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)
            elif mm.shape[1] > seq_len:
                output["mm_token_type_ids"] = mm[:, :seq_len]
        return output


class PairwiseSignAccuracyCallback(TrainerCallback):
    """Run model.generate on a sample of the eval dataset, parse the signed
    integer in each response, and log per-bucket sign accuracy + a few
    qualitative examples. Distributed-aware: each rank handles its slice and
    counts are aggregated via dist.all_gather_object.
    """

    SIGNED_INT_RE = re.compile(r"-?\d+")

    def __init__(self, eval_dataset, processor, max_pairs=200, n_qualitative=3):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.max_pairs = max_pairs
        self.n_qualitative = n_qualitative

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_dataset is None or len(self.eval_dataset) == 0 or self.processor is None:
            return
        model = kwargs.get("model")
        if model is None:
            return

        try:
            import torch.distributed as dist
            world = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            world, rank = 1, 0

        # Reproducible subset that doesn't depend on step (so trends are comparable).
        rng = random.Random(1234)
        all_indices = list(range(len(self.eval_dataset)))
        rng.shuffle(all_indices)
        n = min(self.max_pairs, len(all_indices))
        all_indices = all_indices[:n]

        my_indices = all_indices[rank::world]
        buckets = {}  # bucket -> {correct, total, unparsed}
        qualitative = []

        was_training = model.training
        model.eval()
        try:
            for i in my_indices:
                try:
                    ex = self.eval_dataset[i]
                    bucket = ex.get("bucket", "unknown")
                    target_ans = int(ex["correct_answer"])
                    task_token = ex.get("task_token", "[UNKNOWN]")
                    user_text = USER_PROMPT_TEMPLATE.format(task_token=task_token)

                    f1 = extract_frame(ex["video_path_1"], ex["frame_idx_1"])
                    f2 = extract_frame(ex["video_path_2"], ex["frame_idx_2"])

                    mm_messages = [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f1},
                                {"type": "image", "image": f2},
                                {"type": "text", "text": user_text},
                            ],
                        },
                    ]
                    text = self.processor.apply_chat_template(
                        mm_messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text],
                        images=[[f1, f2]],
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                    gen = self.processor.batch_decode(
                        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    )[0].strip()

                    m = self.SIGNED_INT_RE.search(gen)
                    pred_ans = int(m.group(0)) if m else None

                    stats = buckets.setdefault(bucket, {"correct": 0, "total": 0, "unparsed": 0})
                    stats["total"] += 1
                    if pred_ans is None:
                        stats["unparsed"] += 1
                    elif (pred_ans > 0) == (target_ans > 0):
                        stats["correct"] += 1

                    if rank == 0 and len(qualitative) < self.n_qualitative:
                        qualitative.append({
                            "target": target_ans,
                            "pred_text": gen,
                            "bucket": bucket,
                            "demo": ex.get("demo_id_exact", "?"),
                        })
                except Exception as e:
                    logger.warning(f"PairwiseSignAccuracyCallback: example {i} failed: {e}")

            if world > 1:
                gathered = [None] * world
                try:
                    import torch.distributed as dist
                    dist.all_gather_object(gathered, buckets)
                    merged = {}
                    for b in gathered:
                        for k, v in b.items():
                            m = merged.setdefault(k, {"correct": 0, "total": 0, "unparsed": 0})
                            for kk in m:
                                m[kk] += v.get(kk, 0)
                    buckets = merged
                except Exception as e:
                    logger.warning(f"all_gather_object failed: {e}")

            total = sum(b["total"] for b in buckets.values())
            correct = sum(b["correct"] for b in buckets.values())
            unparsed = sum(b["unparsed"] for b in buckets.values())
            overall = correct / max(total, 1)

            # Per-camera aggregates: bucket names end with "_cam0" / "_cam1".
            def _agg(suffix):
                cs = [b for k, b in buckets.items() if k.endswith(suffix)]
                tot = sum(b["total"] for b in cs)
                cor = sum(b["correct"] for b in cs)
                unp = sum(b["unparsed"] for b in cs)
                return cor / max(tot, 1), unp / max(tot, 1), tot
            cam0_acc, cam0_unp, cam0_n = _agg("_cam0")
            cam1_acc, cam1_unp, cam1_n = _agg("_cam1")

            # Inject into Trainer's metrics dict so metric_for_best_model can find it.
            metrics = kwargs.get("metrics")
            if metrics is not None:
                metrics["eval_sign_acc_overall"] = overall
                metrics["eval_unparsed_overall"] = unparsed / max(total, 1)
                metrics["eval_sign_acc_cam0"] = cam0_acc
                metrics["eval_sign_acc_cam1"] = cam1_acc

            if rank == 0:
                logs = {
                    "eval/sign_acc_overall": overall,
                    "eval/unparsed_overall": unparsed / max(total, 1),
                    "eval/sign_acc_cam0": cam0_acc,
                    "eval/sign_acc_cam1": cam1_acc,
                    "eval/unparsed_cam0": cam0_unp,
                    "eval/unparsed_cam1": cam1_unp,
                    "eval/n_cam0": cam0_n,
                    "eval/n_cam1": cam1_n,
                }
                for k in sorted(buckets):
                    b = buckets[k]
                    logs[f"eval/sign_acc_{k}"] = b["correct"] / max(b["total"], 1)
                    logs[f"eval/unparsed_{k}"] = b["unparsed"] / max(b["total"], 1)
                    logs[f"eval/n_{k}"] = b["total"]

                logger.info(f"=== Sign accuracy @ step {state.global_step} (n={total}) ===")
                for k, v in sorted(logs.items()):
                    logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                if qualitative:
                    logger.info("=== Qualitative samples ===")
                    for q in qualitative:
                        logger.info(json.dumps(q))

                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(logs, step=state.global_step)
                except Exception:
                    pass
        finally:
            if was_training:
                model.train()


ANSWER_MAGNITUDE = 32  # binary +/- this magnitude
# Prompts: two separate images, "first" vs "second".
SYSTEM_PROMPT = (
    "Compare robot task progress. You will see two images. "
    "Respond with a number: positive if the second image shows more progress, "
    "negative if the first does."
)
USER_PROMPT_TEMPLATE = (
    "Task: {task_token}\n"
    "Which image shows more task progress (the first or the second)? "
    "Respond with a number from -100 to 100."
)
from video_frame_utils import find_job_dirs, extract_frame, create_side_by_side  # noqa: E402

# For multi-task training we want a per-video natural-language description in
# the prompt. PairwiseSignAccuracyCallback re-renders the user text at eval
# time from the imported USER_PROMPT_TEMPLATE (single {task_token} slot), so
# rather than override the template (which would diverge at eval), we fold the
# description into the same task_token slot: each row stores
#     task_token = "[CURATED_TOKEN] — natural-language description"
# and both train and eval render identical text via the existing template.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)

DEFAULT_PRETRAIN_ROOT = (
    "/proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy/data/jgd/2026.05.23/"
    "23.28.09_train_diffusion_unet_hybrid_target_atomic_seen_jgd_CloseToasterOvenDoor/"
    "evals/epoch=0100-train_loss=0.0027/pretrain"
)


def task_token_for(task: str) -> str:
    """Reuse the curated token if the task is known, else auto-derive an
    UPPER_SNAKE token (CloseToasterOvenDoor -> [CLOSE_TOASTER_OVEN_DOOR])."""
    if task in TASK_TOKENS:
        return TASK_TOKENS[task]
    return "[" + re.sub(r"(?<!^)(?=[A-Z])", "_", task).upper() + "]"


@lru_cache(maxsize=8192)
def video_num_frames(path: str) -> int:
    """Number of frames in a video, or 0 if the file is unreadable/corrupt.

    Some rollout mp4s are written with a broken codec header (decord raises
    DECORDError "Cannot create buffer source" / invalid pixel format). Treat
    those as having 0 frames so callers can skip them instead of crashing the
    whole run.
    """
    import decord
    try:
        return len(decord.VideoReader(path, num_threads=1))
    except Exception as e:
        logger.warning(f"Unreadable/corrupt video, treating as 0 frames: {path} ({e})")
        return 0


def load_recovered_lang(rollout_dir: str) -> dict:
    """Read recovered_lang.json sitting next to eval_log.json.
    Returns {video_basename: lang_text}. Empty dict if missing/unreadable."""
    path = os.path.join(rollout_dir, "recovered_lang.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return {}
    return {k: v.get("lang", "") for k, v in data.items() if isinstance(v, dict)}


def load_rollout_demos(pretrain_root: str, task: str):
    """Scan every rollout subdir for `task` and group videos by episode index
    (= initial condition). Returns a (by_ep, lang_by_path) tuple where
    by_ep = {ep: {"succ": [mp4...], "fail": [mp4...]}} and
    lang_by_path = {mp4_path: natural-language description} pulled from each
    rollout dir's recovered_lang.json."""
    dirs = find_job_dirs(pretrain_root)
    by_ep, lang_by_path = {}, {}
    n_succ = n_fail = n_dirs = n_corrupt = n_lang = 0
    for d in dirs:
        log_path = os.path.join(d, "eval_log.json")
        try:
            with open(log_path) as fh:
                log = json.load(fh)
        except Exception as e:
            logger.warning(f"Could not read {log_path}: {e}")
            continue
        if log.get("eval_args", {}).get("task") != task:
            continue
        n_dirs += 1
        lang_in_dir = load_recovered_lang(d)
        for k, v in log.items():
            if not k.startswith("test/sim_max_reward_"):
                continue
            ep = int(k.rsplit("_", 1)[1])
            vname = f"seed{ep}.mp4"
            vpath = os.path.join(d, "media", vname)
            if not os.path.isfile(vpath):
                logger.warning(f"Missing video {vpath}, skipping")
                continue
            # Drop unreadable/corrupt videos up front so they never enter the
            # pairing pool (otherwise decord crashes the run when they are read).
            if video_num_frames(vpath) < 2:
                n_corrupt += 1
                continue
            slot = by_ep.setdefault(ep, {"succ": [], "fail": []})
            if float(v) >= 1.0:
                slot["succ"].append(vpath)
                n_succ += 1
            else:
                slot["fail"].append(vpath)
                n_fail += 1
            lang = lang_in_dir.get(vname)
            if lang:
                lang_by_path[vpath] = lang
                n_lang += 1
    pairable = sum(1 for s in by_ep.values() if s["succ"] and s["fail"])
    logger.info(
        f"[{task}] {n_dirs} rollout dirs -> {len(by_ep)} initial conditions; "
        f"{n_succ} success videos, {n_fail} failure videos, "
        f"{n_corrupt} corrupt videos skipped, "
        f"{pairable} pairable initial conditions (have both succ & fail); "
        f"{n_lang} videos with recovered lang"
    )
    return by_ep, lang_by_path


def build_rollout_pairs(
    by_ep,
    episodes,
    intervals,
    train_step,
    task_token,
    task_name,
    lang_by_path,
    fallback_desc,
    failure_last_frac,
    failure_min_frames,
    max_succ_per_fail,
    subsample,
    video_skip_frac=0.0,
    rng=None,
):
    """Build success-vs-success and failure-vs-success pairs for the given
    episode indices. Single camera; bucket names end in "_cam0" so the existing
    PairwiseSignAccuracyCallback per-camera aggregation works unchanged. Each
    pair's user-prompt carries the natural-language description for video_1
    (with v2 / fallback as backstop), so the model can disambiguate tasks in
    multi-task training."""
    rng = rng or random.Random(42)
    out = []
    CAM = "cam0"

    def emit(v1, f1, v2, f2, ans, kind, demo_id, bucket):
        desc = lang_by_path.get(v1) or lang_by_path.get(v2) or fallback_desc
        # Fold the description into the same task_token slot the eval callback
        # reads, so train and eval render identical user text.
        combined_token = f"{task_token} — {desc}"
        user_prompt = USER_PROMPT_TEMPLATE.format(task_token=combined_token)
        out.append({
            "video_path_1": v1, "frame_idx_1": f1,
            "video_path_2": v2, "frame_idx_2": f2,
            "correct_answer": ans,
            # Prompt-completion format so completion_only_loss can mask the long
            # constant prefix and only train on the answer tokens.
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "completion": [
                {"role": "assistant", "content": str(ans)},
            ],
            "demo_id": demo_id,
            "demo_id_exact": f"{kind}_{demo_id}",
            "demo_success": kind,
            "bucket": f"{task_name}_{bucket}",
            "camera": CAM,
            "job_name": task_name,
            "task_token": combined_token,
            "task_token_pure": task_token,
            "task_name": task_name,
            "task_description": desc,
        })

    # success-vs-success: later sub-frame from the same success video = more progress.
    n_ss = 0
    for ep in episodes:
        for spath in by_ep[ep]["succ"]:
            sub_max = (video_num_frames(spath) - 1) // subsample
            # drop the last video_skip_frac of each video; usable = count of kept frames
            usable = int(round((1.0 - video_skip_frac) * (sub_max + 1)))
            for interval in intervals:
                if usable <= interval:
                    continue
                offset = rng.randint(0, train_step - 1)
                for i1 in range(offset, usable - interval, train_step):
                    i2 = i1 + interval
                    f1, f2 = i1 * subsample, i2 * subsample
                    ans = ANSWER_MAGNITUDE
                    v1, vf1, v2, vf2 = spath, f1, spath, f2
                    if rng.random() < 0.5:
                        v1, vf1, v2, vf2 = v2, vf2, v1, vf1
                        ans = -ans
                    emit(v1, vf1, v2, vf2, ans, "success", ep,
                         f"succ_vs_succ_int{interval}_{CAM}")
                    n_ss += 1
    logger.info(f"succ_vs_succ pairs: {n_ss}")

    # failure-vs-success: last K sub-frames of a failure at episode E vs aligned
    # sub-frames of matched success videos at the SAME episode E.
    n_fs = 0
    for ep in episodes:
        succs = by_ep[ep]["succ"]
        fails = by_ep[ep]["fail"]
        if not succs or not fails:
            continue
        for fpath in fails:
            f_sub_max = (video_num_frames(fpath) - 1) // subsample
            n_sub = f_sub_max + 1
            if n_sub < 1:
                continue
            # drop the last video_skip_frac of the video first; usable = kept-frame count
            usable = int(round((1.0 - video_skip_frac) * n_sub))
            k = max(failure_min_frames, int(round(failure_last_frac * usable)))
            k = min(k, usable)
            last_indices = list(range(usable - k, usable))
            for spath in rng.sample(succs, min(max_succ_per_fail, len(succs))):
                s_sub_max = (video_num_frames(spath) - 1) // subsample
                offset = rng.randint(0, train_step - 1)
                for i in last_indices[offset::train_step]:
                    f_src = i * subsample
                    s_src = min(i, s_sub_max) * subsample
                    ans = ANSWER_MAGNITUDE
                    v1, vf1, v2, vf2 = fpath, f_src, spath, s_src
                    if rng.random() < 0.5:
                        v1, vf1, v2, vf2 = v2, vf2, v1, vf1
                        ans = -ans
                    emit(v1, vf1, v2, vf2, ans, "failure", ep, f"fail_vs_succ_{CAM}")
                    n_fs += 1
    logger.info(f"fail_vs_succ pairs: {n_fs}")
    return out


@dataclass
class RolloutArgs:
    pretrain_root: str = DEFAULT_PRETRAIN_ROOT
    # List[str] -> HfArgumentParser uses nargs="+", so the CLI is
    # `--task TaskA TaskB TaskC ...` (single task still works).
    task: List[str] = field(default_factory=lambda: ["CloseToasterOvenDoor"])
    num_eval_episodes: int = 10  # held-out initial conditions (with successes) for eval
    compare_interval: str = "5,10,20,40"
    train_sample_interval: int = 4
    failure_last_frac: float = 0.5
    failure_min_frames: int = 8
    video_skip_frac: float = 0.0  # drop the first N fraction of frames from ALL videos
    max_succ_per_fail: int = 3  # success videos paired per failure video at the same episode
    subsample: int = 1  # frame stride (videos are 10 fps)
    eval_max_pairs: int = 200
    just_visualize: bool = False
    eval_only: bool = False  # Score every held-out pair without training.
    max_pixels: str = "256x256"  # WxH; processor budget per image
    balance_fail_vs_succ: bool = False


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, RolloutArgs))
    script_args, training_args, model_args, cfg = parser.parse_args_and_config()
    training_args.max_length = None
    training_args.remove_unused_columns = False
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_sign_acc_overall"
    training_args.greater_is_better = True
    # Force loss only on the assistant completion tokens (prompt prefix is long
    # and constant; without this the model memorizes it and collapses).
    training_args.completion_only_loss = True

    intervals = [int(x) for x in cfg.compare_interval.split(",")]

    # Multi-task: build pairs per-task, then concat. by_ep is rebuilt per task so
    # pairs are always within a single task by construction (no cross-task pairs).
    # When --balance_fail_vs_succ is set, balance is applied INSIDE the loop so
    # every task contributes equal succ-vs-succ and fail-vs-succ counts (a global
    # balance would let tasks with more failures dominate the fail share).
    train_pairs, eval_pairs = [], []
    train_rng = random.Random(42)
    eval_rng = random.Random(43)
    balance_rng = random.Random(44) if cfg.balance_fail_vs_succ else None
    for task in cfg.task:
        task_token = task_token_for(task)
        # Fallback description if a video isn't in recovered_lang.json: split
        # CamelCase task name into lowercase words ("PackDessert" -> "pack dessert").
        fallback_desc = re.sub(r"(?<!^)(?=[A-Z])", " ", task).lower()
        logger.info(
            f"[{task}] token={task_token}  fallback_desc='{fallback_desc}'"
        )

        by_ep, lang_by_path = load_rollout_demos(cfg.pretrain_root, task)
        if not by_ep:
            logger.warning(f"[{task}] no rollouts found, skipping")
            continue
        episodes_all = sorted(by_ep.keys())
        if not any(by_ep[e]["succ"] for e in episodes_all):
            raise ValueError(
                f"No successful rollouts for task '{task}' under {cfg.pretrain_root}; "
                f"cannot build training pairs."
            )

        # Hold out the last num_eval_episodes PAIRABLE initial conditions per task.
        pair_eps = [e for e in episodes_all if by_ep[e]["succ"] and by_ep[e]["fail"]]
        n_eval = min(cfg.num_eval_episodes, max(0, len(pair_eps) - 1))
        eval_eps = set(pair_eps[-n_eval:]) if n_eval > 0 else set()
        train_eps = [e for e in episodes_all if e not in eval_eps]
        eval_eps_list = sorted(eval_eps)
        if not eval_eps_list:
            logger.warning(
                f"No held-out eval initial conditions for task '{task}' "
                f"(pairable={len(pair_eps)}); training without an eval split."
            )
        logger.info(
            f"[{task}] split -> train: {len(train_eps)} initial conditions, "
            f"eval: {len(eval_eps_list)} initial conditions ({eval_eps_list})"
        )

        task_train_pairs = build_rollout_pairs(
            by_ep, train_eps, intervals, cfg.train_sample_interval,
            task_token, task, lang_by_path, fallback_desc,
            cfg.failure_last_frac, cfg.failure_min_frames, cfg.max_succ_per_fail,
            cfg.subsample, video_skip_frac=cfg.video_skip_frac, rng=train_rng,
        )
        task_eval_pairs = build_rollout_pairs(
            by_ep, eval_eps_list, intervals, cfg.train_sample_interval,
            task_token, task, lang_by_path, fallback_desc,
            cfg.failure_last_frac, cfg.failure_min_frames, cfg.max_succ_per_fail,
            cfg.subsample, video_skip_frac=cfg.video_skip_frac, rng=eval_rng,
        )

        if cfg.balance_fail_vs_succ:
            # Per-task balance: upsample fail-vs-succ pairs to match
            # succ-vs-succ count within THIS task only.
            succ_p = [p for p in task_train_pairs if p["demo_success"] != "failure"]
            fail_p = [p for p in task_train_pairs if p["demo_success"] == "failure"]
            if fail_p and succ_p:
                reps = len(succ_p) // len(fail_p)
                remainder = len(succ_p) - reps * len(fail_p)
                balanced_fail = fail_p * reps + balance_rng.sample(fail_p, remainder)
                task_train_pairs = succ_p + balanced_fail
                logger.info(
                    f"[{task}] Balanced fail_vs_succ: {len(fail_p)} unique -> "
                    f"{len(balanced_fail)} to match {len(succ_p)} succ_vs_succ; "
                    f"task train total: {len(task_train_pairs)}"
                )
            else:
                logger.warning(
                    f"[{task}] balance_fail_vs_succ requested but one bucket is "
                    f"empty (succ={len(succ_p)}, fail={len(fail_p)}); skipping."
                )

        train_pairs += task_train_pairs
        eval_pairs += task_eval_pairs
    logger.info(
        f"Pair counts -> train: {len(train_pairs)}, eval: {len(eval_pairs)} "
        f"across {len(cfg.task)} tasks ({list(cfg.task)})"
    )
    if not train_pairs:
        raise ValueError(
            f"No training pairs built across tasks {list(cfg.task)} under "
            f"{cfg.pretrain_root}; check task names and rollout layout."
        )

    # (balance_fail_vs_succ is applied per-task inside the loop above so each
    # task's succ/fail counts are matched independently; no global step here.)

    # Key on task_name too: the same episode index can legitimately appear in
    # different tasks, but train/eval must still be disjoint within each task.
    train_demos = {(p["task_name"], p["demo_success"], p["demo_id"]) for p in train_pairs}
    eval_demos = {(p["task_name"], p["demo_success"], p["demo_id"]) for p in eval_pairs}
    overlap = train_demos & eval_demos
    if overlap:
        raise RuntimeError(f"Train/eval initial-condition overlap detected: {overlap}")
    logger.info(f"Train bucket counts: {Counter(p['bucket'] for p in train_pairs)}")
    logger.info(f"Eval bucket counts:  {Counter(p['bucket'] for p in eval_pairs)}")

    rng = random.Random(42)
    rng.shuffle(train_pairs)
    rng.shuffle(eval_pairs)

    if cfg.eval_only:
        from rank_videos import load_ranker, _build_prompt_text, _run_pair_chunk
        assert len({pair["task_token"] for pair in eval_pairs}) == 1, "Expected one OpenDrawer prompt"
        processor, model = load_ranker(model_args.model_name_or_path, cfg.max_pixels)
        correct = 0
        for start in range(0, len(eval_pairs), 4):
            batch = eval_pairs[start:start + 4]
            frames = [extract_frame(pair[f"video_path_{side}"], pair[f"frame_idx_{side}"])
                      for pair in batch for side in (1, 2)]
            prompt = _build_prompt_text(processor, frames,
                                       USER_PROMPT_TEMPLATE.format(task_token=batch[0]["task_token"]))
            answers = _run_pair_chunk(processor, model, frames,
                                      [(2*i, 2*i+1) for i in range(len(batch))], prompt)
            for pair, (_, _, text) in zip(batch, answers):
                match = re.search(r"-?\d+", text)
                correct += match is not None and ((int(match.group()) > 0) == (pair["correct_answer"] > 0))
            logger.info(f"Evaluated {start + len(batch)}/{len(eval_pairs)} pairs")
        result = dict(pairs=len(eval_pairs), correct=correct)
        result["accuracy"] = result["correct"] / result["pairs"]
        print(json.dumps(result))
        os.makedirs(training_args.output_dir, exist_ok=True)
        Path(training_args.output_dir, "accuracy.json").write_text(json.dumps(result, indent=2))
        sys.exit(0)

    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0

    mp_w, mp_h = (int(x) for x in cfg.max_pixels.lower().split("x"))
    max_pixels_budget = mp_w * mp_h

    if rank == 0:
        try:
            if train_pairs:
                visualize_dataset(
                    train_pairs, training_args.output_dir, split_name="train",
                    num_examples=20, max_pixels=max_pixels_budget,
                )
            if eval_pairs:
                visualize_dataset(
                    eval_pairs, training_args.output_dir, split_name="eval",
                    num_examples=20, max_pixels=max_pixels_budget,
                )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    if cfg.just_visualize:
        sys.exit(0)

    train_ds = Dataset.from_list(train_pairs)
    # HF's eval loop iterates the full eval_ds to compute eval_loss every
    # eval_steps. Our real metric (eval_sign_acc_overall) comes from the
    # PairwiseSignAccuracyCallback, which already subsamples to eval_max_pairs.
    # Cap eval_ds at 500 so eval_loss is cheap; eval_pairs was shuffled above.
    eval_ds = Dataset.from_list(eval_pairs[:500]) if eval_pairs and training_args.eval_strategy != "no" else None
    del train_pairs, eval_pairs
    gc.collect()
    logger.info(f"Train {len(train_ds)} / Eval {len(eval_ds) if eval_ds else 0}")

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    quant = get_quantization_config(model_args)
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=get_kbit_device_map() if quant is not None else None,
        quantization_config=quant,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        max_pixels=max_pixels_budget,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
    )
    if isinstance(trainer.data_collator, DataCollatorForVisionLanguageModeling):
        old = trainer.data_collator
        trainer.data_collator = TwoImageCollator(
            processor=old.processor,
            max_length=old.max_length,
            completion_only_loss=old.completion_only_loss,
            pad_to_multiple_of=None,
        )
    if eval_ds is not None:
        trainer.add_callback(PairwiseSignAccuracyCallback(
            eval_ds, processor,
            max_pairs=cfg.eval_max_pairs,
            n_qualitative=3,
        ))

    # Drop the frame-count cache built during pair construction; readers are now
    # opened per-process in video_frame_utils so no fork-unsafe decord handles
    # leak into the DataLoader workers.
    video_num_frames.cache_clear()

    # Auto-resume if output_dir already contains a checkpoint-* directory.
    has_ckpt = os.path.isdir(training_args.output_dir) and any(
        d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_args.output_dir, d))
        for d in os.listdir(training_args.output_dir)
    )
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_model(training_args.output_dir)
