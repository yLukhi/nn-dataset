#!/usr/bin/env python3
"""
Convert an ONNX age-estimation model to TFLite and verify MAE gap.

This script performs three steps:
1. Convert ONNX -> TensorFlow SavedModel via onnx2tf
2. Convert SavedModel -> float32 TFLite
3. Evaluate MAE of ONNX and TFLite on the same UTKFace validation samples
"""

from __future__ import annotations

import argparse
import random
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from datasets import load_dataset

# Ensure local package imports work when script is launched from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _patch_onnx_helper_for_onnx2tf() -> None:
    """Patch ONNX helper for environments where onnx2tf expects removed symbols."""
    if hasattr(onnx.helper, "float32_to_bfloat16"):
        return

    def _float32_to_bfloat16(value, truncate: bool = False):
        arr = np.asarray(value, dtype=np.float32)
        u32 = arr.view(np.uint32)
        if not truncate:
            # Round-to-nearest-even before dropping low 16 bits.
            lsb = ((u32 >> 16) & 1) + 0x7FFF
            u32 = u32 + lsb
        return (u32 >> 16).astype(np.uint16)

    onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16  # type: ignore[attr-defined]


def _infer_onnx_layout_and_size(onnx_path: str) -> Tuple[str, int, int]:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    if len(shape) != 4:
        raise ValueError(f"Expected ONNX 4D input, got: {shape}")

    if shape[1] == 3:
        h = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 224
        w = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 224
        return "nchw", h, w

    if shape[-1] == 3:
        h = int(shape[1]) if isinstance(shape[1], int) and shape[1] > 0 else 224
        w = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 224
        return "nhwc", h, w

    raise ValueError(f"Unable to infer input layout from ONNX shape: {shape}")


def _age_bin(age: float) -> int:
    return int(age // 5)


def _stratified_split_indices(ages: List[float], train_ratio=0.70, val_ratio=0.10, seed=42):
    rng = random.Random(seed)
    bins: Dict[int, List[int]] = {}

    for idx, age in enumerate(ages):
        bins.setdefault(_age_bin(age), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, idxs in bins.items():
        rng.shuffle(idxs)
        n = len(idxs)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _prepare_samples(max_samples: int, image_h: int, image_w: int, eval_split: str, split_seed: int) -> List[Tuple[np.ndarray, float]]:
    dataset = load_dataset(
        "nu-delta/utkface",
        split="train",
        cache_dir=str(REPO_ROOT / "data" / "utkface_cache"),
    )

    norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    filtered_items = []
    for item in dataset:
        age = item.get("age")
        img = item.get("image")
        if age is None or img is None:
            continue
        try:
            age_val = float(age)
        except (TypeError, ValueError):
            continue
        if 0.0 <= age_val <= 116.0:
            filtered_items.append(item)

    ages = [float(item["age"]) for item in filtered_items]
    train_idx, val_idx, test_idx = _stratified_split_indices(ages, train_ratio=0.70, val_ratio=0.10, seed=split_seed)

    if eval_split == "train":
        chosen_indices = train_idx
    elif eval_split == "val":
        chosen_indices = val_idx
    elif eval_split in ("heldout", "test"):
        chosen_indices = test_idx
    else:
        raise ValueError(f"Unsupported eval_split: {eval_split}")

    if max_samples > 0:
        chosen_indices = chosen_indices[:max_samples]

    samples: List[Tuple[np.ndarray, float]] = []
    for idx in chosen_indices:
        item = filtered_items[idx]
        if len(samples) >= max_samples > 0:
            break

        img = item["image"].convert("RGB").resize((image_w, image_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - norm_mean) / norm_std

        # Keep canonical in-memory format as CHW; model-specific conversion happens later.
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        age = float(item["age"])
        samples.append((chw, age))

    return samples


def _extract_training_reference_mae(summary_path: str) -> Tuple[float | None, float | None]:
    if not summary_path:
        return None, None
    p = Path(summary_path)
    if not p.exists():
        return None, None

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ts = data.get("training_summary", {})
    best_val_mae = ts.get("best_val_mae_years")
    final_val_mae = ts.get("final_val_mae_years")
    return best_val_mae, final_val_mae


def _prepare_input_for_model(sample_chw: np.ndarray, input_shape: List[int]) -> Tuple[np.ndarray, int]:
    """
    Convert a CHW sample to model input shape and pad batch if static batch > 1.

    Returns:
      input_tensor, valid_batch_items
    """
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D input, got shape spec: {input_shape}")

    shape = list(input_shape)
    batch_dim = shape[0]

    # Handle NCHW vs NHWC
    if shape[1] == 3:
        one = np.expand_dims(sample_chw, axis=0)
    elif shape[-1] == 3:
        one = np.expand_dims(np.transpose(sample_chw, (1, 2, 0)), axis=0)
    else:
        raise ValueError(f"Cannot detect channel layout from input shape: {shape}")

    if isinstance(batch_dim, str):
        return one, 1

    if batch_dim in (-1, 0, 1):
        return one, 1

    if batch_dim > 1:
        padded = np.repeat(one, repeats=batch_dim, axis=0)
        return padded, 1

    raise ValueError(f"Unsupported batch dim in input shape: {shape}")


def _evaluate_onnx_mae(onnx_path: str, samples: List[Tuple[np.ndarray, float]]) -> float:
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    total_abs = 0.0
    total = 0
    for sample, gt in samples:
        x_in, valid = _prepare_input_for_model(sample, inp.shape)
        pred = session.run([out.name], {inp.name: x_in})[0]
        pred_age = float(np.asarray(pred).reshape(-1)[0])
        total_abs += abs(pred_age - gt)
        total += valid

    return total_abs / total if total else 0.0


def _convert_onnx_to_tflite(onnx_path: str, tflite_path: str, workdir: str) -> str:
    _patch_onnx_helper_for_onnx2tf()
    from onnx2tf import convert as onnx2tf_convert

    saved_model_dir = os.path.join(workdir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    onnx2tf_convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        not_use_onnxsim=True,
        non_verbose=True,
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    return saved_model_dir


def _evaluate_tflite_mae(tflite_path: str, samples: List[Tuple[np.ndarray, float]]) -> float:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)

    # If the model has dynamic spatial dims, resize once using the first sample.
    first_sample = samples[0][0]
    input_details = interpreter.get_input_details()[0]
    in_idx = input_details["index"]
    shape_sig = list(input_details.get("shape_signature", input_details["shape"]))
    if -1 in shape_sig:
        h, w = first_sample.shape[1], first_sample.shape[2]
        if len(shape_sig) == 4 and shape_sig[-1] == 3:
            target = [1, h, w, 3]
        elif len(shape_sig) == 4 and shape_sig[1] == 3:
            target = [1, 3, h, w]
        else:
            target = [1, h, w, 3]
        interpreter.resize_tensor_input(in_idx, target, strict=False)

    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_idx = inp["index"]
    out_idx = out["index"]

    # float32 TFLite expected in this pipeline; still keep robust casting.
    input_dtype = inp["dtype"]

    total_abs = 0.0
    total = 0
    for sample, gt in samples:
        x_in, valid = _prepare_input_for_model(sample, list(inp["shape"]))
        x_feed = x_in.astype(input_dtype)

        interpreter.set_tensor(in_idx, x_feed)
        interpreter.invoke()
        pred = interpreter.get_tensor(out_idx)
        pred_age = float(np.asarray(pred).reshape(-1)[0])

        total_abs += abs(pred_age - gt)
        total += valid

    return total_abs / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ONNX age model to TFLite and compare MAE")
    parser.add_argument(
        "--onnx-path",
        default="cluster_results/age-estimation-visualization/models/MobileAgeNet/best_model.onnx",
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--tflite-path",
        default="cluster_results/age-estimation-visualization/models/MobileAgeNet/best_model.tflite",
        help="Path to output TFLite model",
    )
    parser.add_argument(
        "--report-path",
        default="ab/nn/imp/TFLITE_REPORT_AGE_EPOCH50.json",
        help="JSON report output path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Number of samples to evaluate (0 means full selected split)",
    )
    parser.add_argument(
        "--eval-split",
        default="val",
        choices=["train", "val", "heldout", "test"],
        help="UTKFace split to evaluate, matching training loader split strategy",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the age-stratified split (must match training for consistency)",
    )
    parser.add_argument(
        "--mae-gap-threshold",
        type=float,
        default=0.5,
        help="Allowed absolute MAE gap between ONNX and TFLite",
    )
    parser.add_argument(
        "--reference-mae",
        type=float,
        default=4.718171434096904,
        help="Reference MAE from training summary",
    )
    parser.add_argument(
        "--training-summary-path",
        default="cluster_results/age-estimation-visualization/tensorboard/utkface/MobileAgeNet/trial_0_20260402-111050/training_summary.json",
        help="Path to training_summary.json to extract best/final val MAE reference",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    summary_best_mae, summary_final_mae = _extract_training_reference_mae(args.training_summary_path)
    if summary_best_mae is not None:
        args.reference_mae = float(summary_best_mae)

    layout, image_h, image_w = _infer_onnx_layout_and_size(str(onnx_path))
    print(
        f"Loading samples (split={args.eval_split}, n={'all' if args.max_samples == 0 else args.max_samples}, layout={layout}, size={image_h}x{image_w})..."
    )
    samples = _prepare_samples(args.max_samples, image_h, image_w, args.eval_split, args.split_seed)
    if not samples:
        raise RuntimeError("No samples found for selected split and preprocessing.")

    print("Evaluating ONNX MAE...")
    onnx_mae = _evaluate_onnx_mae(str(onnx_path), samples)

    print("Converting ONNX to TFLite...")
    with tempfile.TemporaryDirectory() as tmpdir:
        _convert_onnx_to_tflite(str(onnx_path), args.tflite_path, tmpdir)

    print("Evaluating TFLite MAE...")
    tflite_mae = _evaluate_tflite_mae(args.tflite_path, samples)

    mae_gap = abs(tflite_mae - onnx_mae)
    ref_gap = abs(tflite_mae - args.reference_mae)
    pass_gap = mae_gap <= args.mae_gap_threshold

    model_name = "MobileAgeNet"
    report = {
        model_name: {
            "status": "success" if pass_gap else "warning_large_gap",
            "tflite_path": str(Path(args.tflite_path).as_posix()),
            "tflite_size_mb": round(Path(args.tflite_path).stat().st_size / (1024 * 1024), 4),
            "onnx_mae": round(onnx_mae, 4),
            "tflite_mae": round(tflite_mae, 4),
            "mae_gap_onnx_vs_tflite": round(mae_gap, 4),
            "reference_mae": round(args.reference_mae, 4),
            "mae_gap_vs_reference": round(ref_gap, 4),
            "mae_gap_threshold": round(args.mae_gap_threshold, 4),
            "samples_evaluated": len(samples),
            "eval_split": args.eval_split,
            "split_seed": args.split_seed,
            "training_summary_path": args.training_summary_path,
            "training_summary_best_val_mae": round(float(summary_best_mae), 4) if summary_best_mae is not None else None,
            "training_summary_final_val_mae": round(float(summary_final_mae), 4) if summary_final_mae is not None else None,
            "preprocess": {
                "resize": [image_h, image_w],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        }
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("=== Conversion Summary ===")
    print(f"ONNX MAE   : {onnx_mae:.4f}")
    print(f"TFLite MAE : {tflite_mae:.4f}")
    print(f"MAE Gap    : {mae_gap:.4f} (threshold={args.mae_gap_threshold:.4f})")
    print(f"Reference  : {args.reference_mae:.4f} (gap={ref_gap:.4f})")
    if summary_best_mae is not None or summary_final_mae is not None:
        print(
            f"Training Summary MAE (best/final val): "
            f"{summary_best_mae if summary_best_mae is not None else 'n/a'} / "
            f"{summary_final_mae if summary_final_mae is not None else 'n/a'}"
        )
    print(f"Report     : {report_path}")

    return 0 if pass_gap else 2


if __name__ == "__main__":
    raise SystemExit(main())
