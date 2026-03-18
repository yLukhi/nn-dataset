import os
import sys
import json
import tarfile
import pickle
import urllib.request
import numpy as np
import torch
import warnings
from huggingface_hub import hf_hub_download
import importlib.util

# Silence warnings
warnings.filterwarnings("ignore")

print(f"🚀 STARTING AGE-REGRESSION TFLITE PIPELINE")
print(f"ℹ️  Torch Version: {torch.__version__}")

# ==========================================
# IMPORTS & COMPATIBILITY
# ==========================================
try:
    import ai_edge_torch
    from ai_edge_torch.quantize.quant_config import QuantConfig
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config
    )
except ImportError as e:
    print(f"❌ Import Error: {e}"); sys.exit(1)

class ConfigWrapper:
    def __init__(self, original_config):
        self.original_config = original_config
        self.is_dynamic = False
        self.is_qat = getattr(original_config, 'is_qat', False)
    def __getattr__(self, name): return getattr(self.original_config, name)

try:
    from ai_edge_litert.interpreter import Interpreter
    print(f"✅ LiteRT Interpreter Loaded")
except ImportError:
    print("❌ LiteRT missing."); sys.exit(1)

# ==========================================
# CONFIG
# ==========================================
REPO_ID = "NN-Dataset/checkpoints-epoch-50"
LOCAL_MODEL_DIR = os.path.join("ab", "nn", "nn")
OUTPUT_FOLDER = "final_tflite_results_age_epoch50"
FINAL_REPORT_FILE = "TFLITE_REPORT_AGE_EPOCH50.json"
TEST_BATCH_LIMIT = 20

from ab.nn.loader.utkface import loader as utk_loader

def get_calibration_batch(val_ds, batch_size=32):
    imgs = []
    for i in range(min(batch_size, len(val_ds))):
        x, y = val_ds[i]
        imgs.append(x.unsqueeze(0))
    if not imgs:
        raise RuntimeError("No images available for calibration")
    batch = torch.cat(imgs, dim=0)
    return batch

def evaluate_regression(tflite_path, val_ds, limit_batches):
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_idx = input_details[0]['index']
    out_idx = output_details[0]['index']

    model_input_shape = input_details[0]['shape']
    req_batch_size = model_input_shape[0]
    nhwc = model_input_shape[-1] == 3

    total_abs_error = 0.0
    total_count = 0

    # Create simple generator over val_ds
    def gen(bs):
        for i in range(0, len(val_ds), bs):
            batch = [val_ds[j][0] for j in range(i, min(i+bs, len(val_ds)))]
            labels = [val_ds[j][1] for j in range(i, min(i+bs, len(val_ds)))]
            x = torch.cat([b.unsqueeze(0) if b.dim()==3 else b for b in batch], dim=0)
            y = torch.cat([l.unsqueeze(0) if l.dim()==1 else l for l in labels], dim=0)
            yield x, y

    for i, (images, labels) in enumerate(gen(req_batch_size)):
        if limit_batches and i >= limit_batches: break

        current_bs = images.shape[0]
        input_data = images.numpy()

        if nhwc:
            input_data = np.transpose(input_data, (0,2,3,1))

        if current_bs < req_batch_size:
            pad_needed = req_batch_size - current_bs
            pad_shape = (pad_needed, *input_data.shape[1:])
            padding = np.zeros(pad_shape, dtype=np.float32)
            input_data = np.concatenate([input_data, padding], axis=0)

        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(out_idx)

        # Normalize output shape
        for j in range(current_bs):
            pred = float(np.squeeze(output[j]))
            gt = float(labels[j].item())
            total_abs_error += abs(pred - gt)
            total_count += 1

    mae = total_abs_error / total_count if total_count else 0.0
    return mae

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        summary_path = hf_hub_download(repo_id=REPO_ID, filename="all_models_summary.json")
        with open(summary_path, 'r') as f: original_summary = json.load(f)
    except Exception as e:
        print(f"❌ Download Error: {e}"); return

    final_report = {}
    if os.path.exists(FINAL_REPORT_FILE):
        try:
            with open(FINAL_REPORT_FILE, 'r') as f: final_report = json.load(f)
        except: pass

    # Prepare a small val dataset to sample calibration images
    _, _, train_ds, val_ds = utk_loader(lambda m: (lambda img: img), 'age-regression')

    # Build calibration batch
    try:
        calib_batch = get_calibration_batch(val_ds, batch_size=32)
    except Exception as e:
        print(f"❌ Calibration batch error: {e}"); return

    for i, (model_key, meta_data) in enumerate(original_summary.items(), 1):
        model_name = meta_data.get('nn', model_key)
        print(f"--------------------------------------------------")
        print(f"🔄 Processing [{i}/{len(original_summary)}]: {model_name}")

        if model_name in final_report and final_report[model_name].get("status") == "success":
            print(f"   ⏭️  Skipping (Done)")
            continue

        py_path = os.path.join(LOCAL_MODEL_DIR, f"{model_name}.py")
        if not os.path.exists(py_path):
            print(f"   ⚠️  Model file not found: {py_path}")
            continue

        try:
            spec = importlib.util.spec_from_file_location("dynamic_model", py_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            Net = module.Net

            safe_prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.0}
            if 'prm' in meta_data: safe_prm.update(meta_data['prm'])

            # Instantiate model robustly (try several signatures)
            model_fp32 = None
            try:
                model_fp32 = Net(in_shape=(1,3,32,32), out_shape=(1,), prm=safe_prm, device="cpu").eval()
            except Exception:
                try:
                    model_fp32 = Net(in_shape=(1,3,256,256), out_shape=(1,), prm=safe_prm, device="cpu").eval()
                except Exception:
                    try:
                        model_fp32 = Net(prm=safe_prm, device="cpu").eval()
                    except Exception:
                        model_fp32 = Net().eval()

            ckpt = hf_hub_download(repo_id=REPO_ID, filename=f"{model_name}.pth")
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            if "state_dict" in sd: sd = sd["state_dict"]
            model_fp32.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)

            # Quantization config
            quantizer = XNNPACKQuantizer()
            patched_config = ConfigWrapper(get_symmetric_quantization_config())
            quantizer.set_global(patched_config)
            q_config = QuantConfig(pt2e_quantizer=quantizer)

            # Convert
            tflite_model = ai_edge_torch.convert(model_fp32, (calib_batch,), quant_config=q_config)
            tflite_path = os.path.join(OUTPUT_FOLDER, f"{model_name}.tflite")
            tflite_model.export(tflite_path)
            size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"      ✅ Saved: {size_mb:.2f} MB")

            # Validate (MAE)
            mae = evaluate_regression(tflite_path, val_ds, TEST_BATCH_LIMIT)
            print(f"      🎯 MAE: {mae:.4f}")

            final_report[model_name] = {
                "status": "success",
                "tflite_size_mb": round(size_mb, 2),
                "tflite_mae": round(mae, 4),
                "original_mae": meta_data.get('accuracy', 0.0),
                "drop": round((meta_data.get('accuracy', 0.0) - mae), 4)
            }
            with open(FINAL_REPORT_FILE, 'w') as f: json.dump(final_report, f, indent=4)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            final_report[model_name] = {"status": "failed", "error": str(e)}
            with open(FINAL_REPORT_FILE, 'w') as f: json.dump(final_report, f, indent=4)

if __name__ == "__main__":
    main()
