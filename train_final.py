"""
Phase 2: Train final MobileAgeNet with best hyperparameters from Optuna search.

Workflow:
  1. Run train_age_estimation.py  → Phase 1: 30 trials × 50 epochs (HP search)
  2. Review best HPs from Phase 1 DB / stdout logs
  3. Update the locked values below with the best HPs
  4. Run this script              → Phase 2: 1 trial × 100 epochs (final model)

Target: MAE ≤ 3.5 yrs on UTKFace held-out test set
        normalized accuracy ≥ 0.767  (threshold = 15 yrs)

Usage:
    python train_final.py
"""
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

from ab.nn.train import main
from ab.nn.util.Const import data_dir

# ── Best HPs from Phase 1 Optuna search (30 trials × 50 epochs) ───────────────
# MODEL CHANGED: MobileAgeNet v2 now uses a pretrained MobileNetV3-Large backbone.
# Re-run train_age_estimation.py (Phase 1) to get new best HPs, then update here.
# Placeholder values below are good starting points for pretrained fine-tuning.
BEST_LR        = 1e-3       # head LR; backbone gets 0.05× inside MobileAgeNet
BEST_BATCH_PW  = 6          # batch = 2^6 = 64
BEST_MOMENTUM  = 0.90       # unused by AdamW; kept for framework compatibility
BEST_DROPOUT   = 0.30       # head dropout
BEST_TRANSFORM = ('Resize_ColorJit_Flip_Blur',)   # face-aware 224×224 augmentation
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = 'age-regression_utkface_mae_MobileAgeNet'

    print("=" * 60)
    print("PHASE 2: FINAL TRAINING WITH BEST HYPERPARAMETERS")
    print(f"  Target: MAE ≤ 3.5 yrs  (acc ≥ 0.767 with threshold=15 yrs)")
    print("=" * 60)
    print(f"\nConfig    : {config}")
    print(f"Data      : {data_dir}")
    print(f"LR        : {BEST_LR}")
    print(f"Batch     : 2^{BEST_BATCH_PW} = {2**BEST_BATCH_PW}")
    print(f"Momentum  : {BEST_MOMENTUM}  (AdamW: ignored)")
    print(f"Dropout   : {BEST_DROPOUT}")
    print(f"Transform : {BEST_TRANSFORM[0]}")
    print("=" * 60)

    main(
        config=config,
        epoch_max=100,
        n_optuna_trials=1,               # Single run — no HP search
        min_batch_binary_power=BEST_BATCH_PW,
        max_batch_binary_power=BEST_BATCH_PW,
        min_learning_rate=BEST_LR,
        max_learning_rate=BEST_LR,
        min_momentum=BEST_MOMENTUM,
        max_momentum=BEST_MOMENTUM,
        min_dropout=BEST_DROPOUT,
        max_dropout=BEST_DROPOUT,
        transform=BEST_TRANSFORM,
        save_pth_weights=True,
        save_onnx_weights=1,
        num_workers=8,
        epoch_limit_minutes=600,         # 10 hours max for 100 epochs
    )

    print("\n" + "=" * 60)
    print("PHASE 2 FINAL TRAINING COMPLETE")
    print("Check held-out test MAE printed above.")
    print("=" * 60)
