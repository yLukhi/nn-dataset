# Train MobileAgeNet on UTKFace with Optuna hyperparameter optimisation
# Target: MAE ≤ 3.5 yrs (normalized accuracy ≥ 0.825, threshold=20 yrs)
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

import ab.nn.util.db.Util as _db_util
_db_util.get_package_location = lambda x: None

from ab.nn.train import main
from ab.nn.util.Const import data_dir, db_file, code_tables
from ab.nn.util.db.Init import init_db, sql_conn, close_conn
from ab.nn.util.db.Write import populate_code_table
from ab.nn.util.db.Read import supported_transformers

if __name__ == '__main__':
    # Initialize fresh database with model code
    print("Initializing database...")
    if db_file.exists():
        db_file.unlink()
    init_db()
    conn, cursor = sql_conn()
    for table in code_tables:
        populate_code_table(table, cursor)
    close_conn(conn)
    print("Database initialized with model code!\n")

    # Prefer face-aware transforms; fall back to all valid ones if unavailable.
    # five_crop excluded: returns tuple instead of a single tensor.
    # Grayscale excluded: destroys skin-tone cues that carry age signal.
    all_transforms = supported_transformers()
    preferred_transforms = [
        'Resize_ColorJit_Flip_Blur',          # face-aware 224×224 (best for age)
        'norm_256_flip',                       # simple 256×256 + flip baseline
        'norm_256',                            # no-augmentation reference
        'bf-v1-RandomCrop_Pad_RandomHorizontalFlip_256',  # 256 crop+flip
        'bf-v1-RandomHorizontalFlip_RandomAutocontrast_256',
        'bf-v1-RandomCrop_RandomRotation_RandomAutocontrast_256',
        'bf-v1-RandomPosterize_256',
    ]
    # Keep only transforms that are actually registered in this repo
    valid_preferred = tuple([t for t in preferred_transforms if t in all_transforms])
    # Fallback: use all valid transforms (minus broken ones) when preferred set is too small
    fallback_transforms = tuple([
        t for t in all_transforms
        if t not in ('five_crop',) and 'Grayscale' not in t and 'grayscale' not in t
    ])
    face_transforms = valid_preferred if len(valid_preferred) >= 3 else fallback_transforms
    print(f"Face-aware transforms selected: {len(face_transforms)}\n")

    config = 'age-regression_utkface_mae_MobileAgeNet'

    print("=" * 60)
    print("PHASE 1: HYPERPARAMETER SEARCH — MobileAgeNet on UTKFace")
    print(f"  Target: MAE ≤ 3.5 yrs  (acc ≥ 0.767 with threshold=15 yrs)")
    print("=" * 60)
    print(f"\nConfig : {config}")
    print(f"Data   : {data_dir}")
    print("=" * 60)

    main(
        config=config,
        epoch_max=50,
        n_optuna_trials=30,
        # Batch: 64 wins consistently; allow 128 for Optuna to verify.
        min_batch_binary_power=6,
        max_batch_binary_power=7,
        # LR = head LR for pretrained fine-tuning.
        # Backbone gets 0.05× this inside MobileAgeNet._init_finetune().
        # Range: 3e-4..2e-3 — wide enough to probe without damaging pretrained features.
        min_learning_rate=3e-4,
        max_learning_rate=2e-3,
        # momentum unused by AdamW but kept for framework compatibility
        min_momentum=0.85,
        max_momentum=0.95,
        # Dropout range for the small regression head
        min_dropout=0.1,
        max_dropout=0.5,
        transform=face_transforms,
        save_pth_weights=True,
        save_onnx_weights=1,
        num_workers=8,
        epoch_limit_minutes=1060,
    )

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE — inspect DB / logs for best HPs")
    print("Then update train_final.py and run it for Phase 2.")
    print("=" * 60)
