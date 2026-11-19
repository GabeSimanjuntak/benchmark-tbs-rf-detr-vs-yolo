from pathlib import Path

from rfdetr import RFDETRSmall


PROJECT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_DIR
    / "datasets"
    / "experiments"
    / "v2"
    / "lodo_1_pabrik"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "runs"
    / "smoke_test"
    / "rfdetr_small_lodo1"
)

RESOLUTION = 512


if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

if not (DATASET_DIR / "data.yaml").exists():
    raise FileNotFoundError("data.yaml not found")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


model = RFDETRSmall()

model.train(
    dataset_dir=str(DATASET_DIR),
    dataset_file="yolo",

    epochs=1,
    batch_size=2,
    grad_accum_steps=1,

    resolution=RESOLUTION,

    device="cuda",

    num_workers=0,
    seed=42,

    output_dir=str(OUTPUT_DIR),

    class_names=["Unripe", "Ripe"],

    checkpoint_interval=1,
    eval_interval=1,

    run_test=False,
)

print("\nRF-DETR SMALL SMOKE TEST FINISHED")
print(f"Output directory: {OUTPUT_DIR}")
