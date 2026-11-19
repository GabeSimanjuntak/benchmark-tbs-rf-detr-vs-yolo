from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_YAML = (
    PROJECT_DIR
    / "datasets"
    / "experiments"
    / "v2"
    / "lodo_1_pabrik"
    / "data.yaml"
)

MODEL_NAME = "yolo11n.pt"

# Smoke test parameters
EPOCHS = 1
BATCH_SIZE = 8
IMAGE_SIZE = 640
DEVICE = 0
WORKERS = 0
SEED = 42

# Output directories
PROJECT_OUTPUT = PROJECT_DIR / "runs" / "smoke_test"
EXPERIMENT_NAME = "yolo11n_lodo1"

EXPERIMENT_DIR = PROJECT_OUTPUT / EXPERIMENT_NAME

# Runtime dataset configuration
RUNTIME_DATA_YAML = EXPERIMENT_DIR / "data_runtime.yaml"


# ============================================================
# PREPARE RUNTIME DATASET CONFIGURATION
# ============================================================

def prepare_runtime_dataset():

    print("\nPreparing runtime dataset configuration...")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    with open(DATA_YAML, "r") as file:
        data_config = yaml.safe_load(file)

    dataset_root = DATA_YAML.parent.resolve()

    # Use absolute dataset paths
    data_config["path"] = str(dataset_root)

    RUNTIME_DATA_YAML.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(RUNTIME_DATA_YAML, "w") as file:
        yaml.safe_dump(
            data_config,
            file,
            sort_keys=False
        )

    print("Runtime YAML created:")
    print(RUNTIME_DATA_YAML)

    print("\nDataset paths:")

    for split in ["train", "val", "test"]:

        split_path = (
            dataset_root
            / data_config[split]
        ).resolve()

        print(f"{split}: {split_path}")

        if not split_path.is_dir():
            raise FileNotFoundError(
                f"{split} directory not found: {split_path}"
            )

    return RUNTIME_DATA_YAML


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment():

    print("=" * 65)
    print("YOLO11n SMOKE TEST")
    print("=" * 65)

    print(f"Python        : {__import__('sys').version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available!"
        )

    print(
        f"GPU           : {torch.cuda.get_device_name(DEVICE)}"
    )

    gpu_memory = (
        torch.cuda.get_device_properties(DEVICE).total_memory
        / 1024**3
    )

    print(f"GPU Memory    : {gpu_memory:.2f} GB")
    print(f"Original YAML : {DATA_YAML}")
    print(f"Model         : {MODEL_NAME}")
    print(f"Output        : {EXPERIMENT_DIR}")

    print("=" * 65)


# ============================================================
# TRAINING SMOKE TEST
# ============================================================

def run_smoke_test(runtime_data):

    print("\nLoading YOLO11n model...")

    model = YOLO(MODEL_NAME)

    print("Model loaded successfully.")

    print("\nStarting smoke test training...")

    results = model.train(

        # Dataset
        data=str(runtime_data),

        # Training parameters
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,

        # Hardware
        device=DEVICE,
        workers=WORKERS,

        # Reproducibility
        seed=SEED,
        deterministic=True,

        # Training settings
        amp=True,
        pretrained=True,
        optimizer="auto",

        # Validation
        val=True,

        # Output
        project=str(PROJECT_OUTPUT),
        name=EXPERIMENT_NAME,
        exist_ok=True,

        # Save outputs
        save=True,
        save_period=1,
        plots=True,

        verbose=True,
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    check_environment()

    runtime_data = prepare_runtime_dataset()

    results = run_smoke_test(runtime_data)

    print("\n" + "=" * 65)
    print("SMOKE TEST COMPLETED")
    print("=" * 65)

    print(f"Output directory : {results.save_dir}")

    best_checkpoint = (
        Path(results.save_dir)
        / "weights"
        / "best.pt"
    )

    last_checkpoint = (
        Path(results.save_dir)
        / "weights"
        / "last.pt"
    )

    print(f"Best checkpoint  : {best_checkpoint}")
    print(f"Last checkpoint  : {last_checkpoint}")

    print(f"Best exists      : {best_checkpoint.exists()}")
    print(f"Last exists      : {last_checkpoint.exists()}")

    print("=" * 65)


if __name__ == "__main__":
    main()
