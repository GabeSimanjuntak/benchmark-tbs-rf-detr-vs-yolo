from pathlib import Path
import argparse
import sys
import yaml
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_DIR
    / "configs"
    / "baseline"
    / "baseline_v1.yaml"
)

DATASET_ROOT = (
    PROJECT_DIR
    / "datasets"
    / "experiments"
    / "v2"
)

OUTPUT_ROOT = (
    PROJECT_DIR
    / "runs"
    / "baseline_v1"
)

PILOT_OUTPUT_ROOT = (
    PROJECT_DIR
    / "runs"
    / "baseline_v1_pilot"
)

SUPPORTED_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
}

SUPPORTED_SCENARIOS = [
    "lodo_1_pabrik",
    "lodo_2_kalteng",
    "lodo_3_malaysia",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# CONFIGURATION
# ============================================================

def load_config():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment(device=0):

    print("=" * 65)
    print("BASELINE TRAINING ENVIRONMENT")
    print("=" * 65)

    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Training cancelled."
        )

    print("GPU:", torch.cuda.get_device_name(device))

    gpu_memory = (
        torch.cuda.get_device_properties(device).total_memory
        / 1024**3
    )

    print(f"GPU Memory: {gpu_memory:.2f} GB")

    print("=" * 65)


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(scenario):

    dataset_dir = DATASET_ROOT / scenario
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing dataset YAML: {data_yaml}"
        )

    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data["nc"] != 2:
        raise ValueError(
            f"{scenario}: expected 2 classes"
        )

    expected_names = {
        0: "Unripe",
        1: "Ripe",
    }

    names = data["names"]

    if names not in (
        expected_names,
        ["Unripe", "Ripe"],
    ):
        raise ValueError(
            f"{scenario}: invalid class names: {names}"
        )

    dataset_root = dataset_dir.resolve()

    split_summary = {}

    for split in ["train", "val", "test"]:

        if split not in data:
            raise ValueError(
                f"{scenario}: missing '{split}' in data.yaml"
            )

        split_path = Path(data[split])

        if not split_path.is_absolute():
            split_path = dataset_root / split_path

        split_path = split_path.resolve()

        image_dir = split_path
        label_dir = dataset_root / split / "labels"

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Missing image directory: {image_dir}"
            )

        if not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing label directory: {label_dir}"
            )

        images = {
            path.stem
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        labels = {
            path.stem
            for path in label_dir.glob("*.txt")
        }

        missing_labels = images - labels
        orphan_labels = labels - images

        if missing_labels or orphan_labels:
            raise ValueError(
                f"{scenario}/{split}: "
                f"{len(missing_labels)} missing labels, "
                f"{len(orphan_labels)} orphan labels"
            )

        split_summary[split] = len(images)

    return data, dataset_root, split_summary


# ============================================================
# RUNTIME DATASET YAML
# ============================================================

def prepare_runtime_dataset(scenario, data, dataset_root, output_root):

    runtime_dir = (
        output_root
        / "_runtime"
        / scenario
    )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    runtime_yaml = runtime_dir / "data_runtime.yaml"

    runtime_data = dict(data)

    # Use absolute dataset root.
    runtime_data["path"] = str(dataset_root)

    with runtime_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            runtime_data,
            file,
            sort_keys=False
        )

    return runtime_yaml


# ============================================================
# ACTUAL TRAINING CONFIGURATION CALLBACK
# ============================================================

def report_training_config(trainer):

    batch_size = trainer.batch_size
    nbs = trainer.args.nbs
    accumulation = trainer.accumulate

    effective_batch = (
        batch_size * accumulation
    )

    print("\n")
    print("=" * 65)
    print("ACTUAL YOLO TRAINING CONFIGURATION")
    print("=" * 65)

    print("Physical batch size:", batch_size)
    print("Nominal batch size (NBS):", nbs)
    print("Gradient accumulation:", accumulation)
    print("Nominal effective batch:", effective_batch)

    print("Image size:", trainer.args.imgsz)
    print("Epochs:", trainer.epochs)
    print("Workers:", trainer.args.workers)
    print("AMP:", trainer.args.amp)

    print("=" * 65)


# ============================================================
# YOLO TRAINING
# ============================================================

def train_yolo(
    model_name,
    scenario,
    config,
    pilot=False
):

    training = config["training"]
    augmentation = config["augmentation"]

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    data, dataset_root, split_summary = validate_dataset(
        scenario
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    model_file = SUPPORTED_MODELS[model_name]

    # --------------------------------------------------------
    # Epoch configuration
    # --------------------------------------------------------

    epochs = (
        1
        if pilot
        else training["epochs"]
    )

    # --------------------------------------------------------
    # Output configuration
    # --------------------------------------------------------

    output_root = (
        PILOT_OUTPUT_ROOT
        if pilot
        else OUTPUT_ROOT
    )

    model_output = output_root / model_name

    model_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Runtime dataset YAML
    # --------------------------------------------------------

    runtime_yaml = prepare_runtime_dataset(
        scenario,
        data,
        dataset_root,
        output_root
    )

    # --------------------------------------------------------
    # Print experiment configuration
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("STARTING YOLO EXPERIMENT")
    print("=" * 65)

    print("Model:", model_name)
    print("Pretrained weights:", model_file)
    print("Scenario:", scenario)
    print("Pilot mode:", pilot)

    print("Train images:", split_summary["train"])
    print("Validation images:", split_summary["val"])
    print("Test images:", split_summary["test"])

    print("Epochs:", epochs)
    print("Batch:", training["batch_size"])
    print("NBS:", training["yolo"]["nbs"])
    print("Image size:", training["yolo"]["image_size"])

    print("Output:", model_output)

    print("=" * 65)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = YOLO(model_file)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Register callback
    # --------------------------------------------------------

    model.add_callback(
        "on_train_start",
        report_training_config
    )

    # --------------------------------------------------------
    # Controlled augmentation
    # --------------------------------------------------------

    horizontal_flip = (
        augmentation["horizontal_flip"]["probability"]
        if augmentation["horizontal_flip"]["enabled"]
        else 0.0
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    results = model.train(

        # Dataset
        data=str(runtime_yaml),

        # Training parameters
        epochs=epochs,
        batch=training["batch_size"],
        nbs=training["yolo"]["nbs"],
        imgsz=training["yolo"]["image_size"],

        # Hardware
        device=0,
        workers=training["workers"],

        # Reproducibility
        seed=config["experiment"]["seed"],
        deterministic=config["experiment"]["deterministic"],

        # Pretrained and precision
        pretrained=training["pretrained"],
        amp=training["amp"],

        # Optimizer
        optimizer="auto",

        # Validation
        val=True,
        split="val",

        # Disable early stopping for fixed-epoch baseline
        patience=epochs + 1,

        # Output
        project=str(model_output),
        name=scenario,
        exist_ok=False,

        # Checkpoint
        save=True,
        save_period=1,

        # Visualization
        plots=True,
        verbose=True,

        # ----------------------------------------------------
        # Controlled augmentation
        # ----------------------------------------------------

        fliplr=horizontal_flip,
        flipud=0.0,

        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        cutmix=0.0,

        auto_augment=None,
        erasing=0.0,

        close_mosaic=0,
    )

    # ========================================================
    # TRAINING RESULTS
    # ========================================================

    experiment_dir = Path(results.save_dir)

    best_checkpoint = (
        experiment_dir
        / "weights"
        / "best.pt"
    )

    last_checkpoint = (
        experiment_dir
        / "weights"
        / "last.pt"
    )

    print("\n")
    print("=" * 65)
    print("YOLO TRAINING COMPLETED")
    print("=" * 65)

    print("Experiment:", model_name, scenario)
    print("Output directory:", experiment_dir)

    print(
        "Best checkpoint:",
        best_checkpoint,
        best_checkpoint.exists()
    )

    print(
        "Last checkpoint:",
        last_checkpoint,
        last_checkpoint.exists()
    )

    if pilot:
        print("\nPILOT RESULT ONLY.")
        print("This is NOT a baseline research result.")

    print("=" * 65)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="TBS Cross-Domain Baseline Training Runner"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(SUPPORTED_MODELS.keys())
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=SUPPORTED_SCENARIOS
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run one epoch as a pilot test"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    check_environment(device=0)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    data, dataset_root, split_summary = validate_dataset(
        args.scenario
    )

    output_root = (
        PILOT_OUTPUT_ROOT
        if args.pilot
        else OUTPUT_ROOT
    )

    runtime_yaml = prepare_runtime_dataset(
        args.scenario,
        data,
        dataset_root,
        output_root
    )

    # --------------------------------------------------------
    # Experiment summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("SELECTED EXPERIMENT")
    print("=" * 65)

    print("Model:", args.model)
    print("Scenario:", args.scenario)

    print("Dataset:", runtime_yaml)

    print("Train images:", split_summary["train"])
    print("Validation images:", split_summary["val"])
    print("Test images:", split_summary["test"])

    print("Epochs:", 1 if args.pilot else config["training"]["epochs"])

    print("Batch:", config["training"]["batch_size"])

    print(
        "Image size:",
        config["training"]["yolo"]["image_size"]
    )

    print(
        "Effective batch target:",
        config["training"]["effective_batch_size"]
    )

    print("Pilot mode:", args.pilot)

    print("=" * 65)

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry_run:

        print("\nDRY RUN")
        print("Training was NOT started.")

        return

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    train_yolo(
        model_name=args.model,
        scenario=args.scenario,
        config=config,
        pilot=args.pilot
    )


if __name__ == "__main__":
    main()
