from pathlib import Path
import argparse
import sys
import yaml
import torch

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium


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

PILOT_OUTPUT_ROOT = (
    PROJECT_DIR
    / "runs"
    / "baseline_v1_pilot_rfdetr"
)

BASELINE_OUTPUT_ROOT = (
    PROJECT_DIR
    / "runs"
    / "baseline_v1"
    / "rfdetr"
)

MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
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
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment():
    print("\n" + "=" * 65)
    print("RF-DETR BASELINE ENVIRONMENT")
    print("=" * 65)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Training requires GPU."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU          : {gpu_name}")
    print(f"GPU Memory   : {gpu_memory:.2f} GB")
    print("=" * 65)


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(scenario, config):
    dataset_dir = DATASET_ROOT / scenario
    data_yaml = dataset_dir / "data.yaml"

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}"
        )

    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    expected_classes = config["dataset"]["class_names"]

    if data.get("nc") != len(expected_classes):
        raise ValueError(
            f"Invalid number of classes in {scenario}"
        )

    raw_names = data.get("names")

    if isinstance(raw_names, dict):
        actual_classes = [
            raw_names[i] for i in range(len(raw_names))
        ]
    elif isinstance(raw_names, list):
        actual_classes = raw_names
    else:
        raise ValueError(
            f"Invalid class names in {data_yaml}"
        )

    if actual_classes != expected_classes:
        raise ValueError(
            f"Class mismatch in {scenario}: "
            f"{actual_classes}"
        )

    summary = {}

    for split in ["train", "val", "test"]:
        split_path = Path(data[split])

        if not split_path.is_absolute():
            split_path = dataset_dir / split_path

        image_dir = split_path
        label_dir = split_path.parent / "labels"

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory missing: {image_dir}"
            )

        if not label_dir.is_dir():
            raise FileNotFoundError(
                f"Label directory missing: {label_dir}"
            )

        images = {
            p.stem
            for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        }

        labels = {
            p.stem
            for p in label_dir.glob("*.txt")
        }

        missing_labels = images - labels
        orphan_labels = labels - images

        if missing_labels or orphan_labels:
            raise ValueError(
                f"Image-label mismatch in {scenario}/{split}: "
                f"missing={len(missing_labels)}, "
                f"orphan={len(orphan_labels)}"
            )

        summary[split] = len(images)

    return dataset_dir, summary


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

def get_model_resolution(model_name, config):
    resolutions = config["training"]["rfdetr"]["resolutions"]

    return resolutions[model_name]


def get_output_directory(model_name, scenario, pilot):
    if pilot:
        root = PILOT_OUTPUT_ROOT
    else:
        root = BASELINE_OUTPUT_ROOT

    return root / model_name / scenario


def print_experiment_config(
    model_name,
    scenario,
    resolution,
    epochs,
    config,
    summary,
    output_dir,
    pilot,
):
    training = config["training"]
    rfdetr_config = training["rfdetr"]
    augmentation = config["augmentation"]

    batch_size = training["batch_size"]
    accumulation = rfdetr_config["grad_accum_steps"]

    horizontal_flip = augmentation["horizontal_flip"]

    print("\n" + "=" * 65)
    print("RF-DETR EXPERIMENT CONFIGURATION")
    print("=" * 65)

    print(f"Model             : RF-DETR {model_name.upper()}")
    print(f"Scenario          : {scenario}")
    print(f"Resolution        : {resolution}")
    print(f"Epochs            : {epochs}")
    print(f"Batch size        : {batch_size}")
    print(f"Grad accumulation : {accumulation}")
    print(f"Effective batch   : {batch_size * accumulation}")
    print(f"Workers           : {training['workers']}")
    print(f"Seed              : {config['experiment']['seed']}")
    print(f"Multi-scale       : off")
    print(f"Scale jitter      : False")
    print(
        "Horizontal flip  : "
        f"{horizontal_flip['probability'] if horizontal_flip['enabled'] else 0.0}"
    )
    print(f"Augmentation     : Albumentations")
    print(f"Pilot mode        : {pilot}")

    print("\nDataset:")
    for split, count in summary.items():
        print(f"{split:10s}: {count} images")

    print(f"\nOutput directory  : {output_dir}")
    print("=" * 65)


# ============================================================
# TRAINING
# ============================================================

def train_rfdetr(
    model_name,
    scenario,
    config,
    pilot=False,
):
    training = config["training"]
    rfdetr_config = training["rfdetr"]

    dataset_dir, summary = validate_dataset(
        scenario,
        config,
    )

    resolution = get_model_resolution(
        model_name,
        config,
    )

    epochs = 1 if pilot else training["epochs"]

    output_dir = get_output_directory(
        model_name,
        scenario,
        pilot,
    )

    # Never overwrite an existing experiment.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    horizontal_flip = config["augmentation"]["horizontal_flip"]

    if horizontal_flip["enabled"]:
        aug_config = {
            "HorizontalFlip": {
                "p": horizontal_flip["probability"]
            }
        }
    else:
        aug_config = {}

    print_experiment_config(
        model_name=model_name,
        scenario=scenario,
        resolution=resolution,
        epochs=epochs,
        config=config,
        summary=summary,
        output_dir=output_dir,
        pilot=pilot,
    )

    print("\nInitializing RF-DETR model...")

    model_class = MODEL_CLASSES[model_name]
    model = model_class()

    print("Starting RF-DETR training...")

    model.train(
        dataset_dir=str(dataset_dir),
        dataset_file="yolo",

        # Training
        epochs=epochs,
        batch_size=training["batch_size"],
        grad_accum_steps=rfdetr_config["grad_accum_steps"],
        num_workers=training["workers"],

        # Reproducibility
        seed=config["experiment"]["seed"],

        # Model resolution
        resolution=resolution,

        # Device
        device="cuda",

        # Classes
        class_names=config["dataset"]["class_names"],

        # Augmentation
        aug_config=aug_config,
        scale_jitter=False,
        augmentation_backend="albumentations",
        multi_scale="off",

        # Validation and testing
        eval_interval=1,
        run_test=False,

        # Checkpoint
        checkpoint_interval=1 if pilot else 10,
        output_dir=str(output_dir),

        # Training behavior
        early_stopping=False,
        use_ema=True,
        tensorboard=False,
        save_dataset_grids=False,

        # Metadata
        notes={
            "experiment": config["experiment"]["name"],
            "model": model_name,
            "scenario": scenario,
            "dataset_version": config["dataset"]["version"],
            "pilot": pilot,
        },
    )

    print("\n" + "=" * 65)
    print("RF-DETR TRAINING COMPLETED")
    print("=" * 65)

    print(f"Model       : {model_name}")
    print(f"Scenario    : {scenario}")
    print(f"Output      : {output_dir}")

    expected_files = [
        "checkpoint_best_ema.pth",
        "checkpoint_best_total.pth",
        "last_ema.pth",
        "metrics.csv",
        "training_config.json",
    ]

    for filename in expected_files:
        path = output_dir / filename
        print(f"{filename:30s}: {path.exists()}")

    if pilot:
        print("\nPILOT RESULT ONLY.")
        print("This is NOT a baseline research result.")

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR baseline experiment runner"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=SUPPORTED_SCENARIOS,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config()

    check_environment()

    dataset_dir, summary = validate_dataset(
        args.scenario,
        config,
    )

    resolution = get_model_resolution(
        args.model,
        config,
    )

    epochs = 1 if args.pilot else config["training"]["epochs"]

    output_dir = get_output_directory(
        args.model,
        args.scenario,
        args.pilot,
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}"
        )

    print_experiment_config(
        model_name=args.model,
        scenario=args.scenario,
        resolution=resolution,
        epochs=epochs,
        config=config,
        summary=summary,
        output_dir=output_dir,
        pilot=args.pilot,
    )

    if args.dry_run:
        print("\nDRY RUN")
        print("Training was NOT started.")
        return

    train_rfdetr(
        model_name=args.model,
        scenario=args.scenario,
        config=config,
        pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
