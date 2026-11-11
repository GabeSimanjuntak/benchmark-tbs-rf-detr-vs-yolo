
from pathlib import Path
import shutil
import yaml
import pandas as pd

# =====================================================
# 1. PROJECT DIRECTORY
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

FILTERED_DIR = PROJECT_DIR / "datasets" / "processed" / "filtered"
MALAYSIA_DIR = PROJECT_DIR / "datasets" / "processed" / "dataset-malaysia"

OUTPUT_ROOT = PROJECT_DIR / "datasets" / "experiments"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "Unripe",
    1: "Ripe"
}

# =====================================================
# 2. DEFINISI SUMBER DATA
# =====================================================

SOURCES = {
    "pabrik": FILTERED_DIR / "dataset-pabrik-kalsel",
    "kalteng": FILTERED_DIR / "dataset-perkebunan-kalteng",
    "malaysia": MALAYSIA_DIR,
}

# =====================================================
# 3. DEFINISI SKENARIO LODO
# =====================================================

SCENARIOS = {
    "lodo_1_pabrik": {
        "train": [
            ("kalteng", "train"),
            ("malaysia", "train"),
        ],
        "val": [
            ("kalteng", "valid"),
        ],
        "test": [
            ("pabrik", "test"),
        ],
    },

    "lodo_2_kalteng": {
        "train": [
            ("pabrik", "train"),
            ("malaysia", "train"),
        ],
        "val": [
            ("pabrik", "valid"),
        ],
        "test": [
            ("kalteng", "test"),
        ],
    },

    "lodo_3_malaysia": {
        "train": [
            ("pabrik", "train"),
            ("kalteng", "train"),
        ],
        "val": [
            ("pabrik", "valid"),
            ("kalteng", "valid"),
        ],
        "test": [
            ("malaysia", "test"),
        ],
    },
}

# =====================================================
# 4. FUNGSI MENYALIN DATASET
# =====================================================

def copy_dataset_split(
    dataset_name,
    source_split,
    destination_split,
    scenario_dir
):
    source_dir = SOURCES[dataset_name]

    source_images = source_dir / source_split / "images"
    source_labels = source_dir / source_split / "labels"

    destination_images = (
        scenario_dir / destination_split / "images"
    )

    destination_labels = (
        scenario_dir / destination_split / "labels"
    )

    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    if not source_images.exists():
        raise FileNotFoundError(source_images)

    if not source_labels.exists():
        raise FileNotFoundError(source_labels)

    image_files = [
        p for p in source_images.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    copied = 0

    for image_path in image_files:

        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label tidak ditemukan: {label_path}"
            )

        # Nama file unik berdasarkan sumber dataset
        new_stem = f"{dataset_name}_{image_path.stem}"

        new_image = (
            destination_images /
            f"{new_stem}{image_path.suffix.lower()}"
        )

        new_label = destination_labels / f"{new_stem}.txt"

        shutil.copy2(image_path, new_image)
        shutil.copy2(label_path, new_label)

        copied += 1

    return copied


# =====================================================
# 5. MEMBANGUN TIGA SKENARIO
# =====================================================

summary = []

for scenario_name, configuration in SCENARIOS.items():

    scenario_dir = OUTPUT_ROOT / scenario_name

    print(f"\nMembangun {scenario_name}...")

    for destination_split, source_list in configuration.items():

        for dataset_name, source_split in source_list:

            copied = copy_dataset_split(
                dataset_name=dataset_name,
                source_split=source_split,
                destination_split=destination_split,
                scenario_dir=scenario_dir
            )

            summary.append({
                "scenario": scenario_name,
                "split": destination_split,
                "source": dataset_name,
                "source_split": source_split,
                "images_copied": copied,
            })

    # Membuat data.yaml
    yaml_config = {
        "path": ".",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": CLASS_NAMES,
    }

    yaml_path = scenario_dir / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            yaml_config,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"Berhasil: {scenario_dir}")


# =====================================================
# 6. LAPORAN HASIL
# =====================================================

summary_df = pd.DataFrame(summary)

print(summary_df.to_string(index=False))

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

summary_path = OUTPUT_ROOT / "lodo_build_summary.csv"

summary_df.to_csv(summary_path, index=False)

print("\nSemua skenario selesai dibangun.")
print("Lokasi:", OUTPUT_ROOT)
print("Ringkasan:", summary_path)
