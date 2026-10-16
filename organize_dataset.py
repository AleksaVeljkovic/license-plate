import random
import shutil
from pathlib import Path



SOURCE_IMAGES_DIR = "data/raw_images"
SOURCE_LABELS_DIR = "data/auto_labels"  


OUTPUT_DATASET_DIR = "dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
# Fiksni "seed" - garantuje da je podela ISTA svaki put kad se skripta pokrene
RANDOM_SEED = 42

CLASS_NAMES = ["tablica"]



def find_annotated_images(images_dir: Path, labels_dir: Path):
    valid_pairs = []
    skipped = 0

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            skipped += 1

    if skipped > 0:
        print(f"Upozorenje: {skipped} slika nema odgovarajucu anotaciju - preskocene.")

    return valid_pairs


def split_dataset(pairs, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Odnosi train/val/test moraju sabrati tacno na 1.0"

    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)  # koristi fiksni seed za ponovljivost

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # test dobija sve sto ostane (izbegava gubitak slike zbog zaokruzivanja)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]

    return train_set, val_set, test_set


def copy_split(pairs, split_name: str, output_dir: Path):
    images_out = output_dir / split_name / "images"
    labels_out = output_dir / split_name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        shutil.copy2(img_path, images_out / img_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)


def write_data_yaml(output_dir: Path, class_names: list):
    yaml_content = (
        f"train: {(output_dir / 'train' / 'images').resolve().as_posix()}\n"
        f"val: {(output_dir / 'val' / 'images').resolve().as_posix()}\n"
        f"test: {(output_dir / 'test' / 'images').resolve().as_posix()}\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def main():
    images_dir = Path(SOURCE_IMAGES_DIR)
    labels_dir = Path(SOURCE_LABELS_DIR)
    output_dir = Path(OUTPUT_DATASET_DIR)

    if not images_dir.exists() or not labels_dir.exists():
        print(f"GRESKA: Proveri putanje - {images_dir.resolve()} ili {labels_dir.resolve()} ne postoje.")
        return

    pairs = find_annotated_images(images_dir, labels_dir)

    if not pairs:
        print("Nema kompletnih parova slika+anotacija. Prekidam.")
        return

    print(f"Pronadjeno {len(pairs)} kompletnih parova (slika + anotacija).")

    train_set, val_set, test_set = split_dataset(
        pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    print(f"Podela: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    print("Kopiram train...")
    copy_split(train_set, "train", output_dir)
    print("Kopiram val...")
    copy_split(val_set, "val", output_dir)
    print("Kopiram test...")
    copy_split(test_set, "test", output_dir)

    yaml_path = write_data_yaml(output_dir, CLASS_NAMES)

    print("\nGotovo.")
    print(f"Organizovan dataset: {output_dir.resolve()}")
    print(f"data.yaml sacuvan: {yaml_path.resolve()}")


if __name__ == "__main__":
    main()