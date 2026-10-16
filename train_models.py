import time
from pathlib import Path
from ultralytics import YOLO



DATA_YAML = "dataset/data.yaml"

MODEL_SIZES = ["n", "s", "m", "l", "x"]

EPOCHS = 100

# Velicina slike (YOLO standard je 640) - isto mora biti isto za sve
IMAGE_SIZE = 640

# Batch size - RTX 3090 (24GB) moze da izdrzi veliki batch;
BATCH_SIZE = 16

# Naziv - koristi se kao prefiks foldera u runs/train/
EXPERIMENT_NAME = "tablice"

# Fiksni seed - za ponovljivost treninga (isti raspored batch-eva itd.)
SEED = 42



def main():
    data_yaml_path = Path(DATA_YAML)

    if not data_yaml_path.exists():
        print(f"GRESKA: {data_yaml_path.resolve()} ne postoji.")
        print("Pokreni prvo organize_dataset.py da napravis dataset i data.yaml.")
        return

    print(f"Koristim dataset: {data_yaml_path.resolve()}")
    print(f"Treniram modele: {MODEL_SIZES}")
    print(f"Epohe: {EPOCHS}, Velicina slike: {IMAGE_SIZE}, Batch: {BATCH_SIZE}\n")

    training_times = {}

    for size in MODEL_SIZES:
        model_name = f"models/yolo11{size}.pt"  # bazni modeli su u models/ folderu
        run_name = f"{EXPERIMENT_NAME}_{size}"

        print(f"{'=' * 60}")
        print(f"Pocinje trening: YOLOv11{size.upper()}  ({model_name})")
        print(f"{'=' * 60}")

        start_time = time.time()

        model = YOLO(model_name)

        model.train(
            data=str(data_yaml_path),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            seed=SEED,
            project=str(Path("runs/train").resolve()),
            name=run_name,
            exist_ok=True,  # ne baca gresku ako folder vec postoji
        )

        elapsed = time.time() - start_time
        training_times[size] = elapsed

        actual_dir = Path(model.trainer.save_dir)
        best_pt = actual_dir / "weights" / "best.pt"
        print(f"\nYOLOv11{size.upper()} zavrsen za {elapsed / 60:.1f} minuta.")
        print(f"Rezultati: {actual_dir}")
        print(f"best.pt postoji: {best_pt.exists()}  ({best_pt})\n")


    print(f"\n{'=' * 60}")
    print("SVI MODELI ZAVRSENI")
    print(f"{'=' * 60}")
    for size, seconds in training_times.items():
        print(f"  YOLOv11{size.upper()}: {seconds / 60:.1f} min")

    total = sum(training_times.values())
    print(f"\nUkupno vreme treninga svih modela: {total / 60:.1f} minuta ({total / 3600:.2f} h)")


if __name__ == "__main__":
    main()