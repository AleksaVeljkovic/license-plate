import csv
import re
import time
from pathlib import Path

import yaml
from ultralytics import YOLO


# Koje velicine modela evaluiramo. Putanja do best.pt se trazi automatski
# (vidi find_best_weights) - skripta preskace velicinu ako model ne postoji.
MODEL_SIZES = ["n", "s", "m", "l", "x"]

WEIGHTS_CANDIDATES = [
    "runs/train/tablice_{size}/weights/best.pt",
    "runs/detect/runs/train/tablice_{size}/weights/best.pt",
]


def find_best_weights(size: str):
    """Vraca putanju do best.pt za datu velicinu modela, ili None ako ne postoji."""
    for template in WEIGHTS_CANDIDATES:
        candidate = Path(template.format(size=size))
        if candidate.exists():
            return candidate
    return None


# Recnik {velicina: putanja} samo za modele koji STVARNO postoje
MODEL_PATHS = {
    size: str(path)
    for size in MODEL_SIZES
    if (path := find_best_weights(size)) is not None
}

# Cist (nedegradirani) test skup - koristi se kao baseline/referenca
CLEAN_TEST_IMAGES = "dataset/test_clean/images"

# Glavni folder u kom se nalaze svi degradirani skupovi (iz degrade_*.py)
DEGRADED_ROOT = "degraded_clean"

# Odavde se cita broj klasa i njihova imena (nc, names)
ORIGINAL_DATA_YAML = "dataset/data.yaml"

# Gde se upisuje finalna tabela rezultata
OUTPUT_CSV = "evaluation_results_clean.csv"

# Privremeni folder za yaml fajlove koje skripta sama generise (brisu se na kraju)
TEMP_YAML_DIR = Path("eval_tmp")



def load_class_info(data_yaml_path: str):
    """Cita nc (broj klasa) i names (imena klasa) iz postojeceg data.yaml,
    i putanju do train foldera (treba nam samo kao "dummy" vrednost - YOLO
    yaml format zahteva da postoji, iako je pri evaluaciji ne koristimo)."""
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["nc"], data["names"], data["train"]


def discover_degraded_folders(root: str):
    """Automatski pronalazi sve degradirane foldere u strukturi:
    degraded/<tip>/<nivo>/images
    Vraca listu (tip, nivo, putanja_do_images_foldera)."""
    root_path = Path(root)
    found = []

    if not root_path.exists():
        print(f"GRESKA: {root_path.resolve()} ne postoji.")
        return found

    for type_dir in sorted(root_path.iterdir()):
        if not type_dir.is_dir():
            continue
        for level_dir in sorted(type_dir.iterdir()):
            images_dir = level_dir / "images"
            if images_dir.exists():
                found.append((type_dir.name, level_dir.name, images_dir))

    return found


def extract_ssim_from_name(level_name: str) -> str:
    """Izvlaci SSIM broj iz imena foldera, npr. 'mrak_ssim_85' -> '0.85'.
    Ako ne uspe da parsira (npr. baseline), vraca prazan string."""
    match = re.search(r"ssim_(\d+)", level_name)
    if match:
        return f"0.{match.group(1)}"
    return ""


def write_temp_yaml(images_dir: Path, nc: int, names, dummy_train_dir: str) -> Path:
    """Pravi privremeni data.yaml koji pokazuje na dati folder sa slikama,
    kao 'val' skup. YOLO format zahteva i 'train' kljuc - koristimo postojeci
    (originalni) train folder kao dummy vrednost, jer se pri val-only pozivu
    ne koristi za racunanje, samo mora da postoji validna putanja."""
    TEMP_YAML_DIR.mkdir(exist_ok=True)
    temp_path = TEMP_YAML_DIR / "temp_eval.yaml"

    content = {
        "train": dummy_train_dir,
        "val": str(images_dir.resolve()),
        "nc": nc,
        "names": names,
    }

    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f)

    return temp_path


def evaluate_one(model: YOLO, images_dir: Path, nc: int, names, dummy_train_dir: str):
    """Pokrece model.val() na datom folderu slika i vraca recnik sa metrikama."""
    temp_yaml = write_temp_yaml(images_dir, nc, names, dummy_train_dir)

    start = time.time()
    results = model.val(data=str(temp_yaml), split="val", verbose=False, plots=False)
    elapsed = time.time() - start

    return {
        "mAP50": round(float(results.box.map50), 4),
        "mAP50-95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "inference_ms": round(float(results.speed.get("inference", 0)), 2),
        "total_eval_sec": round(elapsed, 2),
    }


def main():
    if not MODEL_PATHS:
        print("GRESKA: nijedan istreniran model nije pronadjen (trazeno u runs/train/).")
        print("Pokreni prvo train_models.py.")
        return

    nc, names, dummy_train_dir = load_class_info(ORIGINAL_DATA_YAML)
    print(f"Klase: {names} (nc={nc})")

    degraded_folders = discover_degraded_folders(DEGRADED_ROOT)
    print(f"Pronadjeno {len(degraded_folders)} degradiranih foldera.\n")

    # Priprema CSV fajla - upisujemo zaglavlje samo ako fajl jos ne postoji
    csv_path = Path(OUTPUT_CSV)
    write_header = not csv_path.exists()

    fieldnames = [
        "model", "tip_degradacije", "nivo", "ssim",
        "mAP50", "mAP50-95", "precision", "recall",
        "inference_ms", "total_eval_sec",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for model_size, model_path in MODEL_PATHS.items():
            if not Path(model_path).exists():
                print(f"Upozorenje: model {model_path} ne postoji, preskačem.")
                continue

            print(f"\n{'=' * 50}")
            print(f"Ucitavam model: yolo11{model_size} ({model_path})")
            print(f"{'=' * 50}")
            model = YOLO(model_path)

            # 1) Baseline - cist test skup, bez degradacije
            print("Evaluacija: CISTO (baseline)...")
            clean_dir = Path(CLEAN_TEST_IMAGES)
            metrics = evaluate_one(model, clean_dir, nc, names, dummy_train_dir)
            writer.writerow({
                "model": f"yolo11{model_size}",
                "tip_degradacije": "cisto",
                "nivo": "baseline",
                "ssim": "1.0",
                **metrics,
            })
            csv_file.flush()
            print(f"  mAP50={metrics['mAP50']}  Precision={metrics['precision']}  Recall={metrics['recall']}")

            # 2) Svi degradirani uslovi
            for tip, nivo, images_dir in degraded_folders:
                print(f"Evaluacija: {tip} / {nivo}...")
                metrics = evaluate_one(model, images_dir, nc, names, dummy_train_dir)
                writer.writerow({
                    "model": f"yolo11{model_size}",
                    "tip_degradacije": tip,
                    "nivo": nivo,
                    "ssim": extract_ssim_from_name(nivo),
                    **metrics,
                })
                csv_file.flush()  # upisuje odmah, da ne izgubimo podatke ako nesto pukne usput
                print(f"  mAP50={metrics['mAP50']}  Precision={metrics['precision']}  Recall={metrics['recall']}")

    print(f"\n\nGotovo. Rezultati sacuvani u: {csv_path.resolve()}")


if __name__ == "__main__":
    main()