import shutil
from pathlib import Path

import cv2
import numpy as np


SOURCE_IMAGES_DIR = "dataset/test/images"
SOURCE_LABELS_DIR = "dataset/test/labels"
OUTPUT_BASE_DIR = "degraded/kompresija"

# Nivoi jacine - ista konvencija kao mrak/magla/sneg (0.0-1.0)
COMPRESSION_LEVELS = {
    "komp_ssim_90": 0.071,
    "komp_ssim_87": 0.152,
    "komp_ssim_85": 0.251,
    "komp_ssim_82": 0.343,
    "komp_ssim_80": 0.424,
    "komp_ssim_77": 0.499,
    "komp_ssim_75": 0.561,
    "komp_ssim_72": 0.612,
    "komp_ssim_70": 0.657,
    "komp_ssim_67": 0.698,
    "komp_ssim_65": 0.732,
    "komp_ssim_62": 0.764,
    "komp_ssim_60": 0.793,
    "komp_ssim_57": 0.82,
    "komp_ssim_55": 0.847,
    "komp_ssim_52": 0.87,
    "komp_ssim_50": 0.892,
    "komp_ssim_47": 0.913,
    "komp_ssim_45": 0.93,
    "komp_ssim_42": 0.951,
    "komp_ssim_40": 0.963,
}



def apply_downscale_upscale(image: np.ndarray, severity: float) -> np.ndarray:
    """Smanjuje pa vraca sliku na originalnu velicinu - trajno gubi detalj.
    Jacina 0 = bez promene, 1 = smanjeno na svega ~10% originalne velicine."""
    h, w = image.shape[:2]

    # Faktor smanjenja: pri severity=0 ostaje 100% velicine, pri severity=1
    # pada na ~12% velicine (vrlo nisko-rezolucijska slika)
    # Donja granica spustena sa 0.12 na 0.04 - vrlo losa, jeftina nadzorna
    # kamera realno moze imati ekstremno nisku efektivnu rezoluciju
    scale_factor = 1.0 - severity * 0.96
    scale_factor = max(scale_factor, 0.04)

    small_w = max(1, int(w * scale_factor))
    small_h = max(1, int(h * scale_factor))

    # Smanjenje (INTER_AREA je najbolji izbor za downscale - manje aliasing-a)
    downscaled = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Uvecanje nazad na original (INTER_LINEAR - standardno, ne previse "ostro")
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)

    return upscaled


def apply_jpeg_compression(image: np.ndarray, severity: float) -> np.ndarray:
    """Komprimuje sliku kao JPEG sa niskim quality parametrom i vraca
    dekompresovanu verziju - uvodi blokovske artefakte."""
    # JPEG quality ide od 100 (bez gubitka) do ~5 (jako vidljivi artefakti).
    # severity=0 -> quality blizu 95 (skoro neprimetno), severity=1 -> quality 5
    # Donja granica spustena sa 5 na 1 - realan ekstrem za vrlo losu
    # kompresiju (JPEG kvalitet 1 i dalje postoji kao validna vrednost)
    quality = int(95 - severity * 94)
    quality = max(quality, 1)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode(".jpg", image, encode_params)
    if not success:
        return image

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def apply_compression_degradation(image_rgb: np.ndarray, severity: float) -> np.ndarray:
    # Radi sa BGR (OpenCV standard) za JPEG deo, pa vracamo RGB na kraju
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    degraded = apply_downscale_upscale(image_bgr, severity)
    degraded = apply_jpeg_compression(degraded, severity)

    return cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)


def process_level(level_name: str, severity: float, image_paths, labels_dir: Path):
    out_images_dir = Path(OUTPUT_BASE_DIR) / level_name / "images"
    out_labels_dir = Path(OUTPUT_BASE_DIR) / level_name / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Upozorenje: ne mogu da ucitam {img_path.name}, preskacem.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        degraded = apply_compression_degradation(image_rgb, severity)
        degraded_bgr = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)

        out_img_path = out_images_dir / img_path.name
        cv2.imwrite(str(out_img_path), degraded_bgr)

        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, out_labels_dir / label_path.name)
        else:
            print(f"  Upozorenje: nema anotacije za {img_path.name}")

    print(f"  {level_name}: {len(image_paths)} slika sacuvano u {out_images_dir.parent.resolve()}")


def main():
    images_dir = Path(SOURCE_IMAGES_DIR)
    labels_dir = Path(SOURCE_LABELS_DIR)

    if not images_dir.exists():
        print(f"GRESKA: {images_dir.resolve()} ne postoji.")
        print("Pokreni prvo organize_dataset.py, ili promeni SOURCE_IMAGES_DIR.")
        return

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    if not image_paths:
        print(f"Nema slika u {images_dir.resolve()}")
        return

    print(f"Pronadjeno {len(image_paths)} slika u test skupu.")
    print(f"Generisem {len(COMPRESSION_LEVELS)} nivoa kompresije...\n")

    for level_name, severity in COMPRESSION_LEVELS.items():
        print(f"Obradjujem nivo: {level_name} (severity={severity})")
        process_level(level_name, severity, image_paths, labels_dir)

    print("\nGotovo. Svi nivoi kompresije generisani u 'degraded/' folderu.")


if __name__ == "__main__":
    main()