import shutil
from pathlib import Path

import cv2
import numpy as np


SOURCE_IMAGES_DIR = "dataset/test/images"
SOURCE_LABELS_DIR = "dataset/test/labels"
OUTPUT_BASE_DIR = "degraded/blur"

# Nivoi jacine motion blur-a (0.0-1.0), ista konvencija kao mrak/magla.
# Gornja granica je malo "pojacana" u odnosu na realno vozilo u pokretu
# (realno vozilo retko daje toliko ekstreman blur), na zahtev.
BLUR_LEVELS = {
    "blur_ssim_90": 0.015,
    "blur_ssim_87": 0.037,
    "blur_ssim_85": 0.06,
    "blur_ssim_82": 0.088,
    "blur_ssim_80": 0.12,
    "blur_ssim_77": 0.153,
    "blur_ssim_75": 0.195,
    "blur_ssim_72": 0.24,
    "blur_ssim_70": 0.297,
    "blur_ssim_67": 0.365,
    "blur_ssim_65": 0.452,
    "blur_ssim_62": 0.564,
    "blur_ssim_60": 0.718,
    "blur_ssim_57": 0.927,
}

# Ugao pravca kretanja u stepenima (0 = horizontalno - tipicno za vozilo
# koje prolazi ispred kamere). Mozes promeniti ako zelis drugi pravac.
MOTION_ANGLE_DEGREES = 0



def make_motion_kernel(kernel_size: int, angle_degrees: float) -> np.ndarray:
    """Pravi usmereni 'motion blur' kernel - liniju piksela u pravcu
    kretanja, normalizovanu da zbir bude 1 (ne menja ukupnu svetlinu slike)."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0  # horizontalna linija kroz sredinu

    if angle_degrees != 0:
        center = (kernel_size / 2, kernel_size / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        kernel = cv2.warpAffine(kernel, rot_matrix, (kernel_size, kernel_size))

    kernel = kernel / kernel.sum()
    return kernel


def apply_motion_blur(image_rgb: np.ndarray, severity: float) -> np.ndarray:
    """Primenjuje motion blur, sa duzinom kernela skaliranom prema severity.

    Realan opseg: ~3px (jedva primetno) do ~27px (jako zamucenje, vidljivo
    ali jos uvek prepoznatljiv oblik objekta - iznad ovoga bi slika postala
    besmislena mrlja, sto nije realno cak ni za brzo vozilo).
    """
    # ISPRAVKA: duzina zamucenja je sada KONTINUALNA (realan broj), a ne ceo
    # broj piksela. Ranije su susedni nivoi cesto davali isti ceo broj (pa
    # IDENTICNE slike), a kriva nije bila monotona. Sada je krajnji piksel
    # linije uzet sa razlomljenom tezinom, pa se efekat menja glatko.
    length = 3.0 + severity * 24.0  # ~3 do ~27 px (originalna, realna granica)
    half = int(np.ceil(length / 2.0))
    kernel_size = 2 * half + 1  # neparan

    # tezina svakog piksela na liniji: 1 unutar duzine, razlomak na ivici
    offsets = np.abs(np.arange(kernel_size) - half)
    line_weights = np.clip(length / 2.0 - offsets + 0.5, 0.0, 1.0).astype(np.float32)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[half, :] = line_weights

    if MOTION_ANGLE_DEGREES != 0:
        center = (kernel_size / 2, kernel_size / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, MOTION_ANGLE_DEGREES, 1.0)
        kernel = cv2.warpAffine(kernel, rot_matrix, (kernel_size, kernel_size))

    kernel = kernel / kernel.sum()
    blurred = cv2.filter2D(image_rgb, -1, kernel)
    return blurred


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
        degraded = apply_motion_blur(image_rgb, severity)
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
    print(f"Generisem {len(BLUR_LEVELS)} nivoa motion blur-a...\n")

    for level_name, severity in BLUR_LEVELS.items():
        print(f"Obradjujem nivo: {level_name} (severity={severity})")
        process_level(level_name, severity, image_paths, labels_dir)

    print("\nGotovo. Svi nivoi motion blur-a generisani u 'degraded/' folderu.")


if __name__ == "__main__":
    main()