import shutil
from pathlib import Path

import cv2
import numpy as np


SOURCE_IMAGES_DIR = "dataset/test/images"
SOURCE_LABELS_DIR = "dataset/test/labels"
OUTPUT_BASE_DIR = "degraded/magla"

# Nivoi jacine magle - isti brojevi kao kod mraka, radi doslednosti (0.0-1.0)
FOG_LEVELS = {
    "magla_ssim_97": 0.07,
    "magla_ssim_95": 0.118,
    "magla_ssim_92": 0.165,
    "magla_ssim_90": 0.211,
    "magla_ssim_87": 0.256,
    "magla_ssim_85": 0.299,
    "magla_ssim_82": 0.341,
    "magla_ssim_80": 0.381,
    "magla_ssim_77": 0.42,
    "magla_ssim_75": 0.458,
    "magla_ssim_72": 0.494,
    "magla_ssim_70": 0.528,
    "magla_ssim_67": 0.562,
    "magla_ssim_65": 0.594,
    "magla_ssim_62": 0.626,
    "magla_ssim_60": 0.652,
    "magla_ssim_57": 0.671,
    "magla_ssim_55": 0.69,
    "magla_ssim_52": 0.707,
    "magla_ssim_50": 0.727,
    "magla_ssim_47": 0.747,
    "magla_ssim_45": 0.77,
    "magla_ssim_42": 0.794,
    "magla_ssim_40": 0.823,
    "magla_ssim_37": 0.855,
    "magla_ssim_35": 0.891,
    "magla_ssim_32": 0.941,
    "magla_ssim_30": 1.031,
}



def apply_fog(image_rgb: np.ndarray, severity: float) -> np.ndarray:
    """Simulacija magle - mesanje slike sa svetlo-sivom bojom, sa blagim
    gradijentom po visini (vrh slike - dalje, dole - blize, manje magle).
    Mala doza desaturacije za realisticniji utisak, bez izrazenog suma
    ili drasticnog zamucenja.
    """
    h, w = image_rgb.shape[:2]
    fog_color = np.array([200, 200, 200], dtype=np.float32)

    # Blaga varijacija po visini (vrh malo gusci, dno malo redji magla),
    # ali vise NE pada drasticno (ranije 1.0->0.3, zbog cega je npr.
    # severity=0.5 na dnu slike davalo samo ~15% stvarnog efekta).
    # Sada opseg 1.15->0.85 znaci da je "severity" broj skoro direktno
    # stvarna vidljiva jacina magle na celoj slici.
    gradient = np.linspace(1.15, 0.85, h, dtype=np.float32).reshape(h, 1)
    # Plafon podignut sa 0.97 na 1.0 - gusta magla u stvarnosti MOZE
    # prakticno potpuno zakloniti vidljivost (whiteout uslovi su realni)
    alpha = np.clip(gradient * severity, 0, 1.0)[:, :, None]

    # Vrlo blaga desaturacija (mnogo manja nego u prethodnom pokusaju)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_3ch = np.stack([gray] * 3, axis=-1).astype(np.float32)
    desaturation_amount = 0.12 * severity  # bilo 0.35 - sad mnogo suptilnije
    image_f = image_rgb.astype(np.float32) * (1 - desaturation_amount) + gray_3ch * desaturation_amount

    foggy = image_f * (1 - alpha) + fog_color * alpha
    foggy = np.clip(foggy, 0, 255).astype(np.uint8)

    # Zamucenje sad SKALIRA sa jacinom (bilo fiksno 3x3) - pri ekstremnoj
    # gustoj magli (whiteout) realno dolazi i do jaceg gubitka ostrine
    # ISPRAVKA: ranije se zamucenje naglo "ukljucivalo" na severity=0.5
    # (i ponovo skakalo na 0.8 i 1.1, kad se menjala velicina kernela), sto je
    # pravilo skokove u SSIM krivoj. Sada sigma raste GLATKO od 0 (na 0.5) do
    # 1.4 (na 1.1), pa nema skokova.
    sigma = 1.4 * min(max(severity - 0.5, 0.0) / 0.6, 1.0)
    if sigma > 0:
        foggy = cv2.GaussianBlur(foggy, (0, 0), sigmaX=sigma)

    return foggy


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
        degraded = apply_fog(image_rgb, severity)
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
    print(f"Generisem {len(FOG_LEVELS)} nivoa magle...\n")

    for level_name, severity in FOG_LEVELS.items():
        print(f"Obradjujem nivo: {level_name} (severity={severity})")
        process_level(level_name, severity, image_paths, labels_dir)

    print("\nGotovo. Svi nivoi magle generisani u 'degraded/' folderu.")


if __name__ == "__main__":
    main()