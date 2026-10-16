import shutil
from pathlib import Path

import albumentations as A
import cv2


# Odakle uzimamo CISTE slike za degradaciju (obicno test skup)
SOURCE_IMAGES_DIR = "dataset/test/images"
SOURCE_LABELS_DIR = "dataset/test/labels"

# Gde cuvamo degradirane verzije
OUTPUT_BASE_DIR = "degraded/mrak"

# Nivoi mraka: ime foldera -> vrednost "brightness_limit" za albumentations
# (negativna = tamnije, sto je broj negativniji to je mrak jaci).
# Ime nivoa kodira CILJNI SSIM (npr. mrak_ssim_80 = SSIM prema originalu ~0.80),
# a brojeve je izracunala calibrate_ssim.py - ne biraju se rucno.
DARKNESS_LEVELS = {
    "mrak_ssim_97": -0.022,
    "mrak_ssim_95": -0.037,
    "mrak_ssim_92": -0.052,
    "mrak_ssim_90": -0.064,
    "mrak_ssim_87": -0.076,
    "mrak_ssim_85": -0.09,
    "mrak_ssim_82": -0.1,
    "mrak_ssim_80": -0.11,
    "mrak_ssim_77": -0.123,
    "mrak_ssim_75": -0.135,
    "mrak_ssim_72": -0.147,
    "mrak_ssim_70": -0.16,
    "mrak_ssim_67": -0.171,
    "mrak_ssim_65": -0.184,
    "mrak_ssim_62": -0.197,
    "mrak_ssim_60": -0.209,
    "mrak_ssim_57": -0.22,
    "mrak_ssim_55": -0.234,
    "mrak_ssim_52": -0.248,
    "mrak_ssim_50": -0.26,
    "mrak_ssim_47": -0.275,
    "mrak_ssim_45": -0.291,
    "mrak_ssim_42": -0.303,
    "mrak_ssim_40": -0.318,
    "mrak_ssim_37": -0.333,
    "mrak_ssim_35": -0.349,
    "mrak_ssim_32": -0.366,
    "mrak_ssim_30": -0.384,
    "mrak_ssim_27": -0.403,
    "mrak_ssim_25": -0.419,
    "mrak_ssim_22": -0.439,
    "mrak_ssim_20": -0.461,
    "mrak_ssim_17": -0.485,
    "mrak_ssim_15": -0.508,
    "mrak_ssim_12": -0.538,
    "mrak_ssim_10": -0.569,
    "mrak_ssim_7": -0.608,
    "mrak_ssim_5": -0.657,
    "mrak_ssim_2": -0.738,
}



def make_darkness_transform(brightness_limit: float) -> A.Compose:
    """Pravi albumentations transformaciju koja SAMO smanjuje osvetljenje,
    za fiksnu (ne nasumicnu) jacinu - zato su min i max limit isti."""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(brightness_limit, brightness_limit),
            contrast_limit=0,
            p=1.0,  # p=1.0 znaci "uvek primeni", nema nasumicnosti da li ce se desiti
        )
    ])


def process_level(level_name: str, brightness_limit: float, image_paths, labels_dir: Path):
    out_images_dir = Path(OUTPUT_BASE_DIR) / level_name / "images"
    out_labels_dir = Path(OUTPUT_BASE_DIR) / level_name / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    transform = make_darkness_transform(brightness_limit)

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Upozorenje: ne mogu da ucitam {img_path.name}, preskacem.")
            continue

        # albumentations ocekuje RGB, OpenCV ucitava kao BGR - konverzija
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        degraded = transform(image=image_rgb)["image"]
        degraded_bgr = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)

        # Cuvanje zatamnjene slike
        out_img_path = out_images_dir / img_path.name
        cv2.imwrite(str(out_img_path), degraded_bgr)

        # Kopiranje NEPROMENJENE anotacije (tablica je i dalje na istom mestu)
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
    print(f"Generisem {len(DARKNESS_LEVELS)} nivoa mraka...\n")

    for level_name, brightness_limit in DARKNESS_LEVELS.items():
        print(f"Obradjujem nivo: {level_name} (brightness_limit={brightness_limit})")
        process_level(level_name, brightness_limit, image_paths, labels_dir)

    print("\nGotovo. Svi nivoi zatamnjenja generisani u 'degraded/' folderu.")


if __name__ == "__main__":
    main()