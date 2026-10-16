import shutil
from pathlib import Path

import cv2
import numpy as np


SOURCE_IMAGES_DIR = "dataset/test/images"
SOURCE_LABELS_DIR = "dataset/test/labels"
OUTPUT_BASE_DIR = "degraded/sneg"

# Nivoi jacine snega - ista konvencija kao mrak/magla (0.0-1.0)
SNOW_LEVELS = {
    "sneg_ssim_97": 0.015,
    "sneg_ssim_95": 0.072,
    "sneg_ssim_92": 0.118,
    "sneg_ssim_90": 0.166,
    "sneg_ssim_87": 0.216,
    "sneg_ssim_85": 0.263,
    "sneg_ssim_82": 0.313,
    "sneg_ssim_80": 0.365,
    "sneg_ssim_77": 0.416,
    "sneg_ssim_75": 0.467,
    "sneg_ssim_72": 0.515,
    "sneg_ssim_70": 0.568,
    "sneg_ssim_67": 0.619,
    "sneg_ssim_65": 0.67,
    "sneg_ssim_62": 0.722,
    "sneg_ssim_60": 0.766,
    "sneg_ssim_57": 0.795,
    "sneg_ssim_55": 0.813,
    "sneg_ssim_52": 0.83,
    "sneg_ssim_50": 0.849,
    "sneg_ssim_47": 0.87,
    "sneg_ssim_45": 0.897,
    "sneg_ssim_42": 0.95,
    "sneg_ssim_40": 1.012,
    "sneg_ssim_37": 1.081,
    "sneg_ssim_35": 1.154,
    "sneg_ssim_32": 1.228,
    "sneg_ssim_30": 1.3,
    "sneg_ssim_27": 1.381,
}

# Fiksni seed za pahulje - ista pozicija pahulja na istom nivou jacine
# svaki put kad se skripta pokrene (ponovljivost)
SNOW_SEED = 42



def apply_snow_haze(image_rgb: np.ndarray, severity: float) -> np.ndarray:
    """Blaga, hladno-plavkasta 'izmaglica' preko cele slike - atmosfera
    puna sitnih pahulja u vazduhu. Blagi, ravnomeran efekat (bez gradijenta
    po visini kao kod magle - sneg u vazduhu je ravnomernije rasporedjen)."""
    snow_haze_color = np.array([235, 240, 245], dtype=np.float32)  # hladno bela
    # ISPRAVKA: alpha ranije NIJE bila ogranicena na max 1.0. Pošto sada
    # testiramo severity i preko 1.0 (do 2.5) radi kalibracije, alpha je
    # znala preci 1.0 (npr. severity=2.1 -> alpha=1.47), sto je u mesanju
    # boja pravilo "preokret" i cudne artefakte (otuda neujednacena,
    # neopadajuca SSIM kriva na ekstremnim vrednostima).
    alpha = min(severity * 0.7, 1.0)

    image_f = image_rgb.astype(np.float32)
    hazy = image_f * (1 - alpha) + snow_haze_color * alpha
    return np.clip(hazy, 0, 255).astype(np.uint8)


def add_snowflakes(image_rgb: np.ndarray, severity: float, seed: int) -> np.ndarray:
    """Dodaje pojedinacne pahulje, sa MEKIM ivicama i varijabilnom
    providnoscu (ne tvrdi, identicni beli kruzici kao u prvoj verziji -
    to je izgledalo kao sablonski sum, ne kao prirodan sneg).

    Pristup: pahulje se crtaju na posebnom "sloju" (grayscale maska),
    zatim se taj sloj blago zamuti (meke ivice) i blaguje preko slike
    sa providnoscu koja varira po pahulji - vecina pahulja je sitna i
    bleda (dalje u pozadini), manji broj je veci i izrazeniji (blize
    kameri) - to daje utisak dubine, ne ravnomeran "tapet" od tacaka.
    Blaga vertikalna elongacija simulira pravac padanja.
    """
    h, w = image_rgb.shape[:2]
    rng = np.random.default_rng(seed)

    # Sloj pahulja - pocinje od nule (crno = bez pahulje)
    flake_layer = np.zeros((h, w), dtype=np.float32)

    # Prethodna formula (severity^1.2) je davala premalo pahulja na nizim
    # nivoima (npr. 20% je delovalo skoro prazno). Sada ima fiksnu
    # "bazu" gustine koja postoji cak i na niskim nivoima, plus rast.
    density = (0.15 + severity) * 0.03
    num_flakes = int(density * h * w / 10)

    for _ in range(num_flakes):
        x = rng.integers(0, w)
        y = rng.integers(0, h)

        # Eksponencijalna raspodela velicine - VECINA pahulja sitna,
        # retke su vece (realisticnija distribucija nego uniformna)
        size_roll = rng.random()
        if size_roll > 0.92:
            radius = rng.integers(3, 5 + int(severity * 4))  # retke, krupne (blize kameri)
            opacity = rng.uniform(0.6, 0.9)
        else:
            radius = rng.integers(1, 3)  # vecina, sitne (dalje u pozadini)
            opacity = rng.uniform(0.25, 0.55)

        # Blaga vertikalna elongacija (elipsa umesto kruga) - pravac padanja
        axes = (radius, int(radius * rng.uniform(1.3, 1.8)))
        angle = rng.uniform(-15, 15)  # blago nakrivljeno, ne savrseno vertikalno

        cv2.ellipse(flake_layer, (x, y), axes, angle, 0, 360, opacity, -1, lineType=cv2.LINE_AA)

    # Meke ivice - zamucenje celog sloja pahulja (kljucna razlika u odnosu
    # na prvu verziju sa ostrim, "iseckanim" krugovima)
    flake_layer = cv2.GaussianBlur(flake_layer, (3, 3), 0)

    # BOKEH pahulje - par velikih, jako zamucenih, svetlih mrlja koje
    # simuliraju pahulje PREBLIZU kamere da bi bile u fokusu. Ovo je
    # vizuelno vrlo prepoznatljivo kod pravih fotografija snega, a
    # prethodnoj verziji je potpuno nedostajalo (sve pahulje su bile
    # "ostre", kao da je sve u fokusu, sto nije realno).
    bokeh_layer = np.zeros((h, w), dtype=np.float32)
    num_bokeh = int(2 + severity * 10)  # malo njih, ali vizuelno upadljive
    for _ in range(num_bokeh):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        radius = rng.integers(12, 15 + int(severity * 25))
        opacity = rng.uniform(0.15, 0.35)  # blage, providne mrlje
        cv2.circle(bokeh_layer, (x, y), radius, opacity, -1, lineType=cv2.LINE_AA)
    # ISPRAVKA: ranije je ovde koriscena promenljiva 'radius' - ostatak iz
    # for petlje iznad (poslednja nasumicno izvucena vrednost), sto je
    # unosilo nepredvidivu nedoslednost u zamucenje (narocito pri visokim
    # jacinama gde ima malo bokeh pahulja, pa je taj "ostatak" nestabilan).
    # Sada je sigma fiksno vezana za severity, deterministicki.
    bokeh_blur_sigma = 6 + severity * 8
    bokeh_layer = cv2.GaussianBlur(bokeh_layer, (0, 0), sigmaX=bokeh_blur_sigma)

    # Kombinuju se oba sloja (ostre sitne pahulje + mutne bokeh mrlje)
    combined = np.clip(flake_layer + bokeh_layer, 0, 1)

    alpha = combined[:, :, None]
    result = image_rgb.astype(np.float32) * (1 - alpha) + 255 * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_snow(image_rgb: np.ndarray, severity: float, seed: int) -> np.ndarray:
    hazy = apply_snow_haze(image_rgb, severity)
    snowy = add_snowflakes(hazy, severity, seed)

    # Blago zamucenje samo pri vrlo jakom snegu (gust sneg malo zamuti konture)
    # ISPRAVKA: ranije se zamucenje naglo ukljucivalo na severity=0.7, sto je
    # pravilo skok u SSIM krivoj. Sada sigma raste glatko od 0 (na 0.7) do
    # 0.8 (na 0.9) i tu ostaje.
    sigma = 0.8 * min(max(severity - 0.7, 0.0) / 0.2, 1.0)
    if sigma > 0:
        snowy = cv2.GaussianBlur(snowy, (0, 0), sigmaX=sigma)

    return snowy


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
        degraded = apply_snow(image_rgb, severity, SNOW_SEED)
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
    print(f"Generisem {len(SNOW_LEVELS)} nivoa snega...\n")

    for level_name, severity in SNOW_LEVELS.items():
        print(f"Obradjujem nivo: {level_name} (severity={severity})")
        process_level(level_name, severity, image_paths, labels_dir)

    print("\nGotovo. Svi nivoi snega generisani u 'degraded/' folderu.")


if __name__ == "__main__":
    main()