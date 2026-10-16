"""
calibrate_ssim.py

Cilj: Naci koji "severity" parametar (za SVAKI tip degradacije) odgovara
zadatom SSIM-u (Structural Similarity Index) u odnosu na originalne slike,
da bi isti "nivo" (npr. SSIM=0.80) kod mraka, magle, snega, blura i
kompresije predstavljao PRIBLIZNO ISTI stvarni uticaj na sliku, mereno
objektivnom metrikom.

NOVO U OVOJ VERZIJI:
- SSIM se racuna kao PROSEK preko uzorka slika (ne na jednoj slici), pa
  "SSIM=0.80" znaci "prosecni SSIM na uzorku test slika je 0.80".
- Uz svaki nivo ispisuje se i RASPON SSIM-a po pojedinacnim slikama (min-max),
  da se vidi koliko se slike medjusobno razlikuju.
- Racunanje ide paralelno na vise jezgara procesora (inace bi trajalo preko sat).
- Proverava da li neki susedni nivoi daju IDENTICNE slike (duplikati).

Radi za SVIH PET tipova degradacije u jednom pokretanju, koristeci
POSTOJECE funkcije iz degrade_*.py skripti (stil svakog efekta ostaje
tacno onakav kakav smo vec vizuelno doterali). Ova skripta mora da stoji
u istom folderu kao degrade_*.py fajlovi.

Pokretanje (iz korena projekta):
    python degrade_scripts/calibrate_ssim.py
"""

import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Uvozimo POSTOJECE funkcije iz svake degrade_*.py skripte
from degrade_darkness import make_darkness_transform
from degrade_fog import apply_fog
from degrade_snow import apply_snow
from degrade_motion_blur import apply_motion_blur
from degrade_compression import apply_compression_degradation

# ---------------------- PODESAVANJA ----------------------

# Slike na kojima se kalibrise.
# TRENUTNO PODESENO ZA PROBU (probnih 20 slika, jer test folder ima samo 2).
# KAD STIGNU PRAVE SLIKE, ZA KONACNU KALIBRACIJU PROMENI NA:
#     SAMPLE_IMAGES_DIR = "dataset/test/images"
# (mora biti test skup, jer se samo on degradira)
SAMPLE_IMAGES_DIR = "dataset/test/images"

# Najvise koliko slika ulazi u uzorak (ako ih ima vise, bira se nasumicno uz
# fiksni seed, pa je izbor ponovljiv). Vise slika = tacnije, ali sporije.
MAX_SAMPLE_IMAGES = 30   # za probu; za konacnu kalibraciju moze 15-30
SAMPLE_SEED = 42

# Broj procesa. None = automatski (broj jezgara minus 2, najvise 16).
# Svaki proces drzi ceo uzorak slika u memoriji, pa ako su slike jako velike
# (desetine megapiksela) smanji ovaj broj.
N_WORKERS = None

# Ciljani SSIM nivoi, korak 2.5%. Svaki tip dobija samo one koje realno moze da dostigne.
TARGET_SSIM_LEVELS = [round(x, 4) for x in np.arange(0.975, 0.0, -0.025)]

SNOW_SEED = 42  # isti seed kao u degrade_snow.py

# Raspon "severity" vrednosti koji se isprobava za svaki tip
SEVERITY_GRIDS = {
    "mrak": np.arange(-0.02, -0.99, -0.01),   # negativno = tamnije
    "magla": np.arange(0.01, 1.51, 0.015),
    "sneg": np.arange(0.01, 2.51, 0.025),
    "blur": np.arange(0.01, 0.99, 0.01),
    "komp": np.arange(0.01, 1.31, 0.01),
}

DISPLAY_NAMES = {
    "mrak": "MRAK (darkness)",
    "magla": "MAGLA (fog)",
    "sneg": "SNEG (snow)",
    "blur": "ZAMUCENJE POKRETA (motion blur)",
    "komp": "KOMPRESIJA (rezolucija/JPEG)",
}

# Pod kojim se imenom promenljiva zove u odgovarajucoj degrade_*.py skripti
DICT_NAMES = {
    "mrak": "DARKNESS_LEVELS",
    "magla": "FOG_LEVELS",
    "sneg": "SNOW_LEVELS",
    "blur": "BLUR_LEVELS",
    "komp": "COMPRESSION_LEVELS",
}

# -----------------------------------------------------------


# ---------------------- Wrapper funkcije ----------------------
# Svaka ima isti potpis (slika, severity) -> degradirana slika, jer postojece
# funkcije imaju malo razlicite pozive. Moraju biti na nivou modula
# (ne unutar funkcija), da bi ih procesi-radnici mogli da pronadju.

def apply_darkness_wrapped(image: np.ndarray, severity: float) -> np.ndarray:
    transform = make_darkness_transform(severity)
    return transform(image=image)["image"]


def apply_fog_wrapped(image: np.ndarray, severity: float) -> np.ndarray:
    return apply_fog(image, severity)


def apply_snow_wrapped(image: np.ndarray, severity: float) -> np.ndarray:
    return apply_snow(image, severity, SNOW_SEED)


def apply_blur_wrapped(image: np.ndarray, severity: float) -> np.ndarray:
    return apply_motion_blur(image, severity)


def apply_compression_wrapped(image: np.ndarray, severity: float) -> np.ndarray:
    return apply_compression_degradation(image, severity)


APPLY_FUNCTIONS = {
    "mrak": apply_darkness_wrapped,
    "magla": apply_fog_wrapped,
    "sneg": apply_snow_wrapped,
    "blur": apply_blur_wrapped,
    "komp": apply_compression_wrapped,
}


# ---------------------- Ucitavanje slika ----------------------

def load_rgb(path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Ne mogu da ucitam sliku: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def select_sample_paths(dir_str: str):
    folder = Path(dir_str)
    if not folder.is_dir():
        print(f"GRESKA: folder {folder.resolve()} ne postoji.")
        sys.exit(1)

    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not paths:
        print(f"GRESKA: nema slika u {folder.resolve()}")
        sys.exit(1)

    total = len(paths)
    if total > MAX_SAMPLE_IMAGES:
        paths = sorted(random.Random(SAMPLE_SEED).sample(paths, MAX_SAMPLE_IMAGES))

    # Odbacujemo slike koje se ne mogu ucitati (da ne pukne neki proces-radnik)
    readable = []
    for p in paths:
        if cv2.imread(str(p)) is None:
            print(f"Upozorenje: ne mogu da ucitam {p.name}, izbacujem iz uzorka.")
        else:
            readable.append(p)

    if not readable:
        print("GRESKA: nijedna slika iz uzorka se ne moze ucitati.")
        sys.exit(1)

    print(f"Uzorak za kalibraciju: {len(readable)} od {total} slika iz {folder}")
    return readable


# ---------------------- Procesi-radnici ----------------------

_WORKER_IMAGES = []


def _init_worker(image_paths):
    """Poziva se jednom pri pokretanju svakog procesa: ucita ceo uzorak u memoriju."""
    global _WORKER_IMAGES
    cv2.setNumThreads(1)  # OpenCV ne sme da pravi sopstvene niti povrh nasih procesa
    _WORKER_IMAGES = [load_rgb(p) for p in image_paths]


def _eval_point(task):
    """Za jednu (tip, severity) kombinaciju vraca SSIM preko svih slika uzorka:
    (prosek, minimum, maksimum)."""
    type_key, severity = task
    apply_fn = APPLY_FUNCTIONS[type_key]
    scores = []
    for original in _WORKER_IMAGES:
        degraded = apply_fn(original, severity)
        scores.append(ssim(original, degraded, channel_axis=-1))
    return float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))


# ---------------------- Kalibracija ----------------------

def find_duplicate_levels(apply_fn, calibrated: dict, image: np.ndarray):
    """Vraca listu parova (nivo_a, nivo_b) koji daju IDENTICNE slike.
    Proverava se sa zaokruzenim severity vrednostima, jer se upravo one upisuju
    u degrade_*.py skripte. Susedni nivoi su sortirani od blagog ka jakom."""
    duplicates = []
    previous_image, previous_target = None, None
    for target in sorted(calibrated.keys(), reverse=True):
        current = apply_fn(image, calibrated[target])
        if previous_image is not None and np.array_equal(current, previous_image):
            duplicates.append((previous_target, target))
        previous_image, previous_target = current, target
    return duplicates


def calibrate_one_type(type_key: str, map_fn, first_image: np.ndarray):
    print(f"\n{'=' * 60}")
    print(f"KALIBRACIJA: {DISPLAY_NAMES[type_key]}")
    print(f"{'=' * 60}")

    grid = SEVERITY_GRIDS[type_key]
    tasks = [(type_key, float(s)) for s in grid]

    results = []
    for i, result in enumerate(map_fn(tasks), start=1):
        results.append(result)
        if i % 25 == 0 or i == len(tasks):
            print(f"  izmereno {i}/{len(tasks)} jacina...")

    severities = np.array([t[1] for t in tasks])
    means = np.array([r[0] for r in results])
    mins = np.array([r[1] for r in results])
    maxs = np.array([r[2] for r in results])

    # np.interp trazi rastuci niz x vrednosti, pa sortiramo po prosecnom SSIM-u
    order = np.argsort(means)
    means, severities, mins, maxs = means[order], severities[order], mins[order], maxs[order]

    print(f"\nDostizni prosecni SSIM: {means.min():.3f} do {means.max():.3f}\n")

    calibrated = {}
    spread = {}
    skipped = []
    for target in TARGET_SSIM_LEVELS:
        if target < means.min() or target > means.max():
            skipped.append(target)
            continue
        severity = float(np.interp(target, means, severities))
        low = float(np.interp(target, means, mins))
        high = float(np.interp(target, means, maxs))
        calibrated[target] = round(severity, 3)
        spread[target] = (low, high)
        print(f"  SSIM={target:.3f}  ->  severity={severity:8.3f}   "
              f"(po pojedinacnim slikama: {low:.2f} - {high:.2f})")

    if skipped:
        print(f"\n  Nedostizno za ovaj tip: {len(skipped)} nivoa "
              f"(ispod {means.min():.3f} ili iznad {means.max():.3f}), preskoceno.")

    duplicates = find_duplicate_levels(APPLY_FUNCTIONS[type_key], calibrated, first_image)
    if duplicates:
        print("\n  UPOZORENJE: sledeci susedni nivoi daju IDENTICNE slike (isti efekat):")
        for a, b in duplicates:
            print(f"    SSIM={a:.3f} i SSIM={b:.3f}")
        print("  Takvi nivoi se u evaluaciji ne razlikuju, a nose razlicite oznake.")

    prefix = type_key
    print(f"\n{DICT_NAMES[type_key]} = {{")
    for target, severity in calibrated.items():
        print(f'    "{prefix}_ssim_{int(target * 100)}": {severity},')
    print("}")

    return calibrated


# ---------------------- Glavni deo ----------------------

def main():
    start = time.time()
    sample_paths = select_sample_paths(SAMPLE_IMAGES_DIR)
    first_image = load_rgb(sample_paths[0])
    print(f"Prva slika uzorka: {first_image.shape[1]}x{first_image.shape[0]} piksela")

    n_workers = N_WORKERS or max(1, min((os.cpu_count() or 2) - 2, 16))
    print(f"Broj procesa: {n_workers}")

    executor = None
    if n_workers > 1:
        # "spawn" radi isto na svim sistemima (na Windows-u je jedini nacin)
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=([str(p) for p in sample_paths],),
        )
        map_fn = lambda tasks: executor.map(_eval_point, tasks, chunksize=2)
    else:
        _init_worker([str(p) for p in sample_paths])
        map_fn = lambda tasks: map(_eval_point, tasks)

    try:
        for type_key in ["mrak", "magla", "sneg", "blur", "komp"]:
            calibrate_one_type(type_key, map_fn, first_image)
    finally:
        if executor is not None:
            executor.shutdown()

    elapsed = time.time() - start
    print(f"\n\n{'#' * 60}")
    print(f"# SVE KALIBRACIJE ZAVRSENE ({elapsed / 60:.1f} min)")
    print(f"{'#' * 60}")
    print("Kopiraj gornje recnike u odgovarajuce degrade_*.py skripte.")
    print(f"Brojevi vaze za uzorak od {len(sample_paths)} slika iz {SAMPLE_IMAGES_DIR}.")
    print("Ako se test skup promeni, kalibraciju treba ponoviti.")


if __name__ == "__main__":
    main()