"""
Cilj: Pripremiti "ground truth" (tacan tekst) za OCR na ocisenom test skupu
(dataset/test_clean), sa sto manje rucnog posla.

Sta radi:
1. Cita ocr_test_clean.csv (nastao nad starim test skupom) i zadrzava samo
   tablice sa slika koje postoje u dataset/test_clean (bez izbacenih slika i
   bez pogresnih boxova koji su uklonjeni iz labela).
2. Oznacava koje tablice TREBA RUCNO PROVERITI:
     - tekst nema oblik srpske tablice (2 slova + 3-4 cifre + 2 slova), ili
     - pouzdanost OCR-a < LOW_CONFIDENCE.
   Ostale (sigurne i pravilnog oblika) skoro sigurno su tacne, ali se za
   kontrolu nasumicno dodaje i nekoliko njih (CONTROL_SAMPLE) u pregled.
3. Pravi kolaze isecaka za proveru (ocr_gt_pregled/sheet_XX.jpg) sa natpisom
   "slika #idx: procitan tekst (pouzdanost)", i fajl ocr_gt.csv.

U ocr_gt.csv kolona 'tacan_tekst' je na pocetku kopija procitanog teksta.
Ispravljanje redova. Ako box NIJE tablica
(natpis, deo vozila...), upise se X.

"""

import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np


INPUT_CSV = "ocr_test_clean.csv"
CROPS_DIR = Path("ocr_crops_test")
CLEAN_IMAGES = Path("dataset/test_clean/images")
CLEAN_LABELS = Path("dataset/test_clean/labels")

OUTPUT_CSV = "ocr_gt.csv"
SHEETS_DIR = Path("ocr_gt_pregled")

LOW_CONFIDENCE = 0.95
CONTROL_SAMPLE = 1000  # koliko "sigurnih" tablica dodati u pregled (1000 = sve, da proveris svih 102)
SEED = 42

SERBIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{3,4}[A-Z]{2}$")

SHEET_COLS = 3
SHEET_ROWS = 6
TILE_W = 520
TILE_H = 170
CAPTION_H = 30


def make_tile(crop, caption):
    tile = np.full((TILE_H + CAPTION_H, TILE_W, 3), 255, dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = min(TILE_W / w, TILE_H / h)
    resized = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    rh, rw = resized.shape[:2]
    tile[CAPTION_H:CAPTION_H + rh, 0:rw] = resized
    cv2.putText(tile, caption, (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H + CAPTION_H - 1), (160, 160, 160), 1)
    return tile


def save_sheets(tiles):
    SHEETS_DIR.mkdir(exist_ok=True)
    per = SHEET_COLS * SHEET_ROWS
    blank = np.full((TILE_H + CAPTION_H, TILE_W, 3), 255, dtype=np.uint8)
    for s, start in enumerate(range(0, len(tiles), per), 1):
        chunk = tiles[start:start + per]
        chunk = chunk + [blank] * (per - len(chunk))
        rows = [np.hstack(chunk[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
        cv2.imwrite(str(SHEETS_DIR / f"sheet_{s:02d}.jpg"), np.vstack(rows))


def main():
    if not Path(INPUT_CSV).exists():
        print(f"GRESKA: {Path(INPUT_CSV).resolve()} ne postoji.")
        return
    if not CLEAN_IMAGES.exists():
        print(f"GRESKA: {CLEAN_IMAGES.resolve()} ne postoji (pokreni make_clean_test.py --napravi).")
        return

    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    kept = []
    for r in rows:
        img = CLEAN_IMAGES / r["slika"]
        lbl = CLEAN_LABELS / (Path(r["slika"]).stem + ".txt")
        if not img.exists() or not lbl.exists():
            continue
        n_boxes = len([l for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()])
        if int(r["tablica_idx"]) >= n_boxes:
            continue  # box je uklonjen iz labele (pogresan box)
        r["pouzdanost"] = float(r["pouzdanost"])
        kept.append(r)

    def needs_check(r):
        return (not SERBIAN_PLATE_REGEX.match(r["procitan_tekst"])) or r["pouzdanost"] < LOW_CONFIDENCE

    flagged = [r for r in kept if needs_check(r)]
    safe = [r for r in kept if not needs_check(r)]
    random.Random(SEED).shuffle(safe)
    control = safe[:CONTROL_SAMPLE]

    for r in kept:
        r["provera"] = "da" if r in flagged else ("kontrola" if r in control else "ne")
        r["tacan_tekst"] = r["procitan_tekst"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["slika", "tablica_idx", "procitan_tekst",
                                          "pouzdanost", "tacan_tekst", "provera"])
        w.writeheader()
        for r in kept:
            w.writerow({k: r[k] for k in w.fieldnames})

    tiles = []
    for r in flagged + control:
        crop_path = CROPS_DIR / f"{Path(r['slika']).stem}_{r['tablica_idx']}.jpg"
        crop = cv2.imread(str(crop_path))
        if crop is None:
            print(f"  nema isecka: {crop_path}")
            continue
        tag = "PROVERI" if r in flagged else "kontrola"
        cap = f"{r['slika']} #{r['tablica_idx']}: {r['procitan_tekst'] or '-'} ({r['pouzdanost']:.2f}) [{tag}]"
        tiles.append(make_tile(crop, cap))
    if tiles:
        save_sheets(tiles)

    print(f"Tablica u test_clean: {len(kept)} (slika: {len({r['slika'] for r in kept})})")
    print(f"Za proveru: {len(flagged)} + kontrola: {len(control)}")
    print(f"CSV: {Path(OUTPUT_CSV).resolve()}")
    print(f"Kolazi za pregled: {SHEETS_DIR.resolve()}")
    print("\nSledece: pogledaj kolaze, u ocr_gt.csv ispravi 'tacan_tekst' gde OCR gresi,"
          "\na za box koji nije tablica upisi X.")


if __name__ == "__main__":
    main()