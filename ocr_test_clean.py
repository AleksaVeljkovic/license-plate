"""

Cilj: Proci OCR-om kroz SVE oznacene tablice u celom datasetu (train + val +
test) i pronaci sumnjive boxove - one koji verovatno NISU tablice (natpis na
vozilu, pogresno oznacen deo slike, prazan isecak) ili su vrlo nepouzdani.

Tako vidimo koliko pogresnih anotacija zaista ima (u celom skupu, ne samo u
testu) i da li se isplati ispravljati train skup i ponovo trenirati modele.

Sumnjiv box = bar jedno od:
  - OCR tekst je prazan
  - tekst ne odgovara obliku srpske tablice (2 slova + 3-4 cifre + 2 slova)
  - prosecna pouzdanost OCR-a < LOW_CONFIDENCE

NAPOMENA: ovo hvata samo POGRESNE boxove (box bez prave tablice). Ne hvata
tablice koje uopste nisu oznacene.

Izlaz:
  ocr_dataset_check.csv       - svi boxovi (split, slika, idx, tekst, ...)
  ocr_flagged/sheet_XX.jpg    - kolaži isečaka SAMO sumnjivih boxova, sa
                                natpisom "split/slika #idx: procitan tekst"

"""

import csv
import re
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer


DATASET_DIR = Path("dataset")
SPLITS = ["train", "val", "test"]

OUTPUT_CSV = "ocr_dataset_check.csv"
FLAGGED_DIR = Path("ocr_flagged")
TMP_CROP = Path("ocr_flagged/_tmp_crop.jpg")  # privremeni isecak, brise se na kraju

PADDING_RATIO = 0.02
OCR_MODEL_NAME = "cct-s-v2-global-model"

# Ispod ove pouzdanosti box se smatra sumnjivim
LOW_CONFIDENCE = 0.85

# Oblik srpske tablice: 2 slova (grad) + 3-4 cifre + 2 slova, npr. BG1234AB,
# SD085SG, BG740AB. (Posebne tablice - vojne, diplomatske... mogu biti
# oznacene kao "sumnjive" iako su ispravne - zato je potreban pregled.)
SERBIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{3,4}[A-Z]{2}$")

# Kolaz za pregled
SHEET_COLS = 4
SHEET_ROWS = 8
TILE_W = 360
TILE_H = 110
CAPTION_H = 26



def yolo_box_to_pixels(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    _, xc, yc, bw, bh = map(float, parts)
    return (
        int((xc - bw / 2) * w),
        int((yc - bh / 2) * h),
        int((xc + bw / 2) * w),
        int((yc + bh / 2) * h),
    )


def crop_with_padding(image, box, ratio):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    px, py = int((x2 - x1) * ratio), int((y2 - y1) * ratio)
    return image[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]


def read_plate(recognizer, crop):
    """OCR na isecku. Isecak se privremeno snima na disk i cita po putanji,
    isto kao u ocr_test_clean.py (provereni nacin, isti rezultati).
    Vraca (tekst, pouzdanost)."""
    TMP_CROP.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(TMP_CROP), crop)
    predictions = recognizer.run(str(TMP_CROP), return_confidence=True)
    if not predictions:
        return "", 0.0
    pred = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
    if hasattr(pred, "plate"):
        probs = getattr(pred, "char_probs", None)
        conf = float(np.mean(probs)) if probs is not None and len(probs) > 0 else 0.0
        return pred.plate, conf
    return str(pred), 0.0


def make_tile(crop, caption):
    """Jedna plocica kolaza: isecak (smanjen) + natpis."""
    tile = np.full((TILE_H + CAPTION_H, TILE_W, 3), 255, dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = min(TILE_W / w, TILE_H / h)
    resized = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    rh, rw = resized.shape[:2]
    tile[CAPTION_H:CAPTION_H + rh, 0:rw] = resized
    cv2.putText(tile, caption, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H + CAPTION_H - 1), (160, 160, 160), 1)
    return tile


def save_sheets(tiles):
    FLAGGED_DIR.mkdir(exist_ok=True)
    per_sheet = SHEET_COLS * SHEET_ROWS
    blank = np.full((TILE_H + CAPTION_H, TILE_W, 3), 255, dtype=np.uint8)
    for s, start in enumerate(range(0, len(tiles), per_sheet), 1):
        chunk = tiles[start:start + per_sheet]
        chunk += [blank] * (per_sheet - len(chunk))
        rows = [np.hstack(chunk[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
        cv2.imwrite(str(FLAGGED_DIR / f"sheet_{s:02d}.jpg"), np.vstack(rows))


def main():
    print(f"Učitavam fast-plate-ocr model ({OCR_MODEL_NAME})...")
    recognizer = LicensePlateRecognizer(OCR_MODEL_NAME)
    print("Model spreman.\n")

    rows = []
    tiles = []

    for split in SPLITS:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"
        if not images_dir.exists():
            print(f"Upozorenje: {images_dir} ne postoji, preskačem.")
            continue

        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        print(f"[{split}] {len(image_paths)} slika...")

        for img_path in image_paths:
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]

            for idx, line in enumerate(lines):
                box = yolo_box_to_pixels(line, w, h)
                if box is None:
                    continue
                crop = crop_with_padding(image, box, PADDING_RATIO)
                if crop.size == 0:
                    continue

                text, conf = read_plate(recognizer, crop)

                reasons = []
                if not text.strip():
                    reasons.append("prazan")
                elif not SERBIAN_PLATE_REGEX.match(text):
                    reasons.append("oblik")
                if conf < LOW_CONFIDENCE:
                    reasons.append("pouzdanost")

                rows.append({
                    "split": split,
                    "slika": img_path.name,
                    "tablica_idx": idx,
                    "procitan_tekst": text,
                    "pouzdanost": round(conf, 3),
                    "sumnjiv": "da" if reasons else "ne",
                    "razlog": "+".join(reasons),
                })

                if reasons:
                    caption = f"{split}/{img_path.name} #{idx}: {text or '-'} ({conf:.2f})"
                    tiles.append(make_tile(crop, caption))

    TMP_CROP.unlink(missing_ok=True)

    if not rows:
        print("Nema podataka.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if tiles:
        save_sheets(tiles)

    # ---- Sazetak ----
    print("\n" + "=" * 50)
    print(f"{'podela':<8}{'boxova':>8}{'sumnjivih':>12}{'udeo':>8}")
    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        bad = sum(1 for r in sub if r["sumnjiv"] == "da")
        print(f"{split:<8}{len(sub):>8}{bad:>12}{bad / len(sub):>8.1%}")
    total_bad = sum(1 for r in rows if r["sumnjiv"] == "da")
    print(f"{'UKUPNO':<8}{len(rows):>8}{total_bad:>12}{total_bad / len(rows):>8.1%}")
    print("=" * 50)
    print(f"\nCSV: {Path(OUTPUT_CSV).resolve()}")
    if tiles:
        n_sheets = (len(tiles) + SHEET_COLS * SHEET_ROWS - 1) // (SHEET_COLS * SHEET_ROWS)
        print(f"Kolaži sumnjivih ({len(tiles)} isečaka, {n_sheets} slika): {FLAGGED_DIR.resolve()}")


if __name__ == "__main__":
    main()