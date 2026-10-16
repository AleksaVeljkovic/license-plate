"""
Cilj: Napraviti "ciscen" test skup (dataset/test_clean) u kome:
  1) nema slika ciji se automobil (tablica) pojavljuje i u train skupu
     (curenje podataka / data leakage), i
  2) su uklonjeni pogresni boxovi (oznake koje nisu tablice) iz anotacija
     slika koje ostaju.

Isti filter se primenjuje i na sve degradirane foldere:
  degraded/<tip>/<nivo>/{images,labels}  ->  degraded_clean/<tip>/<nivo>/{images,labels}

Originalni dataset/test i degraded/ se NE DIRAJU. Slike se kopiraju (ili
hard-linkuju da se ne trosi prostor), labele se pisu nove.

Ulaz:  ocr_dataset_check.csv (izlaz ocr_check_dataset.py) u glavnom folderu.
Izlaz: dataset/test_clean/, degraded_clean/, test_clean_izvestaj.txt

    python make_clean_test.py            # samo izvestaj, nista ne pise
    python make_clean_test.py --napravi  # stvarno pravi foldere
"""

import csv
import os
import shutil
import sys
from pathlib import Path


OCR_CSV = "ocr_dataset_check.csv"
DATASET_DIR = Path("dataset")
DEGRADED_ROOT = Path("degraded")
OUT_TEST = DATASET_DIR / "test_clean"
OUT_DEGRADED = Path("degraded_clean")
REPORT_FILE = "test_clean_izvestaj.txt"

# Boxovi koji sigurno NISU tablice (split, slika, idx u labeli) - uklanjaju se.
BAD_BOXES = {
    ("train", "IMG0010.jpg", 1), ("train", "IMG0019.jpg", 1), ("train", "IMG0024.jpg", 1),
    ("train", "IMG0199.jpg", 1), ("train", "IMG0373.jpg", 1), ("train", "IMG0427.jpg", 1),
    ("train", "IMG0575.jpg", 1), ("train", "IMG0636.jpg", 1), ("train", "IMG0636.jpg", 2),
    ("val", "IMG0533.jpg", 1),
    ("test", "IMG0031.jpg", 2), ("test", "IMG0312.jpg", 2), ("test", "IMG0365.jpg", 1),
    ("test", "IMG0631.jpg", 1), ("test", "IMG0631.jpg", 2),
}

# Nejasni boxovi - po defaultu ostaju. Ako odlucis da ih ukloniš, dodaj ih
# u BAD_BOXES_TEST_EXTRA, npr. ("IMG0049.jpg", 1), ("IMG0312.jpg", 1).
BAD_BOXES_TEST_EXTRA = set()

# Slike koje rucno dodatno izbacujes iz test_clean (npr. par duplikata unutar
# testa: IMG0662.jpg, IMG0593.jpg). Po defaultu prazno.
EXTRA_EXCLUDE = set()

MIN_TEXT_LEN = 5  # kraci tekst se ne koristi za poredjenje


def lev1(a, b):
    """True ako se stringovi razlikuju za najvise jedan znak."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def matches(text, pool):
    return any(p == text or (len(text) >= 6 and lev1(text, p)) for p in pool)


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def filtered_label_lines(label_path: Path, bad_idx: set):
    lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [l for i, l in enumerate(lines) if i not in bad_idx]


def main():
    write = "--napravi" in sys.argv

    if not Path(OCR_CSV).exists():
        print(f"GRESKA: {Path(OCR_CSV).resolve()} ne postoji.")
        return

    with open(OCR_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["tablica_idx"] = int(r["tablica_idx"])

    bad_all = set(BAD_BOXES) | {("test", s, i) for s, i in BAD_BOXES_TEST_EXTRA}

    def is_bad(r):
        return (r["split"], r["slika"], r["tablica_idx"]) in bad_all

    usable = [r for r in rows if not is_bad(r) and len(r["procitan_tekst"]) >= MIN_TEXT_LEN]
    train_texts = [r["procitan_tekst"] for r in usable if r["split"] == "train"]

    test_images = sorted({r["slika"] for r in rows if r["split"] == "test"})
    excluded = set()
    for r in usable:
        if r["split"] == "test" and matches(r["procitan_tekst"], train_texts):
            excluded.add(r["slika"])
    excluded |= set(EXTRA_EXCLUDE)
    kept = [s for s in test_images if s not in excluded]

    # boxovi za brisanje po slici (samo za slike koje ostaju)
    bad_by_image = {}
    for r in rows:
        if r["split"] == "test" and is_bad(r) and r["slika"] in kept:
            bad_by_image.setdefault(r["slika"], set()).add(r["tablica_idx"])

    n_plates = sum(1 for r in rows if r["split"] == "test" and r["slika"] in kept and not is_bad(r))

    report = []
    report.append(f"Test slika ukupno: {len(test_images)}")
    report.append(f"Izbaceno (vozilo i u train): {len(excluded)}")
    report.append(f"test_clean: {len(kept)} slika, {n_plates} tablica")
    report.append(f"Uklonjeni pogresni boxovi na slikama koje ostaju: {sum(len(v) for v in bad_by_image.values())}")
    for s, idxs in sorted(bad_by_image.items()):
        report.append(f"   {s}: idx {sorted(idxs)}")
    report.append("Izbacene slike: " + ", ".join(sorted(excluded)))
    print("\n".join(report))

    if not write:
        print("\n(Probni rad - nista nije upisano. Pokreni sa --napravi.)")
        return

    # ---- dataset/test_clean ----
    src_img = DATASET_DIR / "test" / "images"
    src_lbl = DATASET_DIR / "test" / "labels"
    for s in kept:
        link_or_copy(src_img / s, OUT_TEST / "images" / s)
        lbl = src_lbl / (Path(s).stem + ".txt")
        out = OUT_TEST / "labels" / lbl.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(filtered_label_lines(lbl, bad_by_image.get(s, set()))) + "\n",
                       encoding="utf-8")
    print(f"\nNapravljeno: {OUT_TEST.resolve()}")

    # ---- degraded_clean ----
    kept_stems = {Path(s).stem: s for s in kept}
    missing = 0
    n_folders = 0
    if DEGRADED_ROOT.exists():
        for img_dir in sorted(DEGRADED_ROOT.glob("*/*/images")):
            level_dir = img_dir.parent
            lbl_dir = level_dir / "labels"
            out_level = OUT_DEGRADED / level_dir.relative_to(DEGRADED_ROOT)
            n_folders += 1
            for stem, name in kept_stems.items():
                found = [p for p in img_dir.glob(stem + ".*")]
                if not found:
                    missing += 1
                    continue
                link_or_copy(found[0], out_level / "images" / found[0].name)
                lbl = lbl_dir / (stem + ".txt")
                if lbl.exists():
                    out = out_level / "labels" / lbl.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(
                        "\n".join(filtered_label_lines(lbl, bad_by_image.get(name, set()))) + "\n",
                        encoding="utf-8")
        print(f"degraded_clean: {n_folders} foldera, nedostajucih slika: {missing}")
    else:
        print(f"Upozorenje: {DEGRADED_ROOT} ne postoji, degradirani folderi nisu napravljeni.")

    Path(REPORT_FILE).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Izvestaj: {Path(REPORT_FILE).resolve()}")


if __name__ == "__main__":
    main()