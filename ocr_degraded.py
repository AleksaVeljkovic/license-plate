"""
Cilj: Izmeriti koliko tacno OCR cita tablice na DEGRADIRANIM slikama i uporediti
sa tackom na kojoj pada detekcija. Tacan tekst je iz rucno proverenog ocr_gt.csv
(redovi sa X se preskacu).

Za svaki uslov (cisto + 131 degradirani folder iz degraded_clean/) i svaku tablicu
iz ocr_gt.csv radi DVA merenja:

  1) "gt_box"    - tablica se isece prema TACNOM boxu iz labele pa se procita.
                   Meri samo koliko OCR trpi od degradacije (bez uticaja detektora).
  2) "detektor"  - YOLO (nano) prvo pronadje tablicu na degradiranoj slici (conf 0.25),
                   pa se procita detektovani isecak. Ako detektor tablicu ne nadje
                   (IoU < 0.5), pokusaj se racuna kao pogresan. To je ceo sistem
                   (detekcija + OCR) kakav bi radio u praksi.

Metrike po uslovu:
  tacnost_cela  - udeo tablica ciji se tekst POTPUNO poklapa sa tacnim
  tacnost_znak  - 1 - (Levenshtein razdaljina / duzina tacnog teksta), prosek
  pouzdanost    - prosecna pouzdanost OCR-a

Izlaz:
  ocr_results.csv          - po uslovu (tip, nivo, ssim, nacin, metrike)
  ocr_results_detail.csv   - po tablici i uslovu
  ocr_prag_pada.csv        - SSIM na kom tacnost_cela prvi put padne ispod 0.5
  ocr_plots/<tip>.png      - detekcija (mAP50) vs OCR vs ceo sistem u odnosu na SSIM

"""

import csv
import re
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO


GT_CSV = "ocr_gt.csv"
CLEAN_ROOT = Path("dataset/test_clean")
DEGRADED_ROOT = Path("degraded_clean")

DETECTOR_SIZE = "n"          # koji YOLO model se koristi za "detektor" nacin
DETECTOR_WEIGHTS = [
    "runs/train/tablice_{size}/weights/best.pt",
    "runs/detect/runs/train/tablice_{size}/weights/best.pt",
]
DET_CONF = 0.25
IOU_MATCH = 0.5

OCR_MODEL_NAME = "cct-s-v2-global-model"
PADDING_RATIO = 0.02         # isto kao u ocr_test_clean.py
PAGE_THRESHOLD = 0.5         # "prag pada": tacnost_cela < 0.5

DETECTION_CSV = "evaluation_results_clean.csv"  # za poredjenje (mAP50 nano modela)
OUT_RESULTS = "ocr_results.csv"
OUT_DETAIL = "ocr_results_detail.csv"
OUT_BREAK = "ocr_prag_pada.csv"
PLOTS_DIR = Path("ocr_plots")
TMP_CROP = Path("_ocr_tmp_crop.jpg")



def norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(truth: str, pred: str) -> float:
    if not truth:
        return 0.0
    return max(0.0, 1.0 - levenshtein(truth, pred) / len(truth))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_boxes(label_path: Path, w: int, h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        _, xc, yc, bw, bh = map(float, p)
        boxes.append((int((xc - bw / 2) * w), int((yc - bh / 2) * h),
                      int((xc + bw / 2) * w), int((yc + bh / 2) * h)))
    return boxes


def crop_with_padding(image, box):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    px, py = int((x2 - x1) * PADDING_RATIO), int((y2 - y1) * PADDING_RATIO)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return image[y1:y2, x1:x2]


def read_plate(recognizer, crop):
    """Vraca (tekst, pouzdanost). Isecak se cuva kao jpg (isto kao pri pravljenju
    tacnog teksta), da bi uslovi citanja bili isti."""
    if crop is None or crop.size == 0:
        return "", 0.0
    cv2.imwrite(str(TMP_CROP), crop)
    preds = recognizer.run(str(TMP_CROP), return_confidence=True)
    if not preds:
        return "", 0.0
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    if hasattr(pred, "plate"):
        cp = getattr(pred, "char_probs", None)
        conf = float(np.mean(cp)) if cp is not None and len(cp) > 0 else 0.0
        return pred.plate, conf
    return str(pred), 0.0


def find_weights(size):
    for t in DETECTOR_WEIGHTS:
        p = Path(t.format(size=size))
        if p.exists():
            return p
    return None


def load_gt():
    by_image = {}
    with open(GT_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = norm(r["tacan_tekst"])
            if not t or t == "X":
                continue
            by_image.setdefault(r["slika"], []).append((int(r["tablica_idx"]), t))
    return by_image


def discover_conditions():
    conds = [("cisto", "baseline", 1.0, CLEAN_ROOT / "images", CLEAN_ROOT / "labels")]
    if DEGRADED_ROOT.exists():
        for tip_dir in sorted(DEGRADED_ROOT.iterdir()):
            if not tip_dir.is_dir():
                continue
            for lvl in sorted(tip_dir.iterdir()):
                if (lvl / "images").exists():
                    m = re.search(r"ssim_(\d+)", lvl.name)
                    ssim = int(m.group(1)) / 100.0 if m else float("nan")
                    conds.append((tip_dir.name, lvl.name, ssim, lvl / "images", lvl / "labels"))
    return conds


def find_image(images_dir: Path, name: str):
    p = images_dir / name
    if p.exists():
        return p
    found = list(images_dir.glob(Path(name).stem + ".*"))
    return found[0] if found else None


def evaluate_condition(tip, nivo, ssim, images_dir, labels_dir, gt, recognizer, detector):
    detail = []
    for name, plates in sorted(gt.items()):
        img_path = find_image(images_dir, name)
        if img_path is None:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes = read_boxes(labels_dir / (Path(name).stem + ".txt"), w, h)

        preds = []
        if detector is not None:
            res = detector.predict(str(img_path), conf=DET_CONF, verbose=False)[0]
            preds = [tuple(int(v) for v in b.xyxy[0].tolist()) for b in res.boxes]

        for idx, truth in plates:
            if idx >= len(boxes):
                continue
            gbox = boxes[idx]

            # 1) isecak prema tacnom boxu
            text, conf = read_plate(recognizer, crop_with_padding(image, gbox))
            detail.append(row(tip, nivo, ssim, "gt_box", name, idx, truth, text, conf, True))

            # 2) isecak prema detekciji
            if detector is not None:
                best, bbox = 0.0, None
                for pb in preds:
                    v = iou(gbox, pb)
                    if v > best:
                        best, bbox = v, pb
                if bbox is not None and best >= IOU_MATCH:
                    text, conf = read_plate(recognizer, crop_with_padding(image, bbox))
                    detail.append(row(tip, nivo, ssim, "detektor", name, idx, truth, text, conf, True))
                else:
                    detail.append(row(tip, nivo, ssim, "detektor", name, idx, truth, "", 0.0, False))
    return detail


def row(tip, nivo, ssim, mode, name, idx, truth, text, conf, detected):
    pred = norm(text)
    return {
        "tip_degradacije": tip, "nivo": nivo, "ssim": ssim, "nacin": mode,
        "slika": name, "tablica_idx": idx, "tacan_tekst": truth,
        "procitan_tekst": pred, "pouzdanost": round(conf, 3),
        "pronadjena": int(detected), "tacno": int(pred == truth),
        "tacnost_znak": round(char_accuracy(truth, pred), 4),
    }


def aggregate(detail):
    groups = {}
    for r in detail:
        groups.setdefault((r["tip_degradacije"], r["nivo"], r["ssim"], r["nacin"]), []).append(r)
    out = []
    for (tip, nivo, ssim, mode), rs in groups.items():
        n = len(rs)
        out.append({
            "tip_degradacije": tip, "nivo": nivo, "ssim": ssim, "nacin": mode, "n_tablica": n,
            "tacnost_cela": round(sum(r["tacno"] for r in rs) / n, 4),
            "tacnost_znak": round(sum(r["tacnost_znak"] for r in rs) / n, 4),
            "pouzdanost": round(sum(r["pouzdanost"] for r in rs) / n, 4),
        })
    out.sort(key=lambda r: (r["tip_degradacije"], r["nacin"], -r["ssim"] if r["ssim"] == r["ssim"] else 0))
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_detection_nano():
    """mAP50 nano modela po (tip, ssim) iz evaluation_results_clean.csv."""
    p = Path(DETECTION_CSV)
    if not p.exists():
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["model"] != f"yolo11{DETECTOR_SIZE}":
                continue
            m = re.search(r"ssim_(\d+)", r["nivo"])
            ssim = int(m.group(1)) / 100.0 if m else 1.0
            out.setdefault(r["tip_degradacije"], {})[ssim] = float(r["mAP50"])
    base = out.get("cisto", {}).get(1.0)
    return out, base


def breakpoints(agg, det):
    """Prvi (najvisi) SSIM nivo gde tacnost_cela < PAGE_THRESHOLD, po tipu i nacinu."""
    rows = []
    tipovi = sorted({r["tip_degradacije"] for r in agg if r["tip_degradacije"] != "cisto"})
    for tip in tipovi:
        row_out = {"tip_degradacije": tip}
        for mode in ("gt_box", "detektor"):
            sub = sorted((r for r in agg if r["tip_degradacije"] == tip and r["nacin"] == mode),
                         key=lambda r: -r["ssim"])
            hit = next((r for r in sub if r["tacnost_cela"] < PAGE_THRESHOLD), None)
            row_out[f"ocr_{mode}"] = f"{hit['ssim']:.2f}" if hit else \
                (f"nije pao (min SSIM {min(r['ssim'] for r in sub):.2f})" if sub else "")
        d = det.get(tip, {}) if det else {}
        hit = next(((s, v) for s, v in sorted(d.items(), reverse=True) if v < 0.5), None)
        row_out["detekcija_mAP50<0.5"] = f"{hit[0]:.2f}" if hit else "nije pao"
        rows.append(row_out)
    return rows


def make_plots(agg, det):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nije instaliran, grafikoni se preskacu.")
        return
    PLOTS_DIR.mkdir(exist_ok=True)
    base = {m: next((r for r in agg if r["tip_degradacije"] == "cisto" and r["nacin"] == m), None)
            for m in ("gt_box", "detektor")}
    tipovi = sorted({r["tip_degradacije"] for r in agg if r["tip_degradacije"] != "cisto"})
    for tip in tipovi:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for mode, label in (("gt_box", "OCR (isecak po tacnom boxu)"),
                            ("detektor", "Ceo sistem (detekcija + OCR)")):
            sub = sorted((r for r in agg if r["tip_degradacije"] == tip and r["nacin"] == mode),
                         key=lambda r: -r["ssim"])
            xs = [1.0] + [r["ssim"] for r in sub]
            ys = ([base[mode]["tacnost_cela"]] if base[mode] else [np.nan]) + [r["tacnost_cela"] for r in sub]
            ax.plot(xs, ys, marker="o", ms=3, label=label)
        d = det.get(tip, {}) if det else {}
        if d:
            xs = [1.0] + sorted(d.keys(), reverse=True)
            b = det.get("cisto", {}).get(1.0)
            ys = [b if b is not None else np.nan] + [d[s] for s in sorted(d.keys(), reverse=True)]
            ax.plot(xs, ys, marker="s", ms=3, ls="--", label=f"Detekcija mAP50 (yolo11{DETECTOR_SIZE})")
        ax.axhline(PAGE_THRESHOLD, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("SSIM (1 = bez degradacije)")
        ax.set_ylabel("Tacnost / mAP50")
        ax.set_title(f"{tip}: detekcija vs. citanje tablice")
        ax.set_ylim(-0.02, 1.02)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{tip}.png", dpi=150)
        plt.close(fig)


def main():
    if not Path(GT_CSV).exists():
        print(f"GRESKA: {Path(GT_CSV).resolve()} ne postoji.")
        return
    gt = load_gt()
    n_plates = sum(len(v) for v in gt.values())
    print(f"Tacan tekst: {n_plates} tablica na {len(gt)} slika (X redovi preskoceni).")

    conds = discover_conditions()
    print(f"Uslova: {len(conds)} (cisto + degradirani)\n")

    weights = find_weights(DETECTOR_SIZE)
    detector = YOLO(str(weights)) if weights else None
    if detector is None:
        print(f"Upozorenje: model yolo11{DETECTOR_SIZE} nije pronadjen, radi samo 'gt_box' nacin.")

    print(f"Ucitavam OCR model ({OCR_MODEL_NAME})...")
    recognizer = LicensePlateRecognizer(OCR_MODEL_NAME)

    detail = []
    for i, (tip, nivo, ssim, images_dir, labels_dir) in enumerate(conds, 1):
        d = evaluate_condition(tip, nivo, ssim, images_dir, labels_dir, gt, recognizer, detector)
        detail.extend(d)
        g = [r for r in d if r["nacin"] == "gt_box"]
        e = [r for r in d if r["nacin"] == "detektor"]
        s1 = sum(r["tacno"] for r in g) / max(1, len(g))
        s2 = sum(r["tacno"] for r in e) / max(1, len(e))
        print(f"[{i}/{len(conds)}] {tip}/{nivo}: OCR={s1:.3f}  ceo sistem={s2:.3f}")

    if TMP_CROP.exists():
        TMP_CROP.unlink()

    agg = aggregate(detail)
    write_csv(OUT_DETAIL, detail)
    write_csv(OUT_RESULTS, agg)

    det_info = load_detection_nano()
    det, _ = det_info if det_info else ({}, None)
    brk = breakpoints(agg, det)
    write_csv(OUT_BREAK, brk)
    make_plots(agg, det)

    print("\n=== Prag pada (prvi SSIM nivo gde tacnost < 0.5 / mAP50 < 0.5) ===")
    for r in brk:
        print(r)
    print(f"\nRezultati: {Path(OUT_RESULTS).resolve()}")
    print(f"Po tablici: {Path(OUT_DETAIL).resolve()}")
    print(f"Grafikoni: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()