from pathlib import Path
from ultralytics import YOLO
import cv2


# Putanja do gotovog modela za detekciju tablica
MODEL_PATH = "models/license-plate-finetune-v1m.pt"

# Folder sa sirovim, neobelezenim slikama 
INPUT_IMAGES_DIR = "data/raw_images"

# Gde upisati YOLO anotacije (.txt fajlove) - ovo kasnije ide u trening
OUTPUT_LABELS_DIR = "data/auto_labels"

# Gde sacuvati slike sa nacrtanim okvirom
OUTPUT_PREVIEW_DIR = "data/auto_preview"

# Minimalna pouzdanost predikcije da se uopste zapise (0.0 - 1.0)
# Nizu vrednost -> vise predloga, ali i vise gresaka za rucnu korekciju
# Visu vrednost -> manje predloga, ali sigurniji (mozda propusti neke tablice)
CONFIDENCE_THRESHOLD = 0.25

# Klasa koju upisujemo u anotaciju - kod nas samo jedna klasa: "tablica" = 0
CLASS_ID = 0


def main():
    input_dir = Path(INPUT_IMAGES_DIR)
    labels_dir = Path(OUTPUT_LABELS_DIR)
    preview_dir = Path(OUTPUT_PREVIEW_DIR)

    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        
        print(f"GRESKA: Folder sa slikama ne postoji: {input_dir.resolve()}")
        print("Proveri putanju u INPUT_IMAGES_DIR na vrhu skripte.")
        return

    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    if not image_paths:
        print(f"Nema slika (.jpg/.png) u folderu: {input_dir.resolve()}")
        return

    print(f"Pronadjeno {len(image_paths)} slika. Ucitavam model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    total_no_detection = 0

    for i, img_path in enumerate(image_paths, start=1):
        results = model.predict(
            source=str(img_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            save=False,
        )

        result = results[0]
        boxes = result.boxes  # detektovani okviri za ovu sliku

        label_lines = []
        if boxes is not None and len(boxes) > 0:
            # boxes.xywhn = normalizovane (0-1) koordinate: center_x, center_y, w, h
            for xywhn in boxes.xywhn:
                x_center, y_center, w, h = xywhn.tolist()
                label_lines.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        else:
            total_no_detection += 1

        # Upis .txt anotacije (prazan fajl ako nista nije detektovano - to je
        # signal da tu sliku treba rucno pregledati i obeleziti, ili odbaciti)
        label_path = labels_dir / (img_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        # Cuvanje vizuelne verzije za brzu proveru
        annotated = result.plot()  # numpy array sa nacrtanim okvirima
        preview_path = preview_dir / img_path.name
        cv2.imwrite(str(preview_path), annotated)

        if i % 50 == 0 or i == len(image_paths):
            print(f"  Obradjeno {i}/{len(image_paths)} slika...")

    print("\nGotovo.")
    print(f"Anotacije sacuvane u: {labels_dir.resolve()}")
    print(f"Vizuelna provera u:   {preview_dir.resolve()}")
    print(f"Slike BEZ detekcije (prazan .txt, treba rucna provera): {total_no_detection}")


if __name__ == "__main__":
    main()