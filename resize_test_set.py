import shutil
from pathlib import Path
import cv2

LONG_SIDE = 1280
TEST = Path("dataset/test")
ORIG = Path("dataset/test_original")

if ORIG.exists():
    print("dataset/test_original vec postoji")
    raise SystemExit

TEST.rename(ORIG)                       # originali ostaju sacuvani
(TEST / "images").mkdir(parents=True)
shutil.copytree(ORIG / "labels", TEST / "labels")   # anotacije su normalizovane, vaze i posle smanjenja

n = 0
for p in sorted((ORIG / "images").iterdir()):
    img = cv2.imread(str(p))
    if img is None:
        print("Ne mogu da ucitam", p.name)
        continue
    h, w = img.shape[:2]
    scale = LONG_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(TEST / "images" / (p.stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    n += 1
print(f"Gotovo: {n} slika smanjeno u dataset/test/images (originali u dataset/test_original)")