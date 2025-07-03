import cv2
import torch
torch.classes.__path__ = []

import os
import glob
from ultralytics import YOLO
import shutil
import random
import yaml
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt

from fiok_train_helper_functions import load_yaml_config
from fiok_train_helper_functions import adjust_brightness_contrast, histogram_equalization, clahe, denoise_image
from fiok_train_helper_functions import create_mosaic
from fiok_train_helper_functions import train_data_flip

from ultralytics import settings
# Return a specific setting
value = settings["runs_dir"]

print("#INFO: Importok OK")

config_path = "config.yaml"  # Cseréld ki a megfelelő fájlútra
config = load_yaml_config(config_path)

if config:
    print("Konfiguráció tartalma:")
    print(config)

train_dir_del = os.path.join(config["input_dir"], "train")
val_dir_del = os.path.join(config["input_dir"], "val")
try:
    shutil.rmtree(train_dir_del)
    shutil.rmtree(val_dir_del)
    print("#INFO: Meglévő Train és Val könyvtárak törölve!")
except:
    print("#INFO: Nem volt train és val könyvtár!")

# Kép betöltése
image = cv2.imread('Frame_20241029_065212_544_LoaderFS1001FK_StorageHangerFaceplateStateDetect_Right.jpg')

# Fényviszonyok javítása
bright_contrast_image = adjust_brightness_contrast(image, alpha=1.5, beta=30)
histogram_image = histogram_equalization(image)
clahe_image = clahe(image)

# Zajcsökkentés alkalmazása minden módosított képre
bright_contrast_image_denoised = denoise_image(bright_contrast_image)
histogram_image_denoised = denoise_image(histogram_image)
clahe_image_denoised = denoise_image(clahe_image)

# Képek megjelenítése
#cv2.imshow("Original Image", image)
#cv2.imshow("Brightness and Contrast Adjusted", bright_contrast_image)
#cv2.imshow("Histogram Equalized", histogram_image)
#cv2.imshow("CLAHE Applied", clahe_image)

#cv2.imshow("Denoised Brightness and Contrast Adjusted", bright_contrast_image_denoised)
#cv2.imshow("Denoised Histogram Equalized", histogram_image_denoised)
#cv2.imshow("Denoised CLAHE Applied", clahe_image_denoised)

# Kép mentése, ha szükséges
cv2.imwrite("_adjusted_brightness_contrast.jpg", bright_contrast_image)
cv2.imwrite("_histogram_equalized.jpg", histogram_image)
cv2.imwrite("_clahe_image.jpg", clahe_image)
cv2.imwrite("_denoised_brightness_contrast.jpg", bright_contrast_image_denoised)
cv2.imwrite("_denoised_histogram_equalized.jpg", histogram_image_denoised)
cv2.imwrite("_denoised_clahe_image.jpg", clahe_image_denoised)
print("#INFO: Összehasonlító képek mentése kész")

#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Képek és feliratok listája
images = [
    bright_contrast_image,
    histogram_image,
    clahe_image,
    bright_contrast_image_denoised,
    histogram_image_denoised,
    clahe_image_denoised
]

titles = [
    "Brightness & Contrast Adjusted",
    "Histogram Equalized",
    "CLAHE Applied",
    "Denoised Brightness & Contrast",
    "Denoised Histogram Equalized",
    "Denoised CLAHE"
]

# Mozaik létrehozása (2 sor, 3 oszlop)
mosaic_image = create_mosaic(images, titles, rows=2, cols=3)

# Mozaik kép megjelenítése
cv2.imshow("Mosaic Image", cv2.resize(mosaic_image, (1280, 720)))

# Kép mentése
cv2.imwrite("mosaic_image.jpg", mosaic_image)
print("#INFO: Mozaik kép mentése kész!")

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.resize(mosaic_image, (1920, 1080))
# Matplotlib segítségével a mozaik képek vizuális megjelenítése
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(mosaic_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mosaic of Image Processing Methods")
plt.show()


def setup_yolov11_dataset(input_dir, output_dir, split_ratio=0.8):
    """
    Létrehozza a YOLOv11 betanításhoz szükséges könyvtárszerkezetet és szétosztja az adatokat.

    :param input_dir: Az a könyvtár, amely tartalmazza a .jpg és .txt fájlokat.
    :param output_dir: Az a könyvtár, ahol a kimeneti struktúra létrejön.
    :param split_ratio: A tréning és validációs adathalmaz aránya (pl. 0.8 = 80% tréning).
    """
    # Ellenőrizze, hogy az input könyvtár létezik-e
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"A megadott bemeneti könyvtár nem létezik: {input_dir}")

    # YOLOv8 könyvtárszerkezet létrehozása
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")

    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(directory, exist_ok=True)

    # Az összes .jpg és .txt fájl beolvasása
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    label_files = glob.glob(os.path.join(input_dir, "*.txt"))

    # Az azonos nevű fájlok párosítása
    base_names = set(os.path.splitext(os.path.basename(f))[0] for f in image_files)
    paired_files = [(os.path.join(input_dir, f + ".jpg"), os.path.join(input_dir, f + ".txt"))
                    for f in base_names if os.path.join(input_dir, f + ".txt") in label_files]

    # A fájlok véletlenszerű sorrendbe állítása
    random.shuffle(paired_files)

    # Az adatok szétosztása tréning és validációs halmazokra
    split_index = int(len(paired_files) * split_ratio)
    train_files = paired_files[:split_index]
    val_files = paired_files[split_index:]

    # Fájlok átmásolása a megfelelő könyvtárakba
    for image_path, label_path in train_files:
        shutil.copy(image_path, train_images_dir)
        shutil.copy(label_path, train_labels_dir)

    for image_path, label_path in val_files:
        shutil.copy(image_path, val_images_dir)
        shutil.copy(label_path, val_labels_dir)

    print(f"#INFO: Adatok sikeresen szétosztva és átmásolva a YOLOv11 könyvtárszerkezetbe.")
    print(f"#INFO: Tréning adatok: {len(train_files)} fájl, Validációs adatok: {len(val_files)} fájl.")

    files = glob.glob(os.path.join(train_images_dir, "*.jpg"))
    for img in files:
        raw_image = cv2.imread(img)
        clahe_image = clahe(raw_image)
        cv2.imwrite(img, clahe_image)

    files = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    for img in files:
        raw_image = cv2.imread(img)
        clahe_image = clahe(raw_image)
        cv2.imwrite(img, clahe_image)

    print("#INFO: Képek előfeldolgozása CLAHE algoritmussal kész!")

setup_yolov11_dataset(config["input_dir"], config["output_dir"], split_ratio=0.8)

if config.flip == "yes":
    train_data_flip()

print("#INFO: Betanítás kezdődik:")
model = YOLO("yolo11n.pt")
train_results = model.train(
    data="dataset_custom.yaml",  # path to dataset YAML
    epochs=20,  # number of training epochs
    imgsz=640,  # training image size
    device='cpu',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch = 8,
    workers = 0,
    patience = 8,
    #project = "loader",
    #name = "11s_clahe"
)

