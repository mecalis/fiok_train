import yaml
import cv2
import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np

def load_yaml_config(file_path):
    """
    Beolvassa a YAML konfigurációs fájlt.

    :param file_path: Az elérési út a YAML fájlhoz.
    :return: A YAML fájl tartalma Python dictionary-ként.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)  # A safe_load biztonságosabb, mint a load
            return config
    except FileNotFoundError:
        print(f"Hiba: A fájl nem található: {file_path}")
    except yaml.YAMLError as e:
        print(f"Hiba történt a YAML beolvasásakor: {e}")

def adjust_brightness_contrast(image, alpha=1.2, beta=20):
    """
    Alkalmaz kontraszt és fényerő módosítást.
    alpha: A kontraszt (1.0 nem változtat semmit, >1.0 növeli, <1.0 csökkenti).
    beta: A fényerő (0 nem változtat semmit, pozitív érték növeli, negatív csökkenti).
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def histogram_equalization(image):
    """
    Histogramaegyenlítés szürkeárnyalatú képeken.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

def clahe(image):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) alkalmazása.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

def denoise_image(image):
    """
    Zajcsökkentés a képen bilaterális szűréssel.
    """
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def create_mosaic(images, titles, rows, cols):
    """
    Képek és feliratok alapján mozaik létrehozása.
    images: A képek listája.
    titles: A képekhez tartozó feliratok listája.
    rows: A mozaik sorainak száma.
    cols: A mozaik oszlopainak száma.
    """
    # A képek átméretezése, hogy illeszkedjenek a mozaikba
    max_height = max([img.shape[0] for img in images])
    max_width = max([img.shape[1] for img in images])

    # Új, üres mozaik készítése
    mosaic = np.zeros((max_height * rows, max_width * cols, 3), dtype=np.uint8)

    # A képek elhelyezése a mozaikban
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y_offset = row * max_height
        x_offset = col * max_width
        mosaic[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

        # Felirat hozzáadása a kép alá
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = titles[idx]
        position = (x_offset + 10, y_offset + img.shape[0] - 10)
        cv2.putText(mosaic, text, position, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return mosaic

def train_data_flip():
    # 📂 Könyvtár megadása, ahol a képek és annotációk vannak
    input_dir_images = os.path.join("train", "images")
    input_dir_labels = os.path.join("train", "labels")

    # 🔍 Összegyűjtjük az összes kép fájlt (jpg, png, stb.)
    image_files = glob.glob(os.path.join(input_dir_images, "*.jpg")) + glob.glob(os.path.join(input_dir_images, "*.png"))

    # 🔄 Minden fájlra végrehajtjuk a tükrözést
    for image_path in image_files:
        # 📌 Alapfájlnevek előkészítése
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # Pl.: "image1"
        annotation_path = os.path.join(input_dir_labels, f"{base_name}.txt")  # YOLO annotáció fájl

        # 📌 Kép beolvasása OpenCV-vel
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hiba a kép beolvasásakor: {image_path}")
            continue

        # 📌 Kép vízszintes tükrözése (horizontal flip)
        flipped_image = cv2.flip(image, 1)

        # 📌 Új fájlnevek generálása (_flip kiegészítéssel)
        flipped_image_path = os.path.join(input_dir_images, f"{base_name}_flip.jpg")
        flipped_annotation_path = os.path.join(input_dir_labels, f"{base_name}_flip.txt")

        # 💾 Kép mentése új fájlként
        cv2.imwrite(flipped_image_path, flipped_image)

        # 🔄 Annotáció módosítása, ha létezik a TXT fájl
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                lines = f.readlines()

            flipped_annotations = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Hibás sorokat kihagyjuk

                class_id, x_center, y_center, width, height = map(float, parts)

                # 📌 X koordináta tükrözése
                new_x_center = 1.0 - x_center  # X flip: 1 - x_center

                # Új annotáció formázása
                flipped_annotations.append(
                    f"{int(class_id)} {new_x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # 💾 Új annotációs fájl mentése
            with open(flipped_annotation_path, "w") as f:
                f.writelines(flipped_annotations)

    #print(f"Mentve: {flipped_image_path}, {flipped_annotation_path}")
    print("#INFO: Tükrözött képek és módosított txt fájlok mentése kész")