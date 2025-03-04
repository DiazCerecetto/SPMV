import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm

class ImageManager:
    
    def descargar_imagen_png(self, grupo, matriz, carpeta_png):
        os.makedirs(carpeta_png, exist_ok=True)
        archivo_png = os.path.join(carpeta_png, f"{matriz}.png")
        url_png = f"https://sparse-files-images.engr.tamu.edu/{grupo}/{matriz}.png"
        try:
            response = requests.get(url_png, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(archivo_png, 'wb') as f, tqdm(
                    desc=f"Descargando {matriz}.png",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                        bar.update(len(chunk))
            else:
                print(f"No se pudo descargar la imagen PNG {matriz}. "
                      f"Status code: {response.status_code}")
        except Exception as e:
            print(f"Excepción al descargar la imagen PNG {matriz}: {e}")

    def check_images(self, df, column_name):
        unreadable_images = []
        indices_to_drop = []
        for index, row in df.iterrows():
            image_path = row[column_name]
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    img.verify()
                    img.close()
                except (IOError, OSError) as e:
                    print(f"Imagen: {image_path}, Error: {e}")
                    unreadable_images.append(image_path)
                    indices_to_drop.append(index)
            else:
                print(f"Archivo no encontrado: {image_path}")
                unreadable_images.append(image_path)
                indices_to_drop.append(index)

        if unreadable_images:
            print("\nLista de imágenes que no se pudieron leer:")
            for img in unreadable_images:
                print(img)

            df.drop(indices_to_drop, inplace=True)
            print(f"\nBorradas {len(indices_to_drop)} filas.")
        else:
            print("\nTodas son legibles.")
        return df

    def crop_to_content(self, image, threshold=240):
        gray_image = image.convert("L")
        image_array = np.array(gray_image)
        mask = image_array < threshold
        coords = np.argwhere(mask)
        if coords.size == 0:
            return image
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return image.crop((y_min, x_min, y_max + 1, x_max + 1))

    def recortar_y_guardar(self,data, conjunto):
        for count, (_, row) in enumerate(data.iterrows(), start=1):
            print(f"Procesando {conjunto} {count}/{len(data)}")
            img = Image.open(row['path_png']).convert('RGB')
            cropped_img = self.crop_to_content(img)
            cropped_img.save(row['path_png'])

    def plot_images_by_label(self, data, num_images_per_row=5):
        labels = data['ganador'].unique()
        labels.sort()

        _, axes = plt.subplots(len(labels), num_images_per_row,
                                 figsize=(num_images_per_row * 2, len(labels) * 2))

        if len(labels) == 1:
            axes = [axes]
        for i, label in enumerate(labels):
            label_data = data[data['ganador'] == label]
            label_images = label_data['path_png'].iloc[:num_images_per_row]
            for j, ax in enumerate(axes[i]):
                if j < len(label_images):
                    image_path = label_images.iloc[j]
                    img = mpimg.imread(image_path)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f"Class: {label}")
                else:
                    ax.axis('off')

        plt.tight_layout()
        plt.show()

    def get_image_shapes(self,data):
        image_shapes = []
        for index, row in data.iterrows():
            image_path = row['path_png']
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        image_shapes.append((img.size, image_path))
                except (IOError, OSError) as e:
                    print(f"Error opening image {image_path}: {e}")
        return sorted(image_shapes, key=lambda x: x[0][0] * x[0][1], reverse=False)

    def get_unusual_images(self, data, limit=10):
        sorted_shapes = self.get_image_shapes(data)
        print("Image Shapes (Largest to Smallest):")
        for i in range(min(limit, len(sorted_shapes))):
            shape, path = sorted_shapes[i]
            print(f"Shape: {shape}, Path: {path}")

