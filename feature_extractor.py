import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models

from sklearn.decomposition import TruncatedSVD
import plotly.express as px

from transformers import ViTModel, ViTImageProcessor

from skimage import io # type: ignore
from skimage.color import rgb2gray # type: ignore
from skimage.transform import resize # type: ignore
from skimage.feature import hog # type: ignore
import cv2

from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, config):
        self.config = config

    def extract_and_visualize_svd_resnet(
        self,
        data,
        model,
        processor=None,
        transform=None,
        num_images_per_class=20,
        dimension=3,
        device=None
    ):

        if device is None:
            device = self.config.device

        assert dimension in [2, 3], "La dimensión debe ser 2 o 3"
        model.eval()
        all_features = []
        all_labels = []

        unique_classes = data['ganador'].unique()
        for class_label in unique_classes:
            class_data = data[data['ganador'] == class_label].head(num_images_per_class)

            with torch.no_grad():
                for _, row in class_data.iterrows():
                    image_path = row['path_png']
                    label = row['ganador']

                    img = Image.open(image_path).convert("RGB")
                    if processor:
                        # Si se usara un processor tipo ViTImageProcessor, etc.
                        inputs = processor(images=img, return_tensors="pt").to(device)
                        outputs = model(**inputs)
                        features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                    elif transform:
                        # Transform para imágenes con torchvision
                        img_tensor = transform(img).unsqueeze(0).to(device)
                        model = model.to(device)
                        features = model(img_tensor).cpu().numpy().flatten()
                    else:
                        raise ValueError("Debes proporcionar un 'processor' o un 'transform'.")

                    all_features.append(features)
                    all_labels.append(label)

        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        print(f"Aplicando SVD con {dimension} dimensiones...")
        svd = TruncatedSVD(n_components=dimension)
        reduced_features = svd.fit_transform(all_features)

        df = pd.DataFrame(
            reduced_features,
            columns=[f"Dim{dim}" for dim in range(1, dimension + 1)]
        )
        df['Etiqueta'] = all_labels

        if dimension == 2:
            fig = px.scatter(
                df,
                x='Dim1',
                y='Dim2',
                color='Etiqueta',
                title="Visualización 2D de Features por Clase (SVD)",
                labels={"Dim1": "Dimensión 1", "Dim2": "Dimensión 2"},
                hover_data={'Etiqueta': True}
            )
        else:  # dimension == 3
            fig = px.scatter_3d(
                df,
                x='Dim1',
                y='Dim2',
                z='Dim3',
                color='Etiqueta',
                title="Visualización 3D de Features por Clase (SVD)",
                labels={"Dim1": "Dimensión 1", "Dim2": "Dimensión 2", "Dim3": "Dimensión 3"},
                hover_data={'Etiqueta': True}
            )
            fig.update_traces(marker=dict(size=5))

        fig.update_layout(width=1000, height=600)
        fig.show()

    def test_resnet(
        self,
        data,
        transform,
        num_images_per_class=20,
        dimension=3
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Reemplazamos la FC final con una identidad para que nos devuelva
        # el penúltimo vector de características.
        resnet.fc = nn.Identity()
        resnet = resnet.to(device)

        self.extract_and_visualize_svd_resnet(
            data,
            model=resnet,
            transform=transform,
            num_images_per_class=num_images_per_class,
            dimension=dimension,
            device=device
        )


    def extract_and_visualize_svd_vit(
        self,
        data,
        model,
        processor,
        num_images_per_class=20,
        dimension=3,
        device=None
    ):
        if device is None:
            device = self.config.device

        assert dimension in [2, 3], "La dimensión debe ser 2 o 3"
        model.eval()
        model.to(device)

        all_features = []
        all_labels = []

        unique_classes = data['ganador'].unique()
        for class_label in unique_classes:
            class_data = data[data['ganador'] == class_label].head(num_images_per_class)

            with torch.no_grad():
                for _, row in class_data.iterrows():
                    image_path = row['path_png']
                    label = row['ganador']

                    img = Image.open(image_path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)

                    outputs = model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

                    all_features.append(features)
                    all_labels.append(label)

        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        svd = TruncatedSVD(n_components=dimension)
        reduced_features = svd.fit_transform(all_features)

        df = pd.DataFrame(
            reduced_features,
            columns=[f"Dim{dim}" for dim in range(1, dimension + 1)]
        )
        df['Etiqueta'] = all_labels

        if dimension == 2:
            fig = px.scatter(
                df,
                x='Dim1',
                y='Dim2',
                color='Etiqueta',
                title="Visualización 2D - ViT Features (SVD)",
                labels={"Dim1": "Dimensión 1", "Dim2": "Dimensión 2"}
            )
        else:
            fig = px.scatter_3d(
                df,
                x='Dim1',
                y='Dim2',
                z='Dim3',
                color='Etiqueta',
                title="Visualización 3D - ViT Features (SVD)",
                labels={"Dim1": "Dimensión 1", "Dim2": "Dimensión 2", "Dim3": "Dimensión 3"}
            )
            fig.update_traces(marker=dict(size=5))

        fig.update_layout(width=900, height=600)
        fig.show()

    def test_vit(
        self,
        data,
        num_images_per_class=20,
        dimension=3
    ):
        processor_vit = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.extract_and_visualize_svd_vit(
            data=data,
            model=model_vit,
            processor=processor_vit,
            num_images_per_class=num_images_per_class,
            dimension=dimension,
            device=self.config.device
        )

    def reduce_dimensionality(
        self,
        df,
        prefix_in="vit_feat",
        prefix_out="vit_svd",
        n_components=10
    ):
        feature_cols = [c for c in df.columns if c.startswith(prefix_in)]
        X = df[feature_cols].values
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X)

        df_out = df.copy()
        for i in range(n_components):
            df_out[f"{prefix_out}_{i}"] = X_reduced[:, i]

        return df_out

    def block_sampling_limited_features(
        self,
        img,
        num_blocks_h=4,
        num_blocks_w=5
    ):
        img = np.array(img, dtype=np.float32)
        img_bin = (img > 0).astype(np.float32)
        h, w = img_bin.shape
        block_h = h // num_blocks_h
        block_w = w // num_blocks_w
        features = []
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = img_bin[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                density = np.mean(block)
                features.append(density)
        return np.array(features, dtype=np.float32)

    def generar_features_limited_block_sampling(
        self,
        df,
        num_blocks_h=4,
        num_blocks_w=5
    ):

        feature_list = []
        for _, row in df.iterrows():
            path = row['path_png']
            try:
                img = io.imread(path, as_gray=True)
                feats = self.block_sampling_limited_features(
                    img,
                    num_blocks_h,
                    num_blocks_w
                )
            except Exception as e:
                print(f"Error leyendo la imagen en {path}: {e}")
                feats = np.zeros(num_blocks_h * num_blocks_w, dtype=np.float32)
            feature_list.append(feats)

        feature_array = np.array(feature_list)
        columns = [f"feat_block_{i}" for i in range(num_blocks_h * num_blocks_w)]
        features_df = pd.DataFrame(feature_array, columns=columns)

        numeric_cols = ["rows", "cols", "aspect", "nnz", "min_nnz", "max_nnz",
                        "avg_nnz", "std_nnz"]
        existing_numeric_cols = [c for c in numeric_cols if c in df.columns]

        new_df = pd.concat([
            df[existing_numeric_cols].reset_index(drop=True),
            features_df.reset_index(drop=True)
        ], axis=1)

        if 'ganador_encoded' in df.columns:
            new_df['ganador_encoded'] = df['ganador_encoded'].values

        return new_df

    def extract_hog_features(self, image_path, limit_features=20):

        image = io.imread(image_path)
        if len(image.shape) == 3:
            image = rgb2gray(image)
        image_resized = resize(image, (128, 128))
        features, _ = hog(
            image_resized,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=True
        )
        # Extra info
        mean_intensity = np.mean(image_resized)
        std_intensity = np.std(image_resized)
        max_intensity = np.max(image_resized)
        min_intensity = np.min(image_resized)

        feature_vector = np.hstack([
            features,
            mean_intensity,
            std_intensity,
            max_intensity,
            min_intensity
        ])

        return feature_vector[:limit_features]

    def extract_sift_features(self, image_path, limit_features=20):

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            return np.zeros(limit_features)
        feature_vector = descriptors.mean(axis=0)
        return feature_vector[:limit_features]

    def extract_orb_features(self, image_path, limit_features=20):

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None:
            return np.zeros(limit_features)
        feature_vector = descriptors.mean(axis=0)
        return feature_vector[:limit_features]

    def extract_color_histogram(self, image_path, limit_features=20):

        image = cv2.imread(image_path)
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

        feature_vector = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        feature_vector = feature_vector / np.sum(feature_vector)  # Normalizamos
        return feature_vector[:limit_features]

    def process_dataframe(
        self,
        df,
        feature_extraction_func
    ):

        feature_list = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            features = feature_extraction_func(row['path_png'])
            feature_list.append(features)
        features_df = pd.DataFrame(
            feature_list,
            columns=[f'feature_{i}' for i in range(1, 21)]
        )
        return pd.concat([df.reset_index(drop=True), features_df], axis=1)
