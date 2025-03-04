import numpy as np
import pandas as pd
from PIL import Image

import torch

from sklearn.decomposition import TruncatedSVD


import plotly.express as px
from transformers import ViTModel, ViTImageProcessor
from config import Config


class TransformerFeatureExtractor:
    def __init__(self):
        self.config = Config()
    
    def extract_and_visualize_svd_vit(self, data, model, processor, num_images_per_class=20, dimension=3,device=None):

        if device is None:
            device =  self.config.device

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

        df = pd.DataFrame(reduced_features, 
                          columns=[f"Dim{dim}" for dim in range(1, dimension + 1)])
        df['Etiqueta'] = all_labels

        if dimension == 2:
            fig = px.scatter(
                df, x='Dim1', y='Dim2', color='Etiqueta',
                title="Visualización 2D - ViT Features (SVD)",
                labels={"Dim1": "Dimensión 1", "Dim2": "Dimensión 2"}
            )
        else:
            fig = px.scatter_3d(
                df, x='Dim1', y='Dim2', z='Dim3', color='Etiqueta',
                title="Visualización 3D - ViT Features (SVD)",
                labels={"Dim1": "Dimensión 1", "Dim2": "Dimensión 2", "Dim3": "Dimensión 3"}
            )
            fig.update_traces(marker=dict(size=5))

        fig.update_layout(width=900, height=600)
        fig.show()

    def test_vit(self, data, num_images_per_class=20, dimension=3):
        processor_vit = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        TransformerFeatureExtractor.extract_and_visualize_svd_vit(
            data=data,
            model=model_vit,
            processor=processor_vit,
            num_images_per_class=num_images_per_class,
            dimension=dimension,
            device=Config.device
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
