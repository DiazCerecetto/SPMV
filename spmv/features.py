import os
import cv2
import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import plotly.express as px
import matplotlib.pyplot as plt
import json
from PIL import Image
from tqdm import tqdm
from skimage import io # type: ignore
from skimage.color import rgb2gray # type: ignore
from skimage.transform import resize # type: ignore
from skimage.feature import hog # type: ignore
from torchvision import models
from sklearn.decomposition import PCA, TruncatedSVD
from transformers import ViTModel, ViTImageProcessor
from torchvision import transforms
from ultralytics import YOLO # type: ignore
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
class FeatureExtractor:
    def __init__(self, config):
        self.config = config
    
    def _extract_features_batch(self, model, images, device):
        images = images.to(device)
        features = model(images).squeeze()
        return features.cpu().numpy()

    def _concat_arrays(self, features_list, labels_list):
        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)
        return features_array, labels_array

    def _plot_pca_2d(self, reduced, labels, label2class, epoch):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            idxs = (labels == lab)
            ax.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label2class[lab], alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA Época {epoch+1}")
        ax.legend()
        plt.show()

    def _plot_pca_3d(self, reduced, labels, label2class, epoch):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            idxs = (labels == lab)
            ax.scatter(reduced[idxs, 0], reduced[idxs, 1], reduced[idxs, 2],
                       label=label2class[lab], alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"PCA Época {epoch+1}")
        ax.legend()
        plt.show()

    def extract_features(self, model, loader, device):
        model.eval()
        features_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in loader:
                batch_features = self._extract_features_batch(model, images, device)
                features_list.append(batch_features)
                labels_list.append(labels.numpy())
        return self._concat_arrays(features_list, labels_list)

    def visualize_pca(self, features, labels, epoch, label2class, dimension=2):
        assert dimension in [2, 3]
        pca = PCA(n_components=dimension)
        reduced = pca.fit_transform(features)
        if dimension == 2:
            self._plot_pca_2d(reduced, labels, label2class, epoch)
        else:
            self._plot_pca_3d(reduced, labels, label2class, epoch)

    def _train_one_epoch(self, model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        return train_loss, train_acc

    def _validate_one_epoch(self, model, loader, criterion, feature_extractor, device):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        features_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in tqdm(loader, leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                features = feature_extractor(images).squeeze()
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        feats_array, labs_array = self._concat_arrays(features_list, labels_list)
        return val_loss, val_acc, feats_array, labs_array

    def finetune_resnet50(
        self,
        train_loader,
        val_loader,
        num_classes,
        epochs=10,
        device=None,
        patience=5,
        save_dir="./checkpoints",
        dimension_pca=2,
        num_samples_for_pca=50,
        label2class=None
    ):
        device = self.config.device
        os.makedirs(save_dir, exist_ok=True)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model = model.to(device)
        feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
        criterion = nn.CrossEntropyLoss()
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params_to_optimize, lr=1e-4)
        best_val_loss = float("inf")
        best_model_path = os.path.join(save_dir, "best_model.pth")
        last_model_path = os.path.join(save_dir, "last_model.pth")
        epochs_without_improvement = 0
        for epoch in range(epochs):
            train_loss, train_acc = self._train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, features_array, labels_array = self._validate_one_epoch(
                model, val_loader, criterion, feature_extractor, device
            )
            print(
                f"[Época {epoch+1}/{epochs}] "
                f"Pérdida Train: {train_loss:.4f} | Acc. Train: {train_acc:.4f} || "
                f"Pérdida Val: {val_loss:.4f} | Acc. Val: {val_acc:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"--> ¡Mejor modelo guardado! {best_model_path}")
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early Stopping en la época {epoch+1}.")
                break
            if label2class is not None and num_samples_for_pca > 0:
                if len(features_array) > num_samples_for_pca:
                    indices = np.random.choice(len(features_array), num_samples_for_pca, replace=False)
                    features_array = features_array[indices]
                    labels_array = labels_array[indices]
                self.visualize_pca(features_array, labels_array, epoch, label2class, dimension=dimension_pca)
        torch.save(model.state_dict(), last_model_path)
        print(f"Último modelo guardado en: {last_model_path}")
        return model

    def obtener_resnet_pretrained(self, weights=models.ResNet50_Weights.IMAGENET1K_V2):
        model_pretrained = models.resnet50(weights=weights)
        num_ftrs = model_pretrained.fc.in_features
        model_pretrained.fc = nn.Linear(num_ftrs, 5)
        model_path = os.path.join(self.config.PATH_CHECKPOINTS, "best_model.pth")
        model_pretrained.load_state_dict(torch.load(model_path))
        model_pretrained.eval()
        return model_pretrained

    def extract_features_to_df(self, df, model, transform_size=224, prefix="feat"):
        transform = transforms.Compose([transforms.Resize((transform_size, transform_size)), transforms.ToTensor()])
        
        model.eval()
        model.to(self.config.device)
        df_out = df.copy()
        all_features = []
        times = []
        for _, row in tqdm(df_out.iterrows(), total=len(df_out)):
            img = Image.open(row["path_png"]).convert("RGB")
            start = time.perf_counter()
            img_tensor = transform(img).unsqueeze(0).to(self.config.device)
            with torch.no_grad():
                features = model(img_tensor)
            features = features.squeeze().cpu().numpy()
            end = time.perf_counter()
            times.append(end - start)
            all_features.append(features)
        all_features = np.array(all_features)
        for dim_idx in range(all_features.shape[1]):
            df_out[f"{prefix}_{dim_idx}"] = all_features[:, dim_idx]
        df_out["time_extraction_sec"] = times
        return df_out

    def extract_features_with_svd(self, df, model, transform_size=224, svd_components=10, prefix="svd_feat"):
        transform = transforms.Compose([transforms.Resize((transform_size, transform_size)), transforms.ToTensor()])
        model.eval()
        model.to(self.config.device)
        df_out = df.copy()
        all_features = []
        times = []
        for _, row in tqdm(df_out.iterrows(), total=len(df_out)):
            img = Image.open(row["path_png"]).convert("RGB")
            start = time.perf_counter()
            img_tensor = transform(img).unsqueeze(0).to(self.config.device)
            with torch.no_grad():
                features = model(img_tensor)
            end = time.perf_counter()
            times.append(end - start)
            all_features.append(features.squeeze().cpu().numpy())
        all_features = np.array(all_features)
        svd = TruncatedSVD(n_components=svd_components)
        reduced_features = svd.fit_transform(all_features)
        for dim_idx in range(svd_components):
            df_out[f"{prefix}_{dim_idx}"] = reduced_features[:, dim_idx]
        df_out["time_extraction_sec"] = times
        return df_out

    def _apply_svd(self, features, dimension):
        svd = TruncatedSVD(n_components=dimension)
        return svd.fit_transform(features)

    def _create_plotly_df(self, reduced_features, labels, dimension):
        df = pd.DataFrame(
            reduced_features,
            columns=[f"Dim{dim}" for dim in range(1, dimension + 1)]
        )
        df['Etiqueta'] = labels
        return df

    def _plot_svd_2d(self, df):
        fig = px.scatter(
            df,
            x='Dim1',
            y='Dim2',
            color='Etiqueta',
            title="Visualización 2D de Features por Clase (SVD)",
            labels={"Dim1": "Dimensión 1", "Dim2": "Dimensión 2"},
            hover_data={'Etiqueta': True}
        )
        fig.update_layout(width=1000, height=600)
        fig.show()

    def _plot_svd_3d(self, df):
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

    def extract_and_visualize_svd_resnet(
        self,
        data,
        model,
        processor=None,
        transform_size=224,
        num_images_per_class=20,
        dimension=3,
        device=None
    ):
        transform = transforms.Compose([transforms.Resize((transform_size, transform_size)), transforms.ToTensor()])
        device = self.config.device
        model.eval()
        all_features = []
        all_labels = []
        unique_classes = data['ganador'].unique()
        with torch.no_grad():
            for class_label in unique_classes:
                class_data = data[data['ganador'] == class_label].head(num_images_per_class)
                for _, row in class_data.iterrows():
                    img = Image.open(row['path_png']).convert("RGB")
                    if processor:
                        inputs = processor(images=img, return_tensors="pt").to(device)
                        outputs = model(**inputs)
                        features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                    elif transform:
                        img_tensor = transform(img).unsqueeze(0).to(device)
                        model = model.to(device)
                        features = model(img_tensor).cpu().numpy().flatten()
                    else:
                        raise ValueError("Debes proporcionar un 'processor' o un 'transform size'.")
                    all_features.append(features)
                    all_labels.append(class_label)
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        reduced_features = self._apply_svd(all_features, dimension)
        df_plot = self._create_plotly_df(reduced_features, all_labels, dimension)
        if dimension == 2:
            self._plot_svd_2d(df_plot)
        else:
            self._plot_svd_3d(df_plot)

    def test_resnet(self, data, transform_size=224, num_images_per_class=20, dimension=3):
        transform = transforms.Compose([transforms.Resize((transform_size, transform_size)), transforms.ToTensor()])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
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

        model.eval()
        model.to(device)

        all_features = []
        all_labels = []
        all_indices = []
        inference_times = []

        unique_classes = data['ganador'].unique()

        with torch.no_grad():
            for class_label in unique_classes:
                class_data = data[data['ganador'] == class_label].head(num_images_per_class)

                for idx, row in class_data.iterrows():
                    img = Image.open(row['path_png']).convert("RGB")
                    start_time = time.perf_counter()
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    end_time = time.perf_counter()

                    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                    all_features.append(features)
                    all_labels.append(class_label)
                    all_indices.append(idx)
                    inference_times.append(end_time - start_time)

        all_features = np.array(all_features)
        all_labels   = np.array(all_labels)
        reduced_features = self._apply_svd(all_features, dimension)

        df_plot = self._create_plotly_df(reduced_features, all_labels, dimension)

        if dimension == 2:
            self._plot_svd_2d(df_plot)
        else:
            self._plot_svd_3d(df_plot)

        df_features = pd.DataFrame(
            reduced_features,
            index=all_indices,
            columns=[f"vit_feat_{i}" for i in range(dimension)]
        )
        df_features["time_extraction_sec"] = inference_times

        data = data.join(df_features, how="left")

        return data


    def test_vit(self, data, num_images_per_class=20, dimension=3):
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

    def reduce_dimensionality(self, df, prefix_in="vit_feat", prefix_out="vit_svd", n_components=10):
        feature_cols = [c for c in df.columns if c.startswith(prefix_in)]
        X = df[feature_cols].values
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X)
        df_out = df.copy()
        for i in range(n_components):
            df_out[f"{prefix_out}_{i}"] = X_reduced[:, i]
        return df_out

    def block_sampling_limited_features(self, img, num_blocks_h=4, num_blocks_w=5):
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

    def generar_features_limited_block_sampling(self, df, num_blocks_h=4, num_blocks_w=5):
        feature_list = []
        times = []
        for _, row in df.iterrows():
            start = time.perf_counter()
            path = row['path_png']
            try:
                img = io.imread(path, as_gray=True)
                feats = self.block_sampling_limited_features(img, num_blocks_h, num_blocks_w)
            except:
                feats = np.zeros(num_blocks_h * num_blocks_w, dtype=np.float32)
            end = time.perf_counter()
            times.append(end - start)
            feature_list.append(feats)
        feature_array = np.array(feature_list)
        columns = [f"feat_block_{i}" for i in range(num_blocks_h * num_blocks_w)]
        features_df = pd.DataFrame(feature_array, columns=columns)
        numeric_cols = ["rows", "cols", "aspect", "nnz", "min_nnz", "max_nnz", "avg_nnz", "std_nnz"]
        existing_numeric_cols = [c for c in numeric_cols if c in df.columns]
        new_df = pd.concat([df[existing_numeric_cols].reset_index(drop=True),
                            features_df.reset_index(drop=True)], axis=1)
        new_df["time_extraction_sec"] = times
        if 'ganador_encoded' in df.columns:
            new_df['ganador_encoded'] = df['ganador_encoded'].values
        return new_df

    def extract_hog_features(self, image_path, limit_features=20):
        image = io.imread(image_path)
        if len(image.shape) == 3:
            image = rgb2gray(image)
        image_resized = resize(image, (128, 128))
        features, _ = hog(image_resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
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
        feature_vector = feature_vector / np.sum(feature_vector)
        return feature_vector[:limit_features]

    def process_dataframe(self, df, feature_extraction_func):
        feature_list = []
        times = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            start = time.perf_counter()
            features = feature_extraction_func(row['path_png'])
            end = time.perf_counter()
            times.append(end - start)
            feature_list.append(features)
        features_df = pd.DataFrame(feature_list, columns=[f'feature_{i}' for i in range(1, 21)])
        features_df['time_extraction_sec'] = times
        return pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    def extract_features_yolo(self, df, model, imgsz=224):
        paths = []
        times = []
        features_list = []

        for _, row in df.iterrows():
            p = row['path_png']
            start = time.perf_counter()

            results = model.predict(source=p, imgsz=imgsz, device='cuda', verbose=False)

            end = time.perf_counter()
            times.append(end - start)

            if results and hasattr(results[0], "probs") and results[0].probs is not None:
                probs_vec = results[0].probs.data.cpu().numpy()
            else:
                probs_vec = np.zeros(len(model.model.names), dtype=float)

            features_list.append(probs_vec)
            paths.append(p)

        features_arr = np.vstack(features_list)
        for i in range(features_arr.shape[1]):
            df[f'yolo_feat_{i}'] = features_arr[:, i]
        df['time_extraction_sec'] = times
        return df

        
    def save_val_results(self, results, results_multi, filepath="val_results.json"):
        filepath = os.path.join(self.config.RESULTS_DIR, filepath)
        def filter_params(params):
            serializable = {}
            for k, v in params.items():
                if k.startswith('preprocessor'):
                    continue
                try:
                    json.dumps(v)
                    serializable[k] = v
                except TypeError:
                    serializable[k] = str(v)
            return serializable

        data_to_save = {}

        for scenario_name, scenario_dict in results.items():
            best_model = scenario_dict["evaluation_results"]["best_model"]
            best_params = filter_params(best_model.get_params()) if best_model else {}
            eval_dict = scenario_dict["evaluation_results"]["evaluation_results"]
            confusion = scenario_dict["evaluation_results"]["confusion_matrix"]
            f1_df = scenario_dict["f1_scores"]

            data_to_save.setdefault("results", {})
            data_to_save["results"][scenario_name] = {
                "best_params": best_params,
                "metrics": eval_dict,
                "confusion_matrix": (confusion.values.tolist() if hasattr(confusion, "values") else []),
                "f1_scores": (f1_df.to_dict(orient='records') if f1_df is not None else [])
            }

        for scenario_name, scenario_dict in results_multi.items():
            best_model = scenario_dict["evaluation_results"]["best_model"]
            best_params = filter_params(best_model.get_params()) if best_model else {}
            eval_dict = scenario_dict["evaluation_results"]["evaluation_results"]
            confusion = scenario_dict["evaluation_results"]["confusion_matrix"]
            f1_df = scenario_dict["f1_scores"]

            data_to_save.setdefault("results_multi", {})
            data_to_save["results_multi"][scenario_name] = {
                "best_params": best_params,
                "metrics": eval_dict,
                "confusion_matrix": (confusion.values.tolist() if hasattr(confusion, "values") else []),
                "f1_scores": (f1_df.to_dict(orient='records') if f1_df is not None else [])
            }

        best_scenario_name = None
        best_macro_f1 = -1.0
        for scenario_name, scenario_dict in results.items():
            if scenario_dict["f1_scores"] is not None:
                macro_value = scenario_dict["f1_scores"]["F1-Score (Macro)"].values[0]
                if macro_value > best_macro_f1:
                    best_macro_f1 = macro_value
                    best_scenario_name = scenario_name
        for scenario_name, scenario_dict in results_multi.items():
            if scenario_dict["f1_scores"] is not None:
                macro_value = scenario_dict["f1_scores"]["F1-Score (Macro)"].values[0]
                if macro_value > best_macro_f1:
                    best_macro_f1 = macro_value
                    best_scenario_name = scenario_name

        data_to_save["best_scenario"] = best_scenario_name
        data_to_save["best_macro_f1"] = best_macro_f1

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"Hiperparámetros y resultados guardados en: {filepath}\nMejor escenario global: {best_scenario_name} (F1-Macro: {best_macro_f1:.4f})")


    def save_topn(self, topn: list, filepath: str = "topn.json"):
        filepath = os.path.join(self.config.RESULTS_DIR, filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(topn, f, indent=4)

    def load_topn(self, filepath: str = "topn.json") -> list:
        filepath = os.path.join(self.config.RESULTS_DIR, filepath)
        if not os.path.exists(filepath):
            return []  # Devuelve una lista vacía si el archivo no existe.
        
        with open(filepath, "r", encoding="utf-8") as f:
            topn = json.load(f)
        
        return topn

    def fine_tune_vit_classifier(self, train_dir, val_dir, num_labels=5, epochs=25, patience=5, lr=2e-5, batch_size=32):
        
        device = self.config.device
        os.makedirs(self.config.PATH_CHECKPOINTS, exist_ok=True)
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        transform = Compose([Resize((224,224)), ToTensor(), Normalize(mean=processor.image_mean, std=processor.image_std)])
        train_dataset = ImageFolder(train_dir, transform=transform)
        val_dataset = ImageFolder(val_dir, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_labels)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            model.train()
            running_loss = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            model.eval()
            running_val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images, labels=labels)
                    loss = outputs.loss
                    running_val_loss += loss.item()
                    preds = outputs.logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(self.config.PATH_CHECKPOINTS, "best_model_vit.pt")
                torch.save(model.state_dict(), checkpoint_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
        model.load_state_dict(torch.load(os.path.join(self.config.PATH_CHECKPOINTS, "best_model_vit.pt")))
        plt.figure(figsize=(10,5))
        plt.plot(train_losses, label="Pérdida de entrenamiento")
        plt.plot(val_losses, label="Pérdida de validación")
        plt.xlabel("Época")
        plt.ylabel("Pérdida")
        plt.title("Pérdida por época")
        plt.legend()
        plt.show()
        return model

    def extract_and_visualize_svd_vit_finetuned(self, data, model, processor, num_images_per_class=20, dimension=3, device=None):
        import torch
        import time
        import numpy as np
        import pandas as pd
        from PIL import Image
        model.eval()
        if device is None:
            device = self.config.device
        model.to(device)
        all_features = []
        all_labels = []
        all_indices = []
        inference_times = []
        unique_classes = data['ganador'].unique()
        with torch.no_grad():
            for class_label in unique_classes:
                class_data = data[data['ganador'] == class_label].head(num_images_per_class)
                for idx, row in class_data.iterrows():
                    img = Image.open(row['path_png']).convert("RGB")
                    start_time = time.perf_counter()
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    outputs = model(**inputs, output_hidden_states=True)
                    end_time = time.perf_counter()
                    hidden = outputs.hidden_states[-1]
                    features = hidden.mean(dim=1).cpu().numpy().flatten()
                    all_features.append(features)
                    all_labels.append(class_label)
                    all_indices.append(idx)
                    inference_times.append(end_time - start_time)
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        reduced_features = self._apply_svd(all_features, dimension)
        df_plot = self._create_plotly_df(reduced_features, all_labels, dimension)
        if dimension == 2:
            self._plot_svd_2d(df_plot)
        else:
            self._plot_svd_3d(df_plot)
        df_features = pd.DataFrame(reduced_features, index=all_indices, columns=[f"vit_feat_{i}" for i in range(dimension)])
        df_features["time_extraction_sec"] = inference_times
        data = data.join(df_features, how="left")
        return data

