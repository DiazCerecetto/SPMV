import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from torchvision import models

from sklearn.decomposition import TruncatedSVD


class FeatureExtraction:
    
    def extract_features(self, model, loader, device):
        model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                features = model(images).squeeze()
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())

        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)
        return features_array, labels_array

    def visualize_pca(self, features, labels, epoch, label2class, dimension=2):
        assert dimension in [2, 3], "dimension debe ser 2 o 3."
        pca = PCA(n_components=dimension)
        reduced = pca.fit_transform(features)

        unique_labels = np.unique(labels)
        fig = plt.figure(figsize=(8, 6))

        if dimension == 2:
            ax = fig.add_subplot(111)
            for lab in unique_labels:
                idxs = (labels == lab)
                ax.scatter(reduced[idxs, 0], reduced[idxs, 1],
                           label=label2class[lab], alpha=0.7)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA Época {epoch+1}")
            ax.legend()
        else:
            ax = fig.add_subplot(111, projection='3d')
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
        if device is None:
            device = Config.device

        os.makedirs(save_dir, exist_ok=True)

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model = model.to(device)

        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor = feature_extractor.to(device)

        criterion = nn.CrossEntropyLoss()
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params_to_optimize, lr=1e-4)

        best_val_loss = float("inf")
        best_model_path = os.path.join(save_dir, "best_model.pth")
        last_model_path = os.path.join(save_dir, "last_model.pth")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader,
                                       desc=f"Epoch {epoch+1}/{epochs} [Train]",
                                       leave=False):
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

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            features_list = []
            labels_list = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader,
                                           desc=f"Epoch {epoch+1}/{epochs} [Val]",
                                           leave=False):
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

            print(f"[Época {epoch+1}/{epochs}] "
                  f"Pérdida Train: {train_loss:.4f} | Acc. Train: {train_acc:.4f} || "
                  f"Pérdida Val: {val_loss:.4f} | Acc. Val: {val_acc:.4f}")

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
                features_array = np.concatenate(features_list, axis=0)
                labels_array = np.concatenate(labels_list, axis=0)

                if len(features_array) > num_samples_for_pca:
                    indices = np.random.choice(len(features_array), num_samples_for_pca, replace=False)
                    features_array = features_array[indices]
                    labels_array = labels_array[indices]

                self.visualize_pca(
                    features_array,
                    labels_array,
                    epoch,
                    label2class,
                    dimension=dimension_pca
                )

        torch.save(model.state_dict(), last_model_path)
        print(f"Último modelo guardado en: {last_model_path}")
        return model

    def obtener_resnet_pretrained(self, weights=models.ResNet50_Weights.IMAGENET1K_V2):
        model_pretrained = models.resnet50(weights=weights)
        num_ftrs = model_pretrained.fc.in_features
        model_pretrained.fc = nn.Linear(num_ftrs, 5)

        model_path = os.path.join(Config.PATH_CHECKPOINTS, "best_model.pth")
        model_pretrained.load_state_dict(torch.load(model_path))
        model_pretrained.eval()
        return model_pretrained

    def extract_features_to_df(self, df, model, transform, device=torch.device("cpu"), prefix="feat"):
        model.eval()
        model.to(device)

        df_out = df.copy()
        all_features = []

        for _, row in tqdm(df_out.iterrows(), total=len(df_out), desc=f"Extrayendo features [{prefix}]"):
            image_path = row["path_png"]
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model(img_tensor)

            features = features.squeeze().cpu().numpy()
            all_features.append(features)

        all_features = np.array(all_features)
        num_features = all_features.shape[1]
        for dim_idx in range(num_features):
            col_name = f"{prefix}_{dim_idx}"
            df_out[col_name] = all_features[:, dim_idx]

        return df_out

    def extract_features_with_svd(self, df, model, transform, device=torch.device("cpu"), svd_components=10, prefix="svd_feat"):
        model.eval()
        model.to(device)

        df_out = df.copy()
        all_features = []

        for _, row in tqdm(df_out.iterrows(), total=len(df_out), desc=f"Extrayendo características del modelo"):
            image_path = row["path_png"]
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model(img_tensor)

            features = features.squeeze().cpu().numpy()
            all_features.append(features)

        all_features = np.array(all_features)
        svd = TruncatedSVD(n_components=svd_components)
        reduced_features = svd.fit_transform(all_features)

        for dim_idx in range(svd_components):
            col_name = f"{prefix}_{dim_idx}"
            df_out[col_name] = reduced_features[:, dim_idx]

        return df_out
