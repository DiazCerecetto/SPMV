from skimage import io# type: ignore
import numpy as np
import pandas as pd
from skimage.color import rgb2gray # type: ignore
from skimage.transform import resize# type: ignore
from skimage.feature import hog# type: ignore
import cv2

from tqdm import tqdm

class TraditionalFeatureExtractor:
    
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
        for _, row in df.iterrows():
            path = row['path_png']
            try:
                img = io.imread(path, as_gray=True)
                feats = TraditionalFeatureExtractor.block_sampling_limited_features(
                    img, num_blocks_h, num_blocks_w
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

        new_df['ganador_encoded'] = df['ganador_encoded'].values
        return new_df

    def extract_hog_features(self, image_path, limit_features=20):
        image = io.imread(image_path)
        if len(image.shape) == 3:
            image = rgb2gray(image)
        image_resized = resize(image, (128, 128))
        features, _ = hog(image_resized, pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2), visualize=True)
        # Extra info
        mean_intensity = np.mean(image_resized)
        std_intensity  = np.std(image_resized)
        max_intensity  = np.max(image_resized)
        min_intensity  = np.min(image_resized)

        feature_vector = np.hstack([features, mean_intensity, std_intensity, 
                                    max_intensity, min_intensity])
        # Limitamos a 20
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
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            features = feature_extraction_func(row['path_png'])
            feature_list.append(features)
        features_df = pd.DataFrame(feature_list, 
                                   columns=[f'feature_{i}' for i in range(1, 21)])
        return pd.concat([df.reset_index(drop=True), features_df], axis=1)

