import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from google.colab import drive # type: ignore
import torch
class Config:
    
    def __init__(self):
        drive.mount('/content/drive')
        self.SEED = 123
        self.columns_to_drop =['avg_bw','time', 'std', 'time.1', 'std.1', 'time.2', 'std.2',
                     'time.3', 'std.3', 'time.4', 'std.4', 'time.5', 'std.5', 'Unnamed: 26']
        
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.BASE_PATH = '/content/drive/MyDrive/ALN/tarea_final'
        self.PATH_ARCHIVO_ENTRADA = os.path.join(self.BASE_PATH, 'pacosi_rtx2080_sinbhsparse.ods')
        self.PATH_CARPETA_MATRICES_PNG = os.path.join(self.BASE_PATH, 'sparse_matrices_png')
        self.PATH_FEATURES_ADICIONALES = os.path.join(self.BASE_PATH, 'features')
        self.PATH_CHECKPOINTS = os.path.join(self.BASE_PATH, 'checkpoints')
        self.MATRIX_RAW_DIR = os.path.join(self.BASE_PATH, 'matrix_raw')
        self.DATASET_YOLO = os.path.join(self.BASE_PATH, "matrices_yolo")
        self.RUNS_FOLDER = os.path.join(self.BASE_PATH, "runs_yolo")
        self.PROCESSED_DATA_DIR = os.path.join(self.BASE_PATH, "processed_data")

        os.makedirs(self.RUNS_FOLDER, exist_ok=True)
        os.makedirs(self.MATRIX_RAW_DIR, exist_ok=True)
        os.makedirs(self.PATH_CARPETA_MATRICES_PNG, exist_ok=True)
        os.makedirs(self.PATH_FEATURES_ADICIONALES, exist_ok=True)
        os.makedirs(self.PATH_CHECKPOINTS, exist_ok=True)
        os.makedirs(self.DATASET_YOLO, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)

        self.MODELS = {
            'SVM': SVC(class_weight='balanced', random_state=self.SEED),
            'MLP': MLPClassifier(max_iter=10000, random_state=self.SEED),
            'RandomForest': RandomForestClassifier(random_state=self.SEED),
            'LogisticRegression': LogisticRegression(max_iter=10000, random_state=self.SEED),
            'KNN': KNeighborsClassifier(),
            'DecisionTree': DecisionTreeClassifier(random_state=self.SEED)
        }

        self.PARAM_GRID = {
            'SVM': {
                'svm__C': [0.1, 1, 10],
                'svm__kernel': ['linear', 'rbf']
            },
            'MLP': {
                'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'mlp__activation': ['relu', 'tanh'],
                'mlp__alpha': [0.0001, 0.001, 0.01]
            },
            'RandomForest': {
                'randomforest__n_estimators': [50, 100, 200],
                'randomforest__max_depth': [None, 10, 20],
                'randomforest__min_samples_split': [2, 5]
            },
            'LogisticRegression': {
                'logisticregression__C': [0.01, 0.1, 1, 10],
                'logisticregression__solver': ['lbfgs', 'liblinear']
            },
            'KNN': {
                'knn__n_neighbors': [3, 5, 7],
                'knn__weights': ['uniform', 'distance']
            },
            'DecisionTree': {
                'decisiontree__max_depth': [None, 10, 20],
                'decisiontree__min_samples_split': [2, 5, 10]
            }
        }
