import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ultralytics import YOLO  # type: ignore

class Trainer:
    def __init__(self, config):
        self.config = config
    
    def train_and_evaluate_model(self, model_name, model, param_grid, X_train, y_train, X_val, y_val):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name.lower(), model)
        ])

        min_class_samples = np.min(np.bincount(y_train))
        cv_folds = min(min_class_samples, 5)
        stratified_kfold = StratifiedKFold(n_splits=cv_folds)

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=stratified_kfold,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val)

        evaluation_results = classification_report(y_val, y_pred_val, output_dict=True)
        conf_matrix = pd.crosstab(y_val, y_pred_val, 
                                  rownames=['Actual'], colnames=['Predicted'])
        return best_model, evaluation_results, conf_matrix

    def ver_matriz_confusion(self, model_name, conf_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def print_results(self, evaluation_results):
        for model_name, results in evaluation_results.items():
            print(f"Model: {model_name}")
            print("Best Parameters:", results['best_model'].get_params())
            print("Evaluation Results:")
            print(pd.DataFrame(results['evaluation_results']).T)
            print("Confusion Matrix:")
            self.ver_matriz_confusion(model_name, results['confusion_matrix'])

    def train_evaluate_models(self, X_train, y_train, X_val, y_val, models=None, param_grid=None):
        if models is None:
            models = self.config.MODELS
        if param_grid is None:
            param_grid = self.config.PARAM_GRID

        evaluation_results = {}
        f1_scores = []

        for model_name, model in models.items():
            print(f"Entrenando y evaluando el modelo {model_name}")
            best_model, results, conf_matrix = self.train_and_evaluate_model(
                model_name, model, param_grid[model_name],
                X_train, y_train, X_val, y_val
            )
            evaluation_results[model_name] = {
                'best_model': best_model,
                'evaluation_results': results,
                'confusion_matrix': conf_matrix
            }
            f1_scores.append({
                'Model': model_name,
                'F1-Score (Weighted)': results['weighted avg']['f1-score'],
                'F1-Score (Macro)': results['macro avg']['f1-score']
            })

        return evaluation_results, f1_scores


    def _get_val_images_and_labels(self, val_folder):
        paths, labels = [], []
        for cls in os.listdir(val_folder):
            d = os.path.join(val_folder, cls)
            if os.path.isdir(d):
                for f in glob.glob(os.path.join(d, '*')):
                    paths.append(f)
                    labels.append(cls)
        return paths, labels

    def investigate_best_yolo(self, models_paths, dataset_path, val_folder, epochs=10, imgsz=1024):
        best_model = None
        best_macro_f1 = -1
        for mp in models_paths:
            model_name = os.path.splitext(os.path.basename(mp))[0]
            model = YOLO(mp)
            model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=imgsz,
                device='cuda',
                augment=False,
                project=self.config.RUNS_FOLDER,
                name=model_name
            )
            imgs, labels = self._get_val_images_and_labels(val_folder)
            preds = []
            for img in imgs:
                r = model.predict(source=img, imgsz=imgsz, device='cuda', verbose=False)
                pred_index = None
                if r and hasattr(r[0], 'pred') and r[0].pred.size > 0:
                    pred_index = int(r[0].pred[0])
                elif r and hasattr(r[0], 'probs'):
                    pred_index = int(r[0].probs.top1)
                if pred_index is not None:
                    names = model.model.names if hasattr(model.model, 'names') else {}
                    preds.append(names.get(pred_index, str(pred_index)))
                else:
                    preds.append("unknown")
            macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_model = mp
        return best_model

    def tune_best_yolo(self, model_path, dataset_path, epochs=10, iterations=30, imgsz=224):
        model = YOLO(model_path)
        model.tune(
            data=dataset_path,
            epochs=epochs,
            iterations=iterations,
            imgsz=imgsz,
            device='cuda',
            optimizer="AdamW",
            plots=True,
            save=True,
            val=True,
            augment=False,
            project=self.config.RUNS_FOLDER,
            name='best_tune',
        )
        
    def train_best_model(self, model_name, dataset_path, epochs=25, hyp=None, imgsz=224):
        model = YOLO(model_name)
        model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=imgsz,
            device='cuda',
            augment=False,
            project=self.config.RUNS_FOLDER,
            name=model_name,
            **hyp
        )
        return model
    

    def train_and_evaluate_model(self, model_name, model, param_grid, X_train, y_train, X_val, y_val):
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (model_name.lower(), model)  
        ])

        min_class_samples = np.min(np.bincount(y_train))
        cv_folds = min(min_class_samples, 5)
        stratified_kfold = StratifiedKFold(n_splits=cv_folds)


        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=stratified_kfold,
            scoring='accuracy'
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred_val = best_model.predict(X_val)

        evaluation_results = classification_report(y_val, y_pred_val, output_dict=True)
        conf_matrix = pd.crosstab(y_val, y_pred_val,
                                rownames=['Actual'], colnames=['Predicted'])

        return best_model, evaluation_results, conf_matrix


    def evaluate_all_scenarios_random_forest(self, all_scenarios, param_grid):
        all_results = {}
        model = RandomForestClassifier(random_state=123) 

        for scenario_name, data_dict in all_scenarios.items():
            print(f"\n=== Entrenando en el escenario: {scenario_name} ===")
            X_train = data_dict["X_train"]
            y_train = data_dict["y_train"]
            X_val   = data_dict["X_val"]
            y_val   = data_dict["y_val"]

            f1_scores = []

            best_model, results, conf_matrix = self.train_and_evaluate_model(
                "randomforest",
                model,
                param_grid,
                X_train,
                y_train,
                X_val,
                y_val
            )
            evaluation_results = {
                'best_model': best_model,
                'evaluation_results': results,
                'confusion_matrix': conf_matrix
            }

            f1_scores.append({
                'Model': 'RandomForest',
                'F1-Score (Weighted)': results['weighted avg']['f1-score'],
                'F1-Score (Macro)': results['macro avg']['f1-score']
            })

            all_results[scenario_name] = {
                "evaluation_results": evaluation_results,
                "f1_scores": pd.DataFrame(f1_scores)
            }

            print("\nResultados en escenario:", scenario_name)
            print("Best Parameters:", best_model.get_params())
            print("\nEvaluation Results (Classification Report):")
            print(pd.DataFrame(results).T)
            print("\nConfusion Matrix:")

            self.ver_matriz_confusion("randomforest", conf_matrix)
            print("\nTabla de F1-Scores:")
            print(all_results[scenario_name]["f1_scores"])
            print("="*50)

        print("\n\n| Scenario | Model | F1-Score (Weighted) | F1-Score (Macro) |")
        print("|----------|-------|----------------------|-------------------|")
        for scenario_name, results in all_results.items():
            for _, row in results["f1_scores"].iterrows():
                print(f"| {scenario_name} | {row['Model']} | {row['F1-Score (Weighted)']} | {row['F1-Score (Macro)']} |")
        print("\n\n")

        return all_results

