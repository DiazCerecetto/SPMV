import itertools
import os
import glob
from IPython.display import display, Markdown
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from itertools import combinations

from ultralytics import YOLO  # type: ignore

class Trainer:
    def __init__(self, config):
        self.config = config
    
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
        
        def prefix_param_grid(param_grid, prefix):
            new_param_grid = {}
            for key, value in param_grid.items():
                if '__' not in key:
                    new_key = f"{prefix}__{key}"
                else:
                    new_key = key
                new_param_grid[new_key] = value
            return new_param_grid
        
        prefixed_param_grid = prefix_param_grid(param_grid, model_name.lower())
        
        min_class_samples = np.min(np.bincount(y_train))
        cv_folds = min(min_class_samples, 5)
        stratified_kfold = StratifiedKFold(n_splits=cv_folds)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=prefixed_param_grid,
            cv=stratified_kfold,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        
        evaluation_results = classification_report(y_val, y_pred_val, output_dict=True)
        conf_matrix = pd.crosstab(y_val, y_pred_val, rownames=['Actual'], colnames=['Predicted'])
        return best_model, evaluation_results, conf_matrix

    def ver_matriz_confusion(self, model_name, conf_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def print_results(self, evaluation_results):
        confusion_data = []

        for model_name, results in evaluation_results.items():
            display(Markdown(f"## Modelo: **{model_name}**"))
            
            best_params = results['best_model'].get_params()
            display(Markdown("### Mejores Parámetros:"))
            display(Markdown(f"```python\n{best_params}\n```"))
            
            display(Markdown("### Resultados de Evaluación:"))
            df_eval = pd.DataFrame(results['evaluation_results']).T
            display(Markdown(df_eval.to_markdown()))
            
            confusion_data.append((model_name, results['confusion_matrix']))

        for model_name, conf_matrix in confusion_data:
            display(Markdown(f"### Matriz de Confusión para {model_name}:"))
            self.ver_matriz_confusion(model_name, conf_matrix)

    def train_evaluate_models(self, X_train, y_train, X_val, y_val, models=None, param_grid=None):
        from IPython.display import display, Markdown
        if models is None:
            models = self.config.MODELS
        if param_grid is None:
            param_grid = self.config.PARAM_GRID

        evaluation_results = {}
        f1_scores = []

        for model_name, model in models.items():
            display(Markdown(f"### Entrenando y evaluando el modelo **{model_name}**"))
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

        display(Markdown("### F1 Scores"))
        df_f1 = pd.DataFrame(f1_scores)
        display(Markdown(df_f1.to_markdown()))
        
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
    
    def evaluate_all_scenarios_random_forest(self, all_scenarios, param_grid, print_parameters=True,
                                             print_report=True, display_confusion_matrix=True, n=4):
        all_results = {}
        model = RandomForestClassifier(random_state=self.config.SEED) 

        for scenario_name, data_dict in all_scenarios.items():
            display(Markdown(f"## Entrenando en el escenario: **{scenario_name}**"))
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

            display(Markdown(f"### Resultados en escenario: **{scenario_name}**"))
            if print_parameters:
                display(Markdown("#### Best Parameters:"))
                display(Markdown(f"```python\n{best_model.get_params()}\n```"))
            if print_report:
                display(Markdown("#### Evaluation Results (Classification Report):"))
                df_results = pd.DataFrame(results).T
                display(Markdown(df_results.to_markdown()))
            if display_confusion_matrix:
                display(Markdown("#### Confusion Matrix:"))
                self.ver_matriz_confusion("randomforest", conf_matrix)

            display(Markdown("#### Tabla de F1-Scores:"))
            display(Markdown(all_results[scenario_name]["f1_scores"].to_markdown()))

            display(Markdown("-" * 50))
    
        display(Markdown("## Resumen de F1-Scores por escenario (ordenado por F1-Score (Macro))"))
        all_rows = []
        for scenario_name, results in all_results.items():
            for _, row in results["f1_scores"].iterrows():
                all_rows.append({
                    "Scenario": scenario_name,
                    "Model": row['Model'],
                    "F1-Score (Weighted)": row['F1-Score (Weighted)'],
                    "F1-Score (Macro)": row['F1-Score (Macro)']
                })

        all_rows_sorted = sorted(all_rows, key=lambda x: x["F1-Score (Macro)"], reverse=True)
        table_md = "| Scenario | Model | F1-Score (Weighted) | F1-Score (Macro) |\n"
        table_md += "|----------|-------|----------------------|-------------------|\n"
        for row in all_rows_sorted:
            table_md += f"| {row['Scenario']} | {row['Model']} | {row['F1-Score (Weighted)']} | {row['F1-Score (Macro)']} |\n"

        display(Markdown(table_md))
        
        topN_scenarios = [row['Scenario'] for row in all_rows_sorted[:n]]
        return all_results, topN_scenarios

    def create_all_scenarios(self, all_scenarios, feature_names_list=None):
        if feature_names_list is None:
            feature_names_list = list(all_scenarios.keys())
        combination_size = len(feature_names_list)
        new_scenarios = {}
        for size in range(2, combination_size + 1):
            for combo in itertools.combinations(feature_names_list, size):
                combo_key = "_".join(combo)
                if all(feat in all_scenarios for feat in combo):
                    X_train_combined = pd.concat(
                        [all_scenarios[feat]["X_train"].add_prefix(f"{feat}_") for feat in combo],
                        axis=1
                    )
                    X_val_combined = pd.concat(
                        [all_scenarios[feat]["X_val"].add_prefix(f"{feat}_") for feat in combo],
                        axis=1
                    )
                    new_scenarios[combo_key] = {
                        "X_train": X_train_combined,
                        "y_train": all_scenarios[combo[0]]["y_train"],
                        "X_val": X_val_combined,
                        "y_val": all_scenarios[combo[0]]["y_val"]
                    }
        return new_scenarios

    def save_ensemble_models(self, best_model, normal_model, best_filename='best_model.pkl', normal_filename='normal_model.pkl'):
        os.makedirs(self.config.PATH_ENSAMBLE, exist_ok=True)
        best_path = os.path.join(self.config.PATH_ENSAMBLE, best_filename)
        normal_path = os.path.join(self.config.PATH_ENSAMBLE, normal_filename)
        import joblib
        joblib.dump(best_model, best_path)
        joblib.dump(normal_model, normal_path)
        print(f"Modelos guardados en:\n{best_path}\n{normal_path}")

    def load_ensemble_models(self, best_filename='best_model.pkl', normal_filename='normal_model.pkl'):
        best_path = os.path.join(self.config.PATH_ENSAMBLE, best_filename)
        normal_path = os.path.join(self.config.PATH_ENSAMBLE, normal_filename)
        import joblib
        best_model = joblib.load(best_path)
        normal_model = joblib.load(normal_path)
        print(f"Modelos cargados desde:\n{best_path}\n{normal_path}")
        return best_model, normal_model

    def save_ensemble_datasets(self, X, y, X_filename='X_ensemble.csv', y_filename='y_ensemble.csv'):
        os.makedirs(self.config.PATH_ENSAMBLE, exist_ok=True)
        X_path = os.path.join(self.config.PATH_ENSAMBLE, X_filename)
        y_path = os.path.join(self.config.PATH_ENSAMBLE, y_filename)
        X.to_csv(X_path, index=False)
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y.to_csv(y_path, index=False)
        else:
            pd.DataFrame(y).to_csv(y_path, index=False)
        print(f"Datasets de ensemble guardados en:\n{X_path}\n{y_path}")

    def load_ensemble_datasets(self, X_filename='X_ensemble.csv', y_filename='y_ensemble.csv'):
        X_path = os.path.join(self.config.PATH_ENSAMBLE, X_filename)
        y_path = os.path.join(self.config.PATH_ENSAMBLE, y_filename)
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        print(f"Datasets de ensemble cargados desde:\n{X_path}\n{y_path}")
        return X, y
    
    def ensemble_predict(self, X, best_model, normal_model, condition=0):
        best_predictions = best_model.predict(X)
        normal_predictions = normal_model.predict(X)
        ensemble_predictions = []
        for i in range(len(best_predictions)):
            if best_predictions[i] == condition:
                ensemble_predictions.append(best_predictions[i])
            else:
                ensemble_predictions.append(normal_predictions[i])
        return np.array(ensemble_predictions)

    def evaluate_ensemble(self, y_true, y_pred):
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        cls_report = classification_report(y_true, y_pred)
        return macro_f1, weighted_f1, cm, cls_report
