from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
