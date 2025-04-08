import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TimeMetricsEvaluator:
    def __init__(self, config):
        self.config = config
        self.time_fields = {
            0: 'time',
            1: 'time.1',
            2: 'time.2',
            3: 'time.3',
            4: 'time.4'
        }
    
    def evaluate_majority_class(self, y_true, data_original):
        """Evaluate metrics when always predicting the most common class"""
        majority_class = y_true.mode()[0]
        y_pred = pd.Series([majority_class] * len(y_true))
        return self._calculate_time_metrics(y_true, y_pred, data_original)
    
    def evaluate_random_forest(self, X_train, y_train, X_test, y_test, data_original):
        """Evaluate metrics using Random Forest with default parameters"""
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        rf = RandomForestClassifier(random_state=self.config.SEED)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', rf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        return self._calculate_time_metrics(y_test, y_pred, data_original)
    
    def evaluate_hog_random_forest(self, X_train, y_train, X_test, y_test, data_original, hog_columns):
        """Evaluate metrics using Random Forest with only HOG features"""
        # Filter only HOG features
        X_train_hog = X_train[hog_columns]
        X_test_hog = X_test[hog_columns]
        
        rf = RandomForestClassifier(random_state=self.config.SEED)
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train_hog)
        X_test_scaled = scaler.transform(X_test_hog)
        
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
        return self._calculate_time_metrics(y_test, y_pred, data_original)
    
    def _calculate_time_metrics(self, y_true, y_pred, data_original):
        """Calculate both standard and time-weighted metrics"""
        # Standard metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Time-based metrics
        time_penalty = 0
        total_samples = len(y_true)
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                # Get execution times for both methods
                true_time = data_original[self.time_fields[true_label]].mean()
                pred_time = data_original[self.time_fields[pred_label]].mean()
                # Add the time difference as penalty
                time_penalty += abs(pred_time - true_time)
        
        avg_time_penalty = time_penalty / total_samples
        time_weighted_accuracy = accuracy * (1 / (1 + avg_time_penalty))
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'avg_time_penalty': avg_time_penalty,
            'time_weighted_accuracy': time_weighted_accuracy
        }
