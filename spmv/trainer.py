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
from ultralytics import YOLO

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
            for k, v in param_grid.items():
                new_param_grid[f"{prefix}__{k}" if '__' not in k else k] = v
            return new_param_grid
        pp = prefix_param_grid(param_grid, model_name.lower())
        mc = np.min(np.bincount(y_train))
        cv_folds = min(mc, 5)
        skf = StratifiedKFold(n_splits=cv_folds)
        gs = GridSearchCV(pipeline, param_grid=pp, cv=skf, scoring='accuracy')
        gs.fit(X_train, y_train)
        bm = gs.best_estimator_
        yp = bm.predict(X_val)
        ev = classification_report(y_val, yp, output_dict=True)
        cm = pd.crosstab(y_val, yp, rownames=['Actual'], colnames=['Predicted'])
        return bm, ev, cm

    def ver_matriz_confusion(self, model_name, conf_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def print_results(self, evaluation_results):
        cdata = []
        for mn, r in evaluation_results.items():
            display(Markdown(f"## Modelo: **{mn}**"))
            bp = r['best_model'].get_params()
            display(Markdown("### Mejores Parámetros:"))
            display(Markdown(f"```python\n{bp}\n```"))
            display(Markdown("### Resultados de Evaluación:"))
            dfe = pd.DataFrame(r['evaluation_results']).T
            display(Markdown(dfe.to_markdown()))
            cdata.append((mn, r['confusion_matrix']))
        for mn, cm in cdata:
            display(Markdown(f"### Matriz de Confusión para {mn}:"))
            self.ver_matriz_confusion(mn, cm)

    def train_evaluate_models(self, X_train, y_train, X_val, y_val, models=None, param_grid=None):
        if models is None:
            models = self.config.MODELS
        if param_grid is None:
            param_grid = self.config.PARAM_GRID
        er = {}
        f1s = []
        for mn, m in models.items():
            bm, rr, cm = self.train_and_evaluate_model(mn, m, param_grid[mn], X_train, y_train, X_val, y_val)
            er[mn] = {'best_model': bm, 'evaluation_results': rr, 'confusion_matrix': cm}
            f1s.append({'Model': mn, 'F1-Score (Weighted)': rr['weighted avg']['f1-score'], 'F1-Score (Macro)': rr['macro avg']['f1-score']})
        return er, f1s

    def investigate_best_yolo(self, models_paths, dataset_path, val_folder, epochs=10, imgsz=1024):
        bm = None
        bf = -1
        for mp in models_paths:
            nm = os.path.splitext(os.path.basename(mp))[0]
            mo = YOLO(mp)
            mo.train(data=dataset_path, epochs=epochs, imgsz=imgsz, device='cuda', augment=False, project=self.config.RUNS_FOLDER, name=nm)
            paths, labels = [], []
            for cls in os.listdir(val_folder):
                d = os.path.join(val_folder, cls)
                if os.path.isdir(d):
                    for f in glob.glob(os.path.join(d, '*')):
                        paths.append(f)
                        labels.append(cls)
            preds = []
            for im in paths:
                r = mo.predict(source=im, imgsz=imgsz, device='cuda', verbose=False)
                pi = None
                if r and hasattr(r[0], 'pred') and r[0].pred.size>0:
                    pi = int(r[0].pred[0])
                elif r and hasattr(r[0], 'probs'):
                    pi = int(r[0].probs.top1)
                if pi is not None:
                    nms = mo.model.names if hasattr(mo.model, 'names') else {}
                    preds.append(nms.get(pi,str(pi)))
                else:
                    preds.append("unknown")
            mf = f1_score(labels, preds, average='macro', zero_division=0)
            if mf>bf:
                bf = mf
                bm = mp
        return bm

    def train_best_model(self, model_name, dataset_path, epochs=25, hyp=None, imgsz=224):
        mo = YOLO(model_name)
        mo.train(data=dataset_path, epochs=epochs, imgsz=imgsz, device='cuda', augment=False, project=self.config.RUNS_FOLDER, name=model_name, **hyp)
        return mo

    def evaluate_all_scenarios_random_forest(self, all_scenarios, param_grid, n=4):
        ar = {}
        m = RandomForestClassifier(random_state=self.config.SEED)
        for scn, dd in all_scenarios.items():
            xt, yt = dd["X_train"], dd["y_train"]
            xv, yv = dd["X_val"], dd["y_val"]
            fs = []
            bm, rs, cm = self.train_and_evaluate_model("randomforest", m, param_grid, xt, yt, xv, yv)
            ev = {'best_model': bm,'evaluation_results': rs,'confusion_matrix': cm}
            fs.append({'Model': 'RandomForest','F1-Score (Weighted)': rs['weighted avg']['f1-score'],'F1-Score (Macro)': rs['macro avg']['f1-score']})
            ar[scn] = {"evaluation_results": ev, "f1_scores": pd.DataFrame(fs)}
        all_rows = []
        for sn, r in ar.items():
            for _, rw in r["f1_scores"].iterrows():
                all_rows.append({"Scenario": sn,"Model": rw['Model'],"F1-Score (Weighted)": rw['F1-Score (Weighted)'],"F1-Score (Macro)": rw['F1-Score (Macro)']})
        srt = sorted(all_rows, key=lambda x: x["F1-Score (Macro)"], reverse=True)
        for r in srt:
            pass
        topN = [row['Scenario'] for row in srt[:n]]
        return ar, topN

    def create_all_scenarios(self, all_scenarios, feats=None):
        if feats is None:
            feats = list(all_scenarios.keys())
        feats = sorted(feats)
        ns = {}
        for size in range(2, len(feats)+1):
            for c in itertools.combinations(feats, size):
                k = "_".join(c)
                if all(f in all_scenarios for f in c):
                    xt = pd.concat([all_scenarios[f]["X_train"].add_prefix(f+"_") for f in c], axis=1)
                    xv = pd.concat([all_scenarios[f]["X_val"].add_prefix(f+"_") for f in c], axis=1)
                    ns[k] = {"X_train": xt,"y_train": all_scenarios[c[0]]["y_train"],"X_val": xv,"y_val": all_scenarios[c[0]]["y_val"]}
        return ns

    def save_ensemble_models(self, best_model, normal_model, best_filename='best_model.pkl', normal_filename='normal_model.pkl'):
        os.makedirs(self.config.PATH_ENSAMBLE, exist_ok=True)
        joblib.dump(best_model, os.path.join(self.config.PATH_ENSAMBLE, best_filename))
        joblib.dump(normal_model, os.path.join(self.config.PATH_ENSAMBLE, normal_filename))

    def load_ensemble_models(self, best_filename='best_model.pkl', normal_filename='normal_model.pkl'):
        best_model = joblib.load(os.path.join(self.config.PATH_ENSAMBLE, best_filename))
        normal_model = joblib.load(os.path.join(self.config.PATH_ENSAMBLE, normal_filename))
        return best_model, normal_model

    def save_ensemble_datasets(self, X, y, X_filename='X_ensemble.csv', y_filename='y_ensemble.csv'):
        os.makedirs(self.config.PATH_ENSAMBLE, exist_ok=True)
        X.to_csv(os.path.join(self.config.PATH_ENSAMBLE, X_filename), index=False)
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y.to_csv(os.path.join(self.config.PATH_ENSAMBLE, y_filename), index=False)
        else:
            pd.DataFrame(y).to_csv(os.path.join(self.config.PATH_ENSAMBLE, y_filename), index=False)

    def load_ensemble_datasets(self, X_filename='X_ensemble.csv', y_filename='y_ensemble.csv'):
        X = pd.read_csv(os.path.join(self.config.PATH_ENSAMBLE, X_filename))
        y = pd.read_csv(os.path.join(self.config.PATH_ENSAMBLE, y_filename))
        if y.shape[1] == 1:
            y = y.iloc[:,0]
        return X, y
    
    def ensemble_predict(self, X, best_model, normal_model, condition=0):
        bp = best_model.predict(X)
        npred = normal_model.predict(X)
        e = []
        for i in range(len(bp)):
            if bp[i] == condition:
                e.append(bp[i])
            else:
                e.append(npred[i])
        return np.array(e)

    def evaluate_ensemble(self, y_true, y_pred):
        mf = f1_score(y_true, y_pred, average='macro')
        wf = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred)
        return mf, wf, cm, cr

    def evaluate_scenarios_test(self, loaded_results, all_scenarios, config, model_trainer):
        tr = {}
        for sn, sd in loaded_results.items():
            if sn not in all_scenarios: continue
            bp = sd["best_params"]
            fp = {k:v for k,v in bp.items() if k.startswith("randomforest__")}
            xt = all_scenarios[sn]["X_train"]
            yt = all_scenarios[sn]["y_train"]
            xv = all_scenarios[sn]["X_val"]
            yv = all_scenarios[sn]["y_val"]
            nf = xt.select_dtypes(include=[np.number]).columns.tolist()
            cf = xt.select_dtypes(exclude=[np.number]).columns.tolist()
            pr = ColumnTransformer([('num', StandardScaler(), nf),('cat', OneHotEncoder(handle_unknown='ignore'), cf)])
            rf = RandomForestClassifier(random_state=config.SEED)
            pp = Pipeline([('preprocessor', pr),('randomforest', rf)])
            pp.set_params(**fp)
            pp.fit(xt, yt)
            ypt = pp.predict(xv)
            crr = classification_report(yv, ypt, output_dict=True)
            cmx = confusion_matrix(yv, ypt)
            tr[sn] = {"report": crr, "confusion": cmx}
            print(f"\n=== Test results for '{sn}' ===")
            print(classification_report(yv, ypt))
            print("Matriz de confusión:\n", cmx)
            model_trainer.ver_matriz_confusion(sn, pd.DataFrame(cmx))
        return tr

    def save_best_scenario(self, best_scenario, filename="best_scenario.txt"):
        with open(os.path.join(self.config.BASE_PATH, filename),"w") as f:
            f.write(best_scenario)

    def load_best_scenario(self, filename="best_scenario.txt"):
        with open(os.path.join(self.config.BASE_PATH, filename),"r") as f:
            return f.read()
