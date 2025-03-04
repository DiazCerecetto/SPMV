from config import Config
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_cleaner import DataCleaner
from image_manager import ImageManager


class DatasetManager:
    
    def __init__(self):
        self.config = Config()
        
    def create_datasets(self,data, save=True, processed_data_dirname = 'processed_data', verbose = True):
        
        processed_data_dir = os.path.join(self.config.BASE_PATH, processed_data_dirname)
        os.makedirs(processed_data_dir, exist_ok=True)

        data_train, data_temp, y_train, y_temp = train_test_split(
            data, data['ganador_encoded'], test_size=0.2, random_state=self.config.SEED
        )
        data_val, data_test, y_val, y_test = train_test_split(
            data_temp, y_temp, test_size=0.5, random_state=self.config.SEED
        )

        X_train = data_train.drop(columns=['ganador', 'ganador_encoded', 'group','matrix'], errors='ignore')
        X_val   = data_val.drop(columns=['ganador', 'ganador_encoded', 'group','matrix'], errors='ignore')
        X_test  = data_test.drop(columns=['ganador', 'ganador_encoded', 'group','matrix'], errors='ignore')

        DataCleaner.check_numeric_columns(X_train)
        DataCleaner.check_numeric_columns(X_val)
        DataCleaner.check_numeric_columns(X_test)

        y_train = data_train['ganador_encoded']
        y_val   = data_val['ganador_encoded']
        y_test  = data_test['ganador_encoded']

        DatasetManager.download_images_and_set_data(data_train, verbose=verbose)
        DatasetManager.download_images_and_set_data(data_val, verbose=verbose)
        DatasetManager.download_images_and_set_data(data_test, verbose=verbose)

        data_train = ImageManager.check_images(data_train, 'path_png')
        data_val   = ImageManager.check_images(data_val, 'path_png')
        data_test  = ImageManager.check_images(data_test, 'path_png')

        ImageManager.recortar_y_guardar(data_train, conjunto="Train")
        ImageManager.recortar_y_guardar(data_val,   conjunto="Val")
        ImageManager.recortar_y_guardar(data_test,  conjunto="Test")

        print("Todas las imágenes han sido recortadas.")

        if save:
            X_train.to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
            X_val.to_csv(os.path.join(processed_data_dir, 'X_val.csv'), index=False)
            X_test.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)

            y_train.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
            y_val.to_csv(os.path.join(processed_data_dir, 'y_val.csv'), index=False)
            y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)

            data_train.to_csv(os.path.join(self.config.BASE_PATH, 'data_train.csv'), index=False)
            data_val.to_csv(os.path.join(self.config.BASE_PATH, 'data_val.csv'), index=False)
            data_test.to_csv(os.path.join(self.config.BASE_PATH, 'data_test.csv'), index=False)

            print(f"Datasets procesados en: {processed_data_dir}")

        return data_train, data_val, data_test, X_train, X_val, X_test, y_train, y_val, y_test

    
    def download_images_and_set_data(self, data, verbose=True):

        for i, row in data.iterrows():
            if verbose:
                print(f"Procesando fila {i+1}/{len(data)}")
            ImageManager.descargar_imagen_png(
                row['group'], row['matrix'], carpeta_png=self.config.PATH_CARPETA_MATRICES_PNG
            )
            archivo = os.path.join(self.config.PATH_CARPETA_MATRICES_PNG, f"{row['matrix']}.png")
            data.at[i, 'path_png'] = archivo

    
    def leer_datasets(self, path_datasets = None):
        if path_datasets is None:
            path_datasets = self.config.PROCESSED_DATA_DIR
        X_train = pd.read_csv(os.path.join(path_datasets, 'X_train.csv'))
        X_val   = pd.read_csv(os.path.join(path_datasets, 'X_val.csv'))
        X_test  = pd.read_csv(os.path.join(path_datasets, 'X_test.csv'))

        y_train = pd.read_csv(os.path.join(path_datasets, 'y_train.csv'))
        y_val   = pd.read_csv(os.path.join(path_datasets, 'y_val.csv'))
        y_test  = pd.read_csv(os.path.join(path_datasets, 'y_test.csv'))

        y_train = y_train.values.ravel()
        y_val   = y_val.values.ravel()
        y_test  = y_test.values.ravel()

        return X_train, X_val, X_test, y_train, y_val, y_test

    
    def leer_datasets_raiz(self, path_datasets = None):
        if path_datasets is None:
            path_datasets = self.config.PROCESSED_DATA_DIR
        data_train = pd.read_csv(os.path.join(path_datasets, 'data_train.csv'))
        data_val   = pd.read_csv(os.path.join(path_datasets, 'data_val.csv'))
        data_test  = pd.read_csv(os.path.join(path_datasets, 'data_test.csv'))
        return data_train, data_val, data_test

