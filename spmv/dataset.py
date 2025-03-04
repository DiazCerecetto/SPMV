from spmv.config import Config
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from spmv.images import ImageManager
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['path_png']).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error al cargar la imagen en {row['path_png']}: {e}")

        if self.transform:
            image = self.transform(image)

        label = (torch.tensor(self.label2idx[row['ganador']], dtype=torch.long)
                 if 'ganador' in row and row['ganador'] in self.label2idx else -1)
        return image, label
    
class DataCleaner:
    def __init__(self, config):
        self.config = config

    def clean_data_input(self,data):
        data = data.drop(columns=self.config.columns_to_drop, errors='ignore')
        data.drop(columns=data.columns[data.isna().all()].tolist(), inplace=True, errors='ignore')
        data.dropna(inplace=True)
        assert not data.isna().values.any()
        data = data[data['ganador'] != 'pcsr']
        return data

    def encode_tags(self,data):
        data.loc[:, 'ganador_encoded'] = LabelEncoder().fit_transform(data['ganador'])
        return data

    def check_numeric_columns(self,df):
        non_numeric_columns = df.columns[~df.map(lambda x: isinstance(x, (int, float))).all()]
        if non_numeric_columns.empty:
            print("Todos los valores en el DataFrame son numéricos.")
        else:
            print("Las siguientes columnas contienen valores no numéricos:")
            print(non_numeric_columns.tolist())

class DatasetManager:
    
    def __init__(self, config, image_manager):
        self.config = config
        self.data_cleaner = DataCleaner(config)
        self.image_manager = image_manager
        
        
    def create_datasets(self, data, save=True, processed_data_dirname = 'processed_data', verbose = True):
        
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

        self.data_cleaner.check_numeric_columns(X_train)
        self.data_cleaner.check_numeric_columns(X_val)
        self.data_cleaner.check_numeric_columns(X_test)

        y_train = data_train['ganador_encoded']
        y_val   = data_val['ganador_encoded']
        y_test  = data_test['ganador_encoded']

        self.download_images_and_set_data(data_train, verbose=verbose)
        self.download_images_and_set_data(data_val, verbose=verbose)
        self.download_images_and_set_data(data_test, verbose=verbose)

        data_train = self.image_manager.check_images(data_train, 'path_png')
        data_val   = self.image_manager.check_images(data_val, 'path_png')
        data_test  = self.image_manager.check_images(data_test, 'path_png')

        self.image_manager.recortar_y_guardar(data_train, conjunto="Train")
        self.image_manager.recortar_y_guardar(data_val,   conjunto="Val")
        self.image_manager.recortar_y_guardar(data_test,  conjunto="Test")

        print("Todas las imágenes han sido recortadas.")

        if save:
            X_train.to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
            X_val.to_csv(os.path.join(processed_data_dir, 'X_val.csv'), index=False)
            X_test.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)

            y_train.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
            y_val.to_csv(os.path.join(processed_data_dir, 'y_val.csv'), index=False)
            y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)

            data_train.to_csv(os.path.join(processed_data_dir, 'data_train.csv'), index=False)
            data_val.to_csv(os.path.join(processed_data_dir, 'data_val.csv'), index=False)
            data_test.to_csv(os.path.join(processed_data_dir, 'data_test.csv'), index=False)

            print(f"Datasets procesados en: {processed_data_dir}")

        return data_train, data_val, data_test, X_train, X_val, X_test, y_train, y_val, y_test

    def create_dataloaders(self, train_df, val_df, label2idx, transform, batch_size=8):
        train_loader = DataLoader(
            CustomDataset(train_df, label2idx=label2idx, transform=transform),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            CustomDataset(val_df, label2idx=label2idx, transform=transform),
            batch_size=batch_size, shuffle=False, drop_last=False
        )
        return train_loader, val_loader

    def download_images_and_set_data(self, data, verbose=True):
        count = 0
        for i, row in data.iterrows():
            if verbose:
                print(f"Procesando fila {count+1}/{len(data)}")
                count += 1
            self.image_manager.descargar_imagen_png(
                row['group'], 
                row['matrix'], 
                carpeta_png=self.config.PATH_CARPETA_MATRICES_PNG
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

