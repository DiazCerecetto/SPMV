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

class DataLoaderFactory:
    def create_dataloaders(train_df, val_df, label2idx, transform, batch_size=8):
        train_loader = DataLoader(
            CustomDataset(train_df, label2idx=label2idx, transform=transform),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            CustomDataset(val_df, label2idx=label2idx, transform=transform),
            batch_size=batch_size, shuffle=False, drop_last=False
        )
        return train_loader, val_loader
