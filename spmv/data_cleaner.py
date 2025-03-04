from sklearn.preprocessing import LabelEncoder
from spmv.config import Config

class DataCleaner:
    def __init__(self):
        self.config = Config()

    def clean_data_input(self,data):
        data = data.drop(columns=self.config.columns, errors='ignore')
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

