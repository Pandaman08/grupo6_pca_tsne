import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def load_and_preprocess(file_or_none):
    if file_or_none is None:
        # Cargar dataset de ejemplo (ej. Iris)
        data = load_iris(as_frame=True).frame
        X = data.iloc[:, :-1]  # Todas las columnas menos la última (target)
    else:
        df = pd.read_csv(file_or_none)
        # Asumir que todas las columnas son features (sin target)
        # Si hay columnas no numéricas, se deben eliminar o codificar
        X = df.select_dtypes(include=['number'])
        if X.empty:
            raise ValueError("El dataset no contiene columnas numéricas.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled