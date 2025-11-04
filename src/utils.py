# utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def load_and_preprocess(file_or_none):
    if file_or_none is None:
        data = load_iris(as_frame=True).frame
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]  # Agregar labels
    else:
        df = pd.read_csv(file_or_none)
        X = df.select_dtypes(include=['number'])
        if X.empty:
            raise ValueError("El dataset no contiene columnas numéricas.")
        y = None  # Si no hay target en CSV
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y  # ✅ Devolver ambos