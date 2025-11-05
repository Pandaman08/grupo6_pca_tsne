import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def load_and_preprocess(file_or_none):
    if file_or_none is None:
        data = load_iris(as_frame=True).frame
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        feature_names = list(X.columns)
    else:
        df = pd.read_csv(file_or_none)
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            raise ValueError("El dataset no contiene columnas numéricas.")
        X = numeric_df
        y = None
        feature_names = list(X.columns)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names  
