"""
Funciones de preprocesamiento para el dataset Titanic.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset Titanic desde un archivo CSV."""
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores faltantes en Age, Embarked y crea indicador de Cabin conocida."""
    df = df.copy()

    # Age: se imputa con la mediana por Pclass y Sex
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    # Embarked: se imputa con la moda (valor más frecuente)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Cabin: se transforma en indicador binario de si el dato es conocido
    df["Cabin_known"] = df["Cabin"].notna().astype(int)
    df = df.drop(columns=["Cabin"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas útiles para el análisis."""
    df = df.copy()

    # Tamaño de familia a bordo
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Título extraído del nombre
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Other")

    df = df.drop(columns=["Name", "Ticket"])
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica las variables categóricas para su uso en modelos."""
    df = df.copy()

    le_sex = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])

    df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

    return df


def scale_numeric(df: pd.DataFrame, columns=("Age", "Fare", "FamilySize")) -> pd.DataFrame:
    """Escala variables numéricas continuas con StandardScaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[list(columns)] = scaler.fit_transform(df[list(columns)])
    return df


def preprocess_pipeline(path: str) -> pd.DataFrame:
    """Ejecuta el pipeline completo de preprocesamiento."""
    df = load_data(path)
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = encode_categorical(df)
    df = scale_numeric(df)
    return df


if __name__ == "__main__":
    processed = preprocess_pipeline("data/titanic.csv")
    processed.to_csv("data/titanic_processed.csv", index=False)
    print(f"Dataset preprocesado: {processed.shape[0]} filas, {processed.shape[1]} columnas")
