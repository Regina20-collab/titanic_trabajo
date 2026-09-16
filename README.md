# Análisis y Preprocesamiento de Datos - Titanic Dataset

Proyecto individual desarrollado para la actividad ED.01.02 "Git, GitHub y reproducibilidad". Consiste en el análisis exploratorio y preprocesamiento del dataset Titanic de Kaggle, siguiendo un flujo de trabajo reproducible mediante entorno virtual y control de versiones con Git.

## Dataset

Titanic - Machine Learning from Disaster (Kaggle): https://www.kaggle.com/c/titanic

El archivo `data/titanic.csv` contiene 891 registros de pasajeros con las siguientes columnas: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

## Estructura del proyecto

```
titanic-project/
├── data/
│   └── titanic.csv          # Dataset original
├── notebooks/
│   └── eda_preprocessing.ipynb   # Análisis exploratorio y preprocesamiento
├── src/
│   └── preprocessing.py     # Funciones reutilizables de limpieza y transformación
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducir el proyecto

1. Clonar el repositorio:
   ```
   git clone <URL_DEL_REPOSITORIO>
   cd titanic-project
   ```

2. Crear y activar un entorno virtual:
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Instalar las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Ejecutar el notebook:
   ```
   jupyter notebook notebooks/eda_preprocessing.ipynb
   ```

## Contenido del análisis

- Inspección inicial del dataset (dimensiones, tipos de datos, valores nulos).
- Análisis exploratorio de variables numéricas y categóricas.
- Tratamiento de valores faltantes en `Age`, `Cabin` y `Embarked`.
- Codificación de variables categóricas (`Sex`, `Embarked`).
- Creación de variables derivadas (tamaño de familia, título extraído del nombre).
- Escalado de variables numéricas.
- Exportación del dataset preprocesado listo para modelado.

## Autor

Regina Zoé García Trejo - IDIA 222, Universidad Politécnica de Querétaro
