# Importamos paquetes
import pandas as pd
from sklearn import impute
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, make_scorer, f1_score, recall_score, precision_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings 
warnings.filterwarnings ('ignore')

############################################################################################################

# Carga de los datos de los pacientes desde el fichero .csv
df = pd.read_csv('EstudioAsturias completo.csv', sep=';', header=0, decimal=',',na_values=' ')

# Problema con el formato de las fechas 
meses = {
    'ene': 'jan', 'feb': 'feb', 'mar': 'mar', 'abr': 'apr', 'may': 'may', 'jun': 'jun',
    'jul': 'jul', 'ago': 'aug', 'sept': 'sep', 'oct': 'oct', 'nov': 'nov', 'dic': 'dec'
}
df['FechNac'] = df['FechNac'].str.lower().replace(meses, regex=True)
df['FechNac'] = pd.to_datetime(df['FechNac'], format='%d-%b-%y')
df['FechNac'] = df['FechNac'].dt.year

# Problema del efecto 2000
df['FechNac'] = df['FechNac'].apply(lambda x: x - 100)

# Modificación del DataFrame para quedarnos primero con los datos generales
df_gen = df.iloc[:, 1:20]

# Modificación del DataFrame para quedarnos con los datos del año 2004
df04 = df.iloc[:, 31:54]
dfaux = df.iloc[:, 94:97] # Columnas GBA04, ITG04 y PrediabHb04
df04 = pd.concat([df04, dfaux], axis=1)

# Creación de la columna Edad04
df04['Edad04'] = df['FechNac'].apply(lambda año: 2004 - año)

# Concatenamos DataFrame con los datos generales con el del año 2004
df04 = pd.concat([df_gen, df04], axis=1)

# Eliminamos aquellas filas que los pacientes hayan fallecido y no tengan datos en el año 2004, 
# utlizamos la columna TAS104 por ejemplo
df04 = df04.dropna(subset=['TAS104'])

# Motivos de fallecimiento relacionados con la diabetes
motivos = ["Enfermedades endocrinas, nutricionales y metabólicas", 
           "Enfermedades del sistema endocrino, nutrición y metabolismo",
           "Enfermedades del sistema endocrino, nutricionales y metabólicas", 
           "Enfermedades del aparato circulatorio", 
           "Enfermedades del sistema circulatorio",
           "Neoplasia maligna del tejido linfoide, hematopoyético"]

# Filtrado de DataFrame con los motivos de fallecimiento
df04 = df04[(df04["Muerte_SN"] != 1) | (df04["Motivo1"].isin(motivos))]

# Eliminación de las columnas que no son necesarias
df04 = df04.drop(columns=['Tseguim', 'MuerteCV_SN', 'MuerteCANCER_SN',
       'Muerte_otras_SN', 'Fechafallec', 'cattres98', 'Categ498', 'Motivo','Motivo1', 'Motivo2', 'TAS98', 'TAD98', 'Peso98'
])

# Reindexamos DataFrame
df04 = df04.reset_index(drop=True)

# Comprobamos cuanto valores NaN hay en cada columna antes de imputar valores
n_nan = df04.isna().sum()
print(n_nan)

# Imputamos valores en los NaN con KNNImputer
imputer_knn = impute.KNNImputer(n_neighbors=5)
df04[df04.columns] = imputer_knn.fit_transform(df04[df04.columns])

# Se reordena el DataFrame para que la clase este en la ultima columna
colum_clase = 2
columnas = df04.columns[:colum_clase].to_list() + df04.columns[(colum_clase+1):].to_list() + df04.columns[colum_clase:(colum_clase+1)].to_list()
df04 = df04[columnas]

# Eliminación de las columnas que no son necesarias
df04 = df04.drop(columns=['FechNac'])
df04['Edad04'] = df04['Edad04'].astype(int)

# DataFrame con los años 1998 y 2004 preparado
print(df04.columns)

# Separamos los atributos y la clase y los almacenamos en X e y respectivamente
X = df04.iloc[:, :-1]
y = df04.iloc[:, -1]

# Obtenemos el número de ejemplos de cada clase
print("Nº de ejemplos de cada clase:")
print(y.value_counts())

############################################################################################################

# Se crea un generador de folds estratificados partiendo el conjunto en 5 trozos
folds5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
# se crea un generador de folds estratificados partiendo el conjunto en 10 trozos
folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

# Creamos el 'scorer' de FBeta y se le asigna el valor de Beta 
fbeta_scorer = make_scorer(fbeta_score, beta = 1, zero_division=0)

# Creamos un 'scorer' para las métricas que queremos obtener
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'fbeta': fbeta_scorer
}

# Creamos el Árbol de decisión (class_weight ---> None / balanced)
arbol_dec = DecisionTreeClassifier(random_state=1234, class_weight = None)

# Grid para la búsqueda de hiperparámetros
param_grid = {
    'modelo__max_depth': range(2, 10),
    'modelo__min_impurity_decrease': [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.008]
}

# Creamos un pipeline con los métodos de OverSampling, UnderSampling y el sistema que queramos
pipeline = Pipeline([
    ("oversampling", SMOTE(sampling_strategy = 0.5, random_state=42)),
    ("undersampling", RandomUnderSampler(sampling_strategy = 0.8, random_state=42)),
    ("modelo", arbol_dec)
])

# GridSearchCV con validación cruzada de 5 folds (Optimización --->  Sobre F1 o FBeta-Score)
grid_search = GridSearchCV(pipeline, param_grid, cv=folds5, scoring = 'f1', verbose=1, n_jobs=-1)

# Obtenemos resultados con validación cruzada
scores = cross_validate(grid_search, X, y, cv=folds10, scoring=scoring, return_estimator=True, verbose=1)

print("\nResultados:")
print("Accuracy (mean ± std): %0.4f ± %0.4f" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
print("F1-score (mean ± std): %0.4f ± %0.4f" % (scores['test_f1'].mean(), scores['test_f1'].std()))
print("Recall (mean ± std): %0.4f ± %0.4f" % (scores['test_recall'].mean(), scores['test_recall'].std()))
print("Precision (mean ± std): %0.4f ± %0.4f" % (scores['test_precision'].mean(), scores['test_precision'].std()))
print("F-Beta Score (mean ± std): %0.4f ± %0.4f" % (scores['test_fbeta'].mean(), scores['test_fbeta'].std()))

# Obtenemos el mejor modelo
best_index = np.argmax(scores['test_f1'])  # Índice del mejor modelo según F1
best_model = scores['estimator'][best_index].best_estimator_  # Acceder al mejor modelo
print("\nMejores hiperparámetros:", best_model.get_params(),"\n")

# Obtenemos las predicciones durante la validación cruzada
y_pred = cross_val_predict(grid_search, X, y, cv=folds10, verbose=0)

# Obtenemos la matriz de confusión
cm = metrics.confusion_matrix(y, y_pred, labels=[0.0, 1.0])
df_cm = pd.DataFrame(cm, index=[0.0, 1.0], columns=[0.0, 1.0])
sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='d', cbar=False)
plt.title("Matriz de confusión 2004 - Arbol de decisión - SVMSMOTE")
plt.tight_layout()
plt.ylabel("Clase verdadera")
plt.xlabel("Clase predicha")
plt.show()