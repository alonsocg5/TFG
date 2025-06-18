# Importamos paquetes
import pandas as pd
from sklearn import impute
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, make_scorer, f1_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
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

# Modificación del DataFrame para quedarnos primero con los datos del año 1998
df98 = df.iloc[:, 1:31]
dfaux = df.iloc[:, 92:94] # Columnas GBA98 e ITG98
df98 = pd.concat([df98,dfaux], axis=1)

# Eliminación de las columnas que no son necesarias
df98 = df98.drop(columns=['Tseguim', 'MuerteCV_SN', 'MuerteCANCER_SN',
       'Muerte_otras_SN', 'Fechafallec', 'Motivo','Motivo1', 'Motivo2'])

# Creación de la columna Edad98
df98['Edad98'] = df['FechNac'].apply(lambda año: 1998 - año)

# Imputamos valores en los NaN con KNNImputer
imputer_knn = impute.KNNImputer(n_neighbors=5)
df98[df98.columns[3:24]] = imputer_knn.fit_transform(df98[df98.columns[3:24]])

# Se reordena el DataFrame para que la clase este en la última columna
colum_clase = 2
columnas = df98.columns[:colum_clase].to_list() + df98.columns[(colum_clase+1):].to_list() + df98.columns[colum_clase:(colum_clase+1)].to_list()
df98 = df98[columnas]

# Eliminación de las columnas que no son necesarias
df98 = df98.drop(columns=['FechNac'])

print(df98)

# Separamos los atributos y la clase y los almacenamos en X e y respectivamente
X = df98.iloc[:, :-1]
y = df98.iloc[:, -1]

# Obtenemos el número de ejemplos de cada clase
print("\nNº de ejemplos de cada clase:")
print(y.value_counts())

############################################################################################################

# Se crea un generador de folds estratificados partiendo el conjunto en 5 trozos
folds5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
# se crea un generador de folds estratificados partiendo el conjunto en 10 trozos
folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

# Creamos un 'scorer' para las métricas que queremos obtener
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score, zero_division=0)
}

# Creamos el Árbol de decisión (class_weight ---> None / balanced)
arbol_dec = DecisionTreeClassifier(random_state=1234, class_weight = None)

# Grid para la búsqueda de hiperparámetros
param_grid = {
    'max_depth': range(2, 10),
    'min_impurity_decrease': [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.008]
}

# GridSearchCV con validación cruzada de 5 folds (Optimización --->  scoring='accuracy' / 'f1')
grid_search = GridSearchCV(arbol_dec, param_grid, cv=folds5, scoring='f1', verbose=1, n_jobs=-1)

# Obtenemos resultados con validación cruzada
scores = cross_validate(grid_search, X, y, cv=folds10, scoring=scoring, return_estimator=True, verbose=1)

print("\nResultados:")
print("Accuracy (mean ± std): %0.4f ± %0.4f" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
print("F1-score (mean ± std): %0.4f ± %0.4f" % (scores['test_f1'].mean(), scores['test_f1'].std()))
print("Recall (mean ± std): %0.4f ± %0.4f" % (scores['test_recall'].mean(), scores['test_recall'].std()))
print("Precision (mean ± std): %0.4f ± %0.4f" % (scores['test_precision'].mean(), scores['test_precision'].std()))

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
plt.title("Matriz de confusión")
plt.ylabel("Clase verdadera")
plt.xlabel("Clase predicha")
plt.show()