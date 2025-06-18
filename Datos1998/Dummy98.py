# Importamos paquetes
import pandas as pd
from sklearn import impute
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, make_scorer, f1_score, recall_score, precision_score
from sklearn.dummy import DummyClassifier
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

# Distribución de personas por año de nacimiento (FechNac)
fig, ax = plt.subplots()
ax.set_title('Distribución de personas por año de nacimiento')
ax.set_xlabel('Año de nacimiento')
ax.set_ylabel('Frecuencia')
ax.set_yticks(range(0, 40, 2))
ax.set_xticks(range(1920, 1971, 1))
plt.xticks(rotation=75, fontsize=9)
fig.set_size_inches((12,8))
ax.hist(df98['FechNac'], bins=range(1920, 1971, 1), color='skyblue', edgecolor='black',  width=0.7)
plt.savefig('Distribución_personas_año.jpg', format='jpg')
plt.show()

# Imputamos valores en los NaN con KNNImputer
imputer_knn = impute.KNNImputer(n_neighbors=5)
df98[df98.columns[3:24]] = imputer_knn.fit_transform(df98[df98.columns[3:24]])

# Comprobamos cuanto valores NaN hay en cada columna antes de imputar valores
n_nan = df98.isna().sum()
print(n_nan)

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

# Se crea un generador de folds estratificados partiendo el conjunto en 10 trozos
folds10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

# Creamos un 'scorer' para las métricas que queremos obtener
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score, zero_division=0)
}

# Creamos el Dummy Classifier
dummy = DummyClassifier(strategy="most_frequent", random_state=1234)

# Obtenemos resultados con validación cruzada
scores = cross_validate(dummy, X, y, cv=folds10, scoring=scoring, verbose=1)

print("\nResultados:")
print("Accuracy (mean ± std): %0.4f ± %0.4f" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
print("F1-score (mean ± std): %0.4f ± %0.4f" % (scores['test_f1'].mean(), scores['test_f1'].std()))
print("Recall (mean ± std): %0.4f ± %0.4f" % (scores['test_recall'].mean(), scores['test_recall'].std()))
print("Precision (mean ± std): %0.4f ± %0.4f" % (scores['test_precision'].mean(), scores['test_precision'].std()))