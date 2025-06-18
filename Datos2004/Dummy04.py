# Importamos paquetes
import pandas as pd
from sklearn import impute
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

# Modificación del DataFrame para quedarnos con los datos del año 2004
df04 = df.iloc[:, 31:54]
dfaux = df.iloc[:, 94:97] # Columnas GBA04, ITG04 y PrediabHb04
df04 = pd.concat([df04, dfaux], axis=1)

# Creación de la columna Edad04
df04['Edad04'] = df['FechNac'].apply(lambda año: 2004 - año)

# Concatenamos DataFrame del año 98 con el del año 2004
df04 = pd.concat([df98, df04], axis=1)

# Eliminamos aquellas filas que los pacientes hayan fallecido y no tengan datos en el año 2004, 
# utlizamos la columna TAS104 por ejemplo
df04 = df04.dropna(subset=['TAS104'])

# Reindexamos DataFrame
df04 = df04.reset_index(drop=True)



# Imputamos valores en los NaN con KNNImputer
imputer_knn = impute.KNNImputer(n_neighbors=5)
df04[df04.columns] = imputer_knn.fit_transform(df04[df04.columns])

# Comprobamos cuanto valores NaN hay en cada columna antes de imputar valores
n_nan = df04.isna().sum()
print(n_nan)

# Se reordena el DataFrame para que la clase este en la ultima columna
colum_clase = 2
columnas = df04.columns[:colum_clase].to_list() + df04.columns[(colum_clase+1):].to_list() + df04.columns[colum_clase:(colum_clase+1)].to_list()
df04 = df04[columnas]

# Eliminación de las columnas que no son necesarias
df04 = df04.drop(columns=['FechNac'])
df04['Edad04'] = df04['Edad04'].astype(int)

# DataFrame con los años 1998 y 2004 preparado
print(df04)

# Separamos los atributos y la clase y los almacenamos en X e y respectivamente
X = df04.iloc[:, :-1]
y = df04.iloc[:, -1]

# Obtenemos el número de ejemplos de cada clase
print("Nº de ejemplos de cada clase:")
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