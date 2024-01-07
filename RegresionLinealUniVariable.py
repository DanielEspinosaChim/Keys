#importamos las librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#cargar el dataSet
df = pd.read_csv('example_semicolon.csv')
print(df.head(10))
#pasarlo a dataFrame para manipular mejor los datos
df=pd.DataFrame(df)
 #cargar las variables de entrada, Features o variables explicativas
X = df[['Seguidores']].values
#cargar las variables objetivo 
y = df[['Interacciones']].values
#ver los datos
plt.plot(X,y,"r.")
plt.show()
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
#entrena el modelo 
modelo.fit(X_train, y_train)

#datos con name
plt.plot(df['Seguidores'],df['Interacciones'],"r.")
plt.xlabel("Equipos afectados ")
plt.ylabel("Coste de los equipos dañados ")
plt.show()
# Calcular el MSE para el conjunto de entrenamiento
y_train_pred = modelo.predict(X_train)
train_error = mean_squared_error(y_train, y_train_pred)
print(f"Error cuadrático medio en el conjunto de entrenamiento: {train_error}")

#Realizar predicciones con el conjunto de prueba
y_pred = modelo.predict(X_test)
# Calcular el MSE para el conjunto de prueba (ya lo tienes en tu código)
error = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio en el conjunto de prueba: {error}")

#predice un valor
valor_a_predecir = np.array([[2500]])

# Realizar la predicción
valor_predicho = modelo.predict(valor_a_predecir)
print(f"El valor aproximado del valor con las características proporcionadas es: {valor_predicho[0]}")
# Predicción para el valor mínimo y máximo del conjunto de datos de entrenamiento para representar la función hipótesis que se ha generado
x_min_max = np.array([[df['Seguidores'].min()], [df['Seguidores'].max()]])
y_train_pred = modelo.predict(x_min_max)

# Representación gráfica de la función hipótesis generada
plt.plot(x_min_max, y_train_pred, "b-")
plt.plot(df['Seguidores'], df['Interacciones'], "g.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()
plt.plot(df['Seguidores'],df['Interacciones'], "b.")
plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(valor_a_predecir, valor_predicho, "rx")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()