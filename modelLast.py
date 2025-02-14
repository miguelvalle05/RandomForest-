# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cargar los datos
data = pd.read_csv("datos/zapateria_datos.csv")
print("Información del dataset:")
print(data.info())
print("\nPrimeras filas del dataset:")
print(data.head())

# Visualización de la distribución de la cantidad vendida
plt.figure(figsize=(8, 6))
sns.histplot(data['Cantidad vendida'], kde=True, bins=30)
plt.title('Distribución de la Cantidad Vendida')
plt.xlabel('Cantidad Vendida')
plt.ylabel('Frecuencia')
plt.show()

# Preprocesamiento de las fechas y otras características
data['Fecha de venta'] = pd.to_datetime(data['Fecha de venta'])
data['Año'] = data['Fecha de venta'].dt.year
data['Mes'] = data['Fecha de venta'].dt.month
data['Día'] = data['Fecha de venta'].dt.day
data['Día_semana'] = data['Fecha de venta'].dt.dayofweek

# Visualización de ventas por mes
plt.figure(figsize=(8, 6))
sns.boxplot(x='Mes', y='Cantidad vendida', data=data)
plt.title('Cantidad Vendida por Mes')
plt.xlabel('Mes')
plt.ylabel('Cantidad Vendida')
plt.show()

# Codificación de variables categóricas
label_enc_producto = LabelEncoder()
label_enc_categoria = LabelEncoder()

data['Producto'] = label_enc_producto.fit_transform(data['Producto'])
data['Categoría'] = label_enc_categoria.fit_transform(data['Categoría'])

# Normalización del precio
scaler = StandardScaler()
data[['Precio']] = scaler.fit_transform(data[['Precio']])

print("\nDataset después del preprocesamiento:")
print(data.head())

# Variables independientes (X) y dependientes (y)
X = data[['Producto', 'Precio', 'Categoría', 'Tallas', 'Año', 'Mes', 'Día', 'Día_semana']]
y = data['Cantidad vendida']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo inicial
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predicciones y métricas del modelo inicial
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModelo inicial:")
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)

# Visualización de valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Valores Reales vs Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.show()

# Optimización de hiperparámetros con Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("\nModelo optimizado:")
print("Mejores parámetros:", grid_search.best_params_)
print("Error cuadrático medio (MSE):", mse_optimized)
print("Coeficiente de determinación (R²):", r2_optimized)

# Visualización de la importancia de las características
importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Importancia de las Características')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()

# Predicción de productos vendidos para un nuevo producto hipotético
input_data = {
    'Producto': label_enc_producto.transform(['Zapato deportivo'])[0],
    'Precio': 1000,
    'Categoría': label_enc_categoria.transform(['Hombres'])[0],
    'Tallas': 8,
    'Año': 2025,
    'Mes': 12,
    'Día': 1,
    'Día_semana': 0
}

input_df = pd.DataFrame([input_data])
input_df[['Precio']] = scaler.transform(input_df[['Precio']])

predicted_sales = best_model.predict(input_df)
print(f"\nPredicción de productos vendidos para el producto 'Zapato deportivo' talla 8 en diciembre: {predicted_sales[0]} unidades")