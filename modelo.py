# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cargar los datos
data = pd.read_csv("zapateria_datos.csv")

# Exploración inicial
print(data.info())
print(data.head())

# Preprocesamiento
data['Fecha de venta'] = pd.to_datetime(data['Fecha de venta'])
data['Año'] = data['Fecha de venta'].dt.year
data['Mes'] = data['Fecha de venta'].dt.month
data['Día'] = data['Fecha de venta'].dt.day
data['Día_semana'] = data['Fecha de venta'].dt.dayofweek

# Codificar variables categóricas
label_enc = LabelEncoder()
data['Producto'] = label_enc.fit_transform(data['Producto'])
data['Categoría'] = label_enc.fit_transform(data['Categoría'])

# Normalizar la columna de precios
scaler = StandardScaler()
data[['Precio']] = scaler.fit_transform(data[['Precio']])

# Variables predictoras y objetivo
X = data[['Producto', 'Precio', 'Categoría', 'Tallas', 'Año', 'Mes', 'Día', 'Día_semana']]
y = data['Cantidad vendida']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo inicial
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Modelo inicial:")
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)

# Optimización de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
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
