Convertir un código secuencial a un enfoque de Programación Orientada a Objetos (POO) puede mejorar la organización, reutilización y mantenibilidad del código. A continuación, se muestra cómo transformar el código secuencial proporcionado en un diseño orientado a objetos y cómo implementar pruebas para asegurar su correcto funcionamiento.

## 1. **Diseño de Clases para el Enfoque OOP**

Para estructurar el código en POO, podemos identificar las responsabilidades principales y asignarlas a diferentes clases. A continuación, se presentan algunas clases clave que podríamos definir:

1. **`DataPreprocessor`**: Encargada de la preparación y transformación de los datos.
2. **`ModelTrainer`**: Responsable de la creación, entrenamiento y evaluación del modelo XGBoost.
3. **`ModelEvaluator`**: Opcionalmente, para manejar la evaluación del modelo de manera separada.
4. **`Pipeline`**: Coordina el flujo completo de procesamiento de datos, entrenamiento y evaluación.

### **1.1. Clase `DataPreprocessor`**

Esta clase manejará la creación de características de lag, la división de los datos en conjuntos de entrenamiento y prueba, y la estandarización de las características.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, lags=5, train_size=0.7):
        self.lags = lags
        self.train_size = train_size
        self.scaler = StandardScaler()

    def create_lag_features(self, df):
        df = df.copy()
        for lag in range(1, self.lags + 1):
            df[f'lag_{lag}'] = df['Price'].shift(lag)
        df = df.dropna().reset_index(drop=True)
        return df

    def split_data(self, df):
        train_size = int(len(df) * self.train_size)
        train, test = df.iloc[:train_size], df.iloc[train_size:]
        X_train = train.drop(columns=['Price'])
        y_train = train['Price']
        X_test = test.drop(columns=['Price'])
        y_test = test['Price']
        return X_train, y_train, X_test, y_test

    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
```

### **1.2. Clase `ModelTrainer`**

Esta clase se encargará de inicializar, entrenar y realizar predicciones con el modelo XGBoost.

```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd

class ModelTrainer:
    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 3000,
                'max_depth': 4,
                'learning_rate': 0.001,
                'colsample_bytree': 0.3,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        self.params = params
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return mse

    def get_model(self):
        return self.model
```

### **1.3. Clase `Pipeline`**

Esta clase integrará las clases anteriores para ejecutar el flujo completo de procesamiento, entrenamiento y evaluación.

```python
class Pipeline:
    def __init__(self, lags=5, train_size=0.7, model_params=None):
        self.preprocessor = DataPreprocessor(lags=lags, train_size=train_size)
        self.trainer = ModelTrainer(params=model_params)

    def run(self, df):
        # Preprocesar datos
        df_preprocessed = self.preprocessor.create_lag_features(df)
        X_train, y_train, X_test, y_test = self.preprocessor.split_data(df_preprocessed)
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # Entrenar modelo
        self.trainer.train(X_train_scaled, y_train)
        
        # Hacer predicciones
        predictions = self.trainer.predict(X_train_scaled)
        futuros = pd.DataFrame(predictions, index=train.index, columns=['Prediction'])
        
        # Evaluar modelo
        mse = self.trainer.evaluate(y_train, predictions)
        print(f"Error cuadrático medio (MSE): {mse}")
        
        return {
            'model': self.trainer.get_model(),
            'mse': mse,
            'predictions': futuros,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test
        }
```

## 2. **Implementación del Código en OOP**

A continuación, se muestra cómo utilizar las clases diseñadas para ejecutar el flujo completo.

```python
import pandas as pd
from sklearn.datasets import make_regression  # Solo para ejemplo

# Supongamos que 'df' es nuestro DataFrame con la columna 'Price'
# Aquí generamos un DataFrame de ejemplo
X, y = make_regression(n_samples=1000, n_features=1, noise=0.1)
df = pd.DataFrame({'Price': y})

# Inicializar y ejecutar el pipeline
pipeline = Pipeline(lags=10, train_size=0.7)
result = pipeline.run(df)

# Acceder a los resultados
model = result['model']
mse = result['mse']
predictions = result['predictions']
```

## 3. **Pruebas del Código OOP**

Para asegurar que nuestras clases funcionan correctamente, podemos implementar pruebas unitarias utilizando el módulo `unittest` de Python o una librería como `pytest`. A continuación, se presenta un ejemplo utilizando `unittest`.

### **3.1. Instalación de Dependencias para Testing**

Asegúrate de tener instaladas las librerías necesarias para las pruebas:

```bash
pip install unittest2
```

### **3.2. Escritura de Pruebas Unitarias**

Crearemos un archivo llamado `test_pipeline.py` que contendrá nuestras pruebas.

```python
import unittest
import pandas as pd
from sklearn.datasets import make_regression
from pipeline import DataPreprocessor, ModelTrainer, Pipeline  # Asegúrate de que las clases estén en 'pipeline.py'

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        self.df = pd.DataFrame({'Price': y})
        self.preprocessor = DataPreprocessor(lags=5, train_size=0.7)

    def test_create_lag_features(self):
        df_lag = self.preprocessor.create_lag_features(self.df)
        expected_columns = ['Price'] + [f'lag_{i}' for i in range(1, 6)]
        self.assertListEqual(list(df_lag.columns), expected_columns)
        self.assertEqual(len(df_lag), 100 - 5)

    def test_split_data(self):
        df_lag = self.preprocessor.create_lag_features(self.df)
        X_train, y_train, X_test, y_test = self.preprocessor.split_data(df_lag)
        self.assertEqual(len(X_train), int(95 * 0.7))
        self.assertEqual(len(X_test), 95 - int(95 * 0.7))
        self.assertEqual(X_train.shape[1], 5)  # 5 lag features

    def test_scale_features(self):
        df_lag = self.preprocessor.create_lag_features(self.df)
        X_train, y_train, X_test, y_test = self.preprocessor.split_data(df_lag)
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)
        # Verificar que la media de X_train_scaled es aproximadamente 0
        self.assertAlmostEqual(X_train_scaled.mean(), 0, places=1)

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
        self.X_train = X
        self.y_train = y
        self.trainer = ModelTrainer()

    def test_train_and_predict(self):
        self.trainer.train(self.X_train, self.y_train)
        predictions = self.trainer.predict(self.X_train)
        self.assertEqual(len(predictions), len(self.y_train))
        # Verificar que las predicciones sean números
        self.assertTrue(all(isinstance(p, float) for p in predictions))

    def test_evaluate(self):
        self.trainer.train(self.X_train, self.y_train)
        predictions = self.trainer.predict(self.X_train)
        mse = self.trainer.evaluate(self.y_train, predictions)
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0)

class TestPipeline(unittest.TestCase):
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        self.df = pd.DataFrame({'Price': y})
        self.pipeline = Pipeline(lags=5, train_size=0.7)

    def test_run_pipeline(self):
        result = self.pipeline.run(self.df)
        self.assertIn('model', result)
        self.assertIn('mse', result)
        self.assertIn('predictions', result)
        self.assertIn('X_test_scaled', result)
        self.assertIn('y_test', result)
        self.assertIsInstance(result['model'], xgb.XGBRegressor)
        self.assertIsInstance(result['mse'], float)
        self.assertIsInstance(result['predictions'], pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
```

### **3.3. Ejecutar las Pruebas**

Para ejecutar las pruebas, utiliza el siguiente comando en la terminal:

```bash
python test_pipeline.py
```

Si todas las pruebas pasan, verás una salida similar a:

```
......
----------------------------------------------------------------------
Ran 6 tests in 0.123s

OK
```

## 4. **Consideraciones Adicionales para el Testing**

### **4.1. Uso de Mocks**

En situaciones donde ciertas partes del código interactúan con sistemas externos o realizan operaciones costosas, es útil utilizar **mocks** para simular estas interacciones durante las pruebas. Por ejemplo, podríamos simular la función `fit` del modelo para evitar entrenar realmente el modelo durante las pruebas.

```python
from unittest.mock import patch

class TestModelTrainerWithMock(unittest.TestCase):
    @patch('xgboost.XGBRegressor.fit')
    def test_train_mock(self, mock_fit):
        X, y = make_regression(n_samples=10, n_features=5, noise=0.1)
        trainer = ModelTrainer()
        trainer.train(X, y)
        mock_fit.assert_called_once_with(X, y)
```

### **4.2. Pruebas de Integración**

Además de las pruebas unitarias, es recomendable implementar **pruebas de integración** que verifiquen que las diferentes partes del sistema funcionan correctamente juntas. El `TestPipeline` presentado anteriormente es un ejemplo de prueba de integración básica.

### **4.3. Cobertura de Código**

Utiliza herramientas como `coverage.py` para medir la **cobertura de tus pruebas**, asegurándote de que la mayor parte posible del código está siendo probado.

```bash
pip install coverage
coverage run -m unittest discover
coverage report
```

## 5. **Resumen**

Al convertir el código secuencial a un enfoque orientado a objetos, logramos una mejor organización y modularidad, facilitando la reutilización y el mantenimiento. Además, al implementar pruebas unitarias y de integración, garantizamos la robustez y fiabilidad del sistema. Este diseño facilita futuras expansiones, como la incorporación de diferentes modelos de machine learning o nuevos métodos de preprocesamiento de datos, sin afectar significativamente el resto del sistema.

## 6. **Código Completo en OOP**

Para referencia, a continuación se presenta el código completo integrado en un solo archivo. Se recomienda estructurar el código en módulos separados para una mejor organización en proyectos más grandes.

```python
# pipeline.py
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class DataPreprocessor:
    def __init__(self, lags=5, train_size=0.7):
        self.lags = lags
        self.train_size = train_size
        self.scaler = StandardScaler()

    def create_lag_features(self, df):
        df = df.copy()
        for lag in range(1, self.lags + 1):
            df[f'lag_{lag}'] = df['Price'].shift(lag)
        df = df.dropna().reset_index(drop=True)
        return df

    def split_data(self, df):
        train_size = int(len(df) * self.train_size)
        train, test = df.iloc[:train_size], df.iloc[train_size:]
        X_train = train.drop(columns=['Price'])
        y_train = train['Price']
        X_test = test.drop(columns=['Price'])
        y_test = test['Price']
        return X_train, y_train, X_test, y_test

    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

class ModelTrainer:
    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 3000,
                'max_depth': 4,
                'learning_rate': 0.001,
                'colsample_bytree': 0.3,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        self.params = params
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return mse

    def get_model(self):
        return self.model

class Pipeline:
    def __init__(self, lags=5, train_size=0.7, model_params=None):
        self.preprocessor = DataPreprocessor(lags=lags, train_size=train_size)
        self.trainer = ModelTrainer(params=model_params)

    def run(self, df):
        # Preprocesar datos
        df_preprocessed = self.preprocessor.create_lag_features(df)
        X_train, y_train, X_test, y_test = self.preprocessor.split_data(df_preprocessed)
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # Entrenar modelo
        self.trainer.train(X_train_scaled, y_train)
        
        # Hacer predicciones
        predictions = self.trainer.predict(X_train_scaled)
        futuros = pd.DataFrame(predictions, index=X_train.index, columns=['Prediction'])
        
        # Evaluar modelo
        mse = self.trainer.evaluate(y_train, predictions)
        print(f"Error cuadrático medio (MSE): {mse}")
        
        return {
            'model': self.trainer.get_model(),
            'mse': mse,
            'predictions': futuros,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test
        }

# Uso del Pipeline
if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # Generar datos de ejemplo
    X, y = make_regression(n_samples=1000, n_features=1, noise=0.1)
    df = pd.DataFrame({'Price': y})

    # Inicializar y ejecutar el pipeline
    pipeline = Pipeline(lags=10, train_size=0.7)
    result = pipeline.run(df)

    # Acceder a los resultados
    model = result['model']
    mse = result['mse']
    predictions = result['predictions']
```

Este diseño modular facilita la extensión y el mantenimiento del código, permitiendo agregar nuevas funcionalidades con mínima modificación al código existente.