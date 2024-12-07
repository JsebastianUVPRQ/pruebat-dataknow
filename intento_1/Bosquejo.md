---
title: "Análisis de Políticas Públicas"
author: "Tu Nombre"
date: "2024-12-05"
bibliography: "referencias.bib"
---


## 1. Caso de Negocio

Se ha requerido estimar los costos de dos equipos esenciales para un proyecto de construcción, con una duración de 36 meses. El cliente (empresa constructora) debe proporcionar los equipos necesarios, y el análisis se centra en la estimación del precio de los equipos __1__ y __2__. Dichos precios dependen directamente del valor de mercado de las materias primas $X$, $Y$, $Z$. El equipo 1 está compuesto en un 20% por la materia prima $X$ y un 80% por la materia prima $Y$. Por otro lado, el equipo 2 está compuesto por iguales proporciones de las materias primas $X$, $Y$ y $Z$.
El objetivo de este estudio es optimizar el monto de inversión que la empresa constructora debe realizar en la adquisición de los equipos en el futuro. Para esto, se usarán técnicas de análisis de series temporales (Teoría: Ver Anexo 1) para predecir los precios de las materias primas $X$, $Y$ y $Z$ en los próximos 36 meses.

## 2. Supuestos

- Se cuenta con datos históricos diarios de precios de las materias primas $X$, $Y$ y $Z$.
- Los datos históricos se recopilan desde junio 1988, noviembre 2006 y enero 2010 para $X$, $Y$ y $Z$, respectivamente.
- Los datos históricos se recopilan hasta abril 2024, diciembre 2023 y agosto 2023 para $X$, $Y$ y $Z$, respectivamente.
- Cada materia prima es susceptible a distinta volatilidad[^1].

## 3. Metodología

1. __Preprocesamiento de los datos__: Los datos históricos fueron analizados y preparados, asegurando que no hubiese valores faltantes ni inconsistencias. Se estandarizó el formato de las fechas (YYYY-MM-DD) y los separadores decimales. (Ver Anexo 2)
2. __Trabajo en la suit de Azure__: Se utilizó la suite de Azure para el análisis de series temporales de la materia prima $X$. Se obtuvo un modelo ARIMA ajustado y luego, en un entorno local, se llevaron a cabo las predicciones para los próximos 36 meses. (Ver Anexo 3)
3. __Trabajo en entorno local__: Para hacer forecasting de las materias primas $Y$ y $Z$, se utilizó el modelo XGBoost, el cual permite trabajar con series temporales no estacionarias y con múltiples variables predictoras. (Ver Anexo 4)
4. __Evaluar resultados__: Con medidas de dispersión como el error cuadrático medio (MSE) y el error absoluto medio (MAE), se evaluaron las predicciones obtenidas.
5. __Consideraciones finales__: Desarrollo de las conclusiones y consideración de las posibles modificaciones futuras en pos de enriquecer la excatitud y precisión de este análisis.

## 4. Análisis

### Preprocesamiento de los datos

En primer lugar, los datasets de las materias primas $X$, $Y$ y $Z$ diferían en el formato de las fechas, orden de las columnas, y separadores del formato csv. En un notebook (para el código completo Ver Anexo 5) se realiza el preprocesamiento necesario para que estén listos para ser trabajados

    ```python
    >>>
    df1 = pd.read_csv('Datos/X.csv')
    df2 = pd.read_csv('Datos/Y.csv', delimiter=';')
    df3 = pd.read_csv('Datos/Z.csv')
    # ...
    df1 = df1.iloc[::-1]
    # resetear el indice de df1
    df1 = df1.reset_index(drop=True)
    # ...
    df2['Date'] = df2['Date'].str.replace('/', '-')
    df2['Price'] = df2['Price'].str.replace(',', '.').astype(float)
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    
    ```

### Entrenamiento en Azure Machine Learning

Azure cloud tiene una serie de herramientas que van desde no-code hasta ejecución de scripts locales[]. En este caso, se utilizó ML Studio, el cual permite el entrenamiento y evaluación con _multiples modelos_  y distingue el mejor a partir de la métrica de evaluación seleccionada por el desarrollador. Se requiere una suscripción de azure, en la cual se creará un workspace[^2] en el que estarán los recursos a usar (almacenamiento, cómputo, etc).

[^2]: <https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources>

La carga de los datasets, y almacenamiento de los datasets, el aprovisionamiento de los recursos de cómputo, monitoreo de trabajos, etc; está en el (Anexo 6). El procedimiento lo desarrollé en la interfaz gráfica de Azure AutoML, donde se selecciona el dataset, se elige la variable objetivo, se selecciona el tipo de predicción y se lanzan los experimentos; pero se genera el código que se ejecuta en el entorno de Azure. A continueciòn presento un extracto.

    ```python

    # Import the required libraries
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import AmlCompute

    # The workspace information from the previous experiment has been pre-filled for you.
    subscription_id = "b0108ced-f6be-4641-917c-8e9e1cf2c8d9"
    resource_group = "modelo_ARIMA"
    workspace_name = "workspace_arima_3011"

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    workspace = ml_client.workspaces.get(name=ml_client.workspace_name)
    print(ml_client.workspace_name, workspace.resource_group, workspace.location, ml_client.connections._subscription_id, sep = '\n')
    # ...
    # Choose a name for your CPU cluster
    cluster_name = "cpu-cluster"

    # Verify that cluster does not exist already
    try:
        cluster = ml_client.compute.get(cluster_name)
        print('Found existing cluster, use it.')
    except Exception:
        compute = AmlCompute(name=cluster_name, size='STANDARD_DS4_V2',
                            max_instances=4)
        cluster = ml_client.compute.begin_create_or_update(compute)
    
    # ...
    from azure.ai.ml import command, Input

    # To test with new training / validation datasets, replace the default dataset id(s)/uri(s) taken from parent run below
    command_str = 'python script.py --training_dataset_uri azureml://locations/eastus/workspaces/036f090e-319e-4836-bac1-a5ad6f8e86a0/data/train-XA/versions/1 --validation_dataset_uri azureml://locations/eastus/workspaces/036f090e-319e-4836-bac1-a5ad6f8e86a0/data/test-XA/versions/1'
    command_job = command(
        code=project_folder,
        command=command_str,
        tags=dict(automl_child_run_id='exp-98-r2-para-x_2'),
        environment='AzureML-ai-ml-automl:7',
        compute='cpu-cluster',
        experiment_name='Default')
    
    returned_job = ml_client.create_or_update(command_job)
    returned_job.studio_url
    ```

Luego de los experimentos, se obtiene el mejor modelo, que en este caso fue Arimax (Teoría: Ver Anexo 7). Las métricas principales se muestran en la figura:
![[metricas_arima.png]]

Se descarga el modelo y se ejecuta en un entorno local para hacer las predicciones.

### Trabajo en entorno local

El entrenamiento para las materias primas $Y$ y $Z$ se realizó en notebooks ejecutados en un entorno local. Se utilizó la clase `XGBRegressor` de la librería `xgboost` para entrenar los modelos y hacer las predicciones, a través del método __recursive forecasting__. A continuación, se muestra un extracto del código utilizado para el entrenamiento y la predicción de los precios de la materia prima $Y$. El código para z es equivalente.

    ```python
    % from sklearn.model_selection import train_test_split
    % from sklearn.metrics import mean_squared_error
    % import xgboost as xgb
    % from sklearn.preprocessing import StandardScaler
    # ...
    df['Date'] = pd.to_datetime(df['Date'])

    # Establecer la columna 'fecha' como índice
    df.set_index('Date', inplace=True)
    # ...
    # Crear características de serie temporal (lag features)
    def create_lag_features(df, lags=5):
        for lag in range(1, lags + 1):
            df[f'lag_{lag}'] = df['Price'].shift(lag)
        df = df.dropna()  # Eliminar filas con valores nulos
        return df

    # Crear características de lag
    df = create_lag_features(df, lags=5)

    # Dividir los datos en conjunto de entrenamiento y prueba
    train_size = int(len(df) * 0.7)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Separar las características (X) y el target (y)
    X_train = train.drop(columns=['Price'])
    y_train = train['Price']
    X_test = test.drop(columns=['Price'])
    y_test = test['Price']
    # ...
    # Create forecaster
    # ==============================================================================
    end_validation = '2023-12-09'

    window_features = RollingFeatures(stats=["coef_variation"], window_sizes=4200)
    forecaster = ForecasterRecursive(
                    regressor       = XGBRegressor(random_state=15926, enable_categorical=True),
                    lags            = 3000,
                    window_features = window_features
                 )

    # Train forecaster
    # ==============================================================================
    forecaster.fit(y=df.loc[:end_validation, 'Price'])
    
    # Calcular el error cuadrático medio (MSE) -> El resultado fue 61.2
    mse = mean_squared_error(y_train, predictions)
    print(f"Error cuadrático medio (MSE): {mse  }")

    # prediccion_anio_n = forecaster.predict(steps=(365*n))


    ```
La mse (error cuadrático medio) obtenida fue de 61.2, lo cual es un valor relativamente alto para los valores registrados como precio en el dataset de $Y$. Si graficamos los históricos, el set de entrenamiento y las predicciones para el set de entrenamiento, obtenemos lo siguiente:
![[y_historico.png]]
![[train_vs_pred_volatilidad.png]]

Observamos que se tienen fluctuaciones abruptas en los precios. Esto explica el valor que obtuvimos para la métrica de evaluación, pues los 'lags' que se usaron para entrenar el modelo no capturan suficientemente la volatilidad de los precios. En la sección de consideraciones finales este será uno de los temas a tratar.

Teniendo los modelos entrenados podemos hacer las predicciones para los próximos 36 meses. Teniendo en cuenta que un dataset tiene datos hasta 2023 mientras que los otros dos tienen datos hasta 2024 (año actual), en dicho modelo se harán las predicciones para 'sus' próximos 48 meses.
En la siguiente tabla se consignan los valores obtenidos para los meses 0 (actualidad), 12, 24 y 36.

'''
89,180  	82,790	    88,200	    79,520
547,330	    614,070	    589,300 	568,930
2165,250	1834,140	2246,750	2483,450

'''

| Materia Prima | Mes 0 | Mes 12 | Mes 24 | Mes 36 |
|---------------|-------|--------|--------|--------|
|      $X$ +- 1.73      | 89.2  | 82.8   | 88.2   | 79.5   |
|      $Y$ +- 12.4      | 547.3 | 614.1  | 589.3  | 568.9  |
|      $Z$ +- 32.49     | 2165.3| 1834.1 | 2246.7 | 2483.5 |

La medida de confianza que se usa para reportar cada predicción es el intervalo de predicción, que se calcula a partir de la __desviación estándar de los residuos__(Ver Anexo 8) y el $mse$ de cada modelo.
Veamos ahora el calculo de los costos de los equipos 1 y 2 para el futuro (tabla: \ref{tab:costos_equipos}).
$$
\text{Costo Equipo 1} = 0.2 \times \text{Precio $X$} + 0.8 \times \text{Precio $Y$}~ \text{Costo Equipo 2} = dfrac{1}{3} (\times \text{Precio $X$} + \times \text{Precio $Y$} + \times \text{Precio $Z$})
$$



|      | Actualidad | Mes 12 | Mes 24 | Mes 36 |
|------|------------|--------|--------|--------|
| Equipo 1 | 455.7 +- (89.2 * 0.0173)+ (547.3 * 0.124) | 507.8 +- | 489.08 +- | 471.05 +- |
| Equipo 2 | 933.92 +- (89.2 * 0.0173)+ (547.3 * 0.124) + (2165.3 * 0.3249) | 843.67 +- | 974.75 +- | 1043.97 +- |
| Total | 1389.62 +- | 1351.5 +- | 1463.83 +- | 1515.0 +- |

## 5. Consideraciones finales
En conclusión, el momento óptimo para la adquisición de los equipos es en 12 meses de desarrollo del proyecto, pues los costos de los equipos 1 y 2 son menores en comparación con los costos actuales y futuros. Sin embargo, es importante tener en cuenta que las predicciones realizadas tienen un margen de error, por lo que se recomienda monitorear los precios de las materias primas y ajustar las estrategias de inversión en consecuencia. Por ejemplo, reentrenar los modelos en el transcurso del futuro. También se podría agregar una medida de la inflación prevista por el banco mundial (u organismos similares) en el pais donde se desarrolla el proyecto; además de una medida de la volatilidad del mercado (Hstórica y esperada) para las materias primas. 



## Conclusión

El análisis de series temporales abarca una amplia variedad de métodos, desde los tradicionales enfoques estadísticos (como ARIMA o GARCH) hasta técnicas más modernas de aprendizaje automático (como redes neuronales y árboles de decisión). La elección del enfoque depende de la naturaleza de los datos, los objetivos del análisis (predicción, comprensión de patrones) y las características específicas de la serie temporal que se está analizando (estacionalidad, volatilidad, tendencia, etc.).


## Anexo 1

El estudio de series temporales es un tema extenso, por lo tanto se tratará solamente modelos de Machine Learning: XGBoost y ARIMAX. XGBoost es un algoritmo de aprendizaje automático basado en árboles de decisión que se utiliza comúnmente para problemas de regresión. Su formalización matemática se basa en la minimización de una función de pérdida, de la siguiente forma:

\[
\mathcal{L} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(h_k)
\]
XGBoost utiliza una aproximación de **segunda orden** (incluyendo derivadas primeras y segundas) para optimizar la función de pérdida. Para cada iteración \( k \), se calcula la mejora \( \gamma_k \) que minimiza la función objetivo:

\[
\gamma_k = \arg\min_{\gamma} \left[ \sum_{i \in I_k} L(y_i, F_{k-1}(x_i) + \gamma) \right] + \Omega(\gamma)
\]
XGBoost construye árboles de decisión de manera aditiva. Cada árbol nuevo se agrega al modelo existente para corregir los errores residuales. La construcción de cada árbol implica:

1. **División de Nodos:** Selección de la mejor característica y punto de división que maximice la ganancia de información.
2. **Asignación de Pesos:** Determinación de los pesos óptimos para cada hoja, minimizando la función objetivo localmente.
3. **Poda:** Eliminación de ramas que no contribuyen significativamente al modelo, según los parámetros de regularización.

Por otro lado, ARIMAX es un modelo de regresión lineal que incorpora términos autorregresivos (AR), de medias móviles (MA) y diferenciación (I) para modelar series temporales. Antes de profundizar en ARIMAX, es esencial comprender los componentes básicos de ARIMA:

- **AR (AutoRegressive):** Captura la relación lineal entre una observación actual y un número de observaciones anteriores.
- **I (Integrated):** Indica el número de veces que la serie debe ser diferenciada para alcanzar la estacionariedad.
- **MA (Moving Average):** Modela el error de predicción como una combinación lineal de errores pasados.
El modelo ARIMAX puede expresarse como:

\[
\Phi(B) (1 - B)^d Y_t = \Theta(B) \epsilon_t + \beta_1 X_{1,t} + \beta_2 X_{2,t} + \dots + \beta_k X_{k,t}
\]

Donde:

- \( B \) es el operador de rezago (\( B Y_t = Y_{t-1} \)).
- \( \Phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \dots - \phi_p B^p \) representa la parte autorregresiva.
- \( \Theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \dots + \theta_q B^q \) representa la parte de media móvil.
- \( \epsilon_t \) es el término de error aleatorio, generalmente asumido como ruido blanco (\( \epsilon_t \sim \mathcal{N}(0, \sigma^2) \)).
- \( \beta_1, \beta_2, \dots, \beta_k \) son los coeficientes asociados a las variables exógenas \( X_{1,t}, X_{2,t}, \dots, X_{k,t} \).

ARIMAX es usado en multiples campos, tales como:

- **Economía:** Modelado de indicadores macroeconómicos influenciados por variables exógenas como tasas de interés.
- **Marketing:** Predicción de ventas basada en campañas publicitarias u otras variables de marketing.
- **Salud Pública:** Pronóstico de enfermedades considerando factores externos como clima o políticas de salud.
- **Ingeniería:** Monitoreo de procesos industriales con influencias externas.

## Anexo 2

## Anexo 8

En estadística, un __intervalo de predicción__ proporciona un rango dentro del cual se espera que caiga una observación futura individual con una cierta probabilidad. A diferencia de los intervalos de confianza, que se utilizan para estimar parámetros poblacionales, los intervalos de predicción están diseñados para predecir valores individuales futuros.

### Definición Formal

Sea \( Y_{\text{nuevo}} \) una nueva observación que se desea predecir. Un intervalo de predicción de nivel \( 1 - \alpha \) para \( Y_{\text{nuevo}} \) está definido como el intervalo \([L, U]\) tal que:

\[
P(L \leq Y_{\text{nuevo}} \leq U) = 1 - \alpha
\]

Donde:

- \( P \) denota la probabilidad.
- \( L \) es el límite inferior del intervalo de predicción.
- \( U \) es el límite superior del intervalo de predicción.
- \( \alpha \) es el nivel de significancia (por ejemplo, \( \alpha = 0.05 \) para un intervalo de predicción del 95%).

### Cálculo del Intervalo de Predicción

En el contexto de un modelo de regresión lineal, el intervalo de predicción para una nueva observación \( Y_{\text{nuevo}} \) dada una nueva variable independiente \( \mathbf{x}_{\text{nuevo}} \) se calcula de la siguiente manera:

\[
\hat{Y}_{\text{nuevo}} \pm t_{\alpha/2, n-p} \cdot s \cdot \sqrt{1 + \mathbf{x}_{\text{nuevo}}^\top (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}_{\text{nuevo}}}
\]

Donde:

- \( \hat{Y}_{\text{nuevo}} \) es la predicción puntual para \( Y_{\text{nuevo}} \).
- \( t_{\alpha/2, n-p} \) es el valor crítico de la distribución \( t \) de Student con \( n - p \) grados de libertad.
- \( s \) es la estimación de la desviación estándar del error.
- \( \mathbf{X} \) es la matriz de diseño de las variables independientes en el modelo de regresión.
- \( n \) es el número de observaciones.
- \( p \) es el número de parámetros en el modelo.

### Interpretación

El intervalo de predicción proporciona un rango donde se espera que caiga una nueva observación futura con una probabilidad especificada. Es especialmente útil en aplicaciones donde se necesita predecir valores individuales más que estimar parámetros poblacionales.

### Diferencias con el Intervalo de Confianza

- **Intervalo de Confianza:** Estima un parámetro poblacional (como la media) y proporciona un rango donde se espera que se encuentre el parámetro con cierta probabilidad.
  
- **Intervalo de Predicción:** Predice un valor individual futuro y proporciona un rango donde se espera que caiga esa observación con cierta probabilidad.

La implementación en python es la siguiente:

    ```python
    # usar el dataframe predictions para calcular el intervalo de confianza
    model = sm.OLS(y_train, X_train).fit()
    prediccion = model.get_prediction(test)
    # crear un dataset que tenga la columna 'Price' con las predicciones
    prediccion = prediccion.summary_frame(alpha=0.05)

    # renombrar la columna mean a 'Price'
    prediccion = prediccion.rename(columns={'mean': 'Price'})
    # renombrar la columna obs_ci_lower a 'lower_bound'
    prediccion = prediccion.rename(columns={'obs_ci_lower': 'lower_bound'})
    # renombrar la columna obs_ci_upper a 'upper_bound'
    prediccion = prediccion.rename(columns={'obs_ci_upper': 'upper_bound'})
    # seleccionar solamente las columnas 'Price', 'lower_bound' y 'upper_bound'
    prediccion = prediccion[['Price', 'lower_bound', 'upper_bound']]


    # calculo para cada fila del dataframe el intervalo de confianza
    prediccion['intervalo'] = prediccion.apply(lambda x: (x['Price'] - x['lower_bound'], x['upper_bound'] - x['Price']), axis=1)
    ```
