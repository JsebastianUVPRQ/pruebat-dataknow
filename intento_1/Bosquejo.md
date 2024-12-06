---
title: "Análisis de Políticas Públicas"
author: "Tu Nombre"
date: "2024-12-05"
bibliography: "referencias.bib"
---


## 1. Caso de Negocio

Se ha requerido estimar los costos de dos equipos esenciales para un proyecto de construcción, con una duración de 36 meses. El cliente (empresa constructora) debe proporcionar los equipos necesarios, y el análisis se centra en la estimación del precio de los equipos __1__ y __2__. Dichos precios dependen directamente del valor de mercado de las materias primas $X$, $Y$, $Z$. El equipo 1 está compuesto en un 20% por la materia prima $X$ y un 80% por la materia prima $Y$. Por otro lado, el equipo 2 está compuesto por iguales proporciones de las materias primas $X$, $Y$ y $Z$.
El objetivo de este estudio es optimizar el monto de inversión que la empresa constructora debe realizar en la adquisición de los equipos en el futuro. Para esto, se usarán técnicas de análisis de series temporales para predecir los precios de las materias primas $X$, $Y$ y $Z$ en los próximos 36 meses.

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

## 4. Análisis completo

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

El entrenamiento para las materias primas $Y$ y $Z$ se realizó en notebooks ejecutados en un entorno local. Se utilizó la clase `XGBRegressor` de la librería `xgboost' para entrenar los modelos y hacer las predicciones. A continuación, se muestra un extracto del código utilizado para el entrenamiento y la predicción de los precios de la materia prima $Y$. El código para z es equivalente.

    ```python

    ```

$$
\int_{-\infty}
$$

### Código utilizado

```python
# Ejemplo de código de ajuste del modelo ARIMA para la materia prima X
from statsmodels.tsa.arima.model import ARIMA
modelo = ARIMA(precio_x, order=(1,1,1))
modelo_ajustado = modelo.fit()
forecast = modelo_ajustado.forecast(steps=12)
```

## Estudio sobre Políticas Públicas

El análisis de las políticas públicas ha demostrado que las reformas implementadas han tenido efectos significativos en la sociedad [@perez2020].

Además, algunos estudios sugieren que las políticas sociales,
cuando están bien implementadas, tienen un impacto directo sobre la calidad de vida [@gonzalez2019].

__Modelos de Longo Plazo y Ciclos__

- En algunos casos, las series temporales presentan __ciclos__ o tendencias de largo plazo que no son estrictamente estacionales, sino influenciados por factores macroeconómicos o industriales. Los __modelos de ciclos económicos__ o __ciclos de mercado__ pueden ser útiles para capturar estos patrones.

## Conclusión

El análisis de series temporales abarca una amplia variedad de métodos, desde los tradicionales enfoques estadísticos (como ARIMA o GARCH) hasta técnicas más modernas de aprendizaje automático (como redes neuronales y árboles de decisión). La elección del enfoque depende de la naturaleza de los datos, los objetivos del análisis (predicción, comprensión de patrones) y las características específicas de la serie temporal que se está analizando (estacionalidad, volatilidad, tendencia, etc.).

% Bibliografía
% \nocite{*}
