---
title: "Análisis de Políticas Públicas"
author: "Tu Nombre"
date: "2024-12-05"
csl: ieee.csl
---


## 1. Explicación del Caso

El objetivo de este análisis es estimar los costos de dos equipos esenciales para un proyecto de construcción con una duración de 36 meses. La empresa constructora debe proporcionar los equipos necesarios, y el análisis se centra en la estimación de los costos de los equipos 1 y 2. Los precios de los equipos dependen de los costos de las materias primas que los componen, cuyas proporciones y precios varían a lo largo del tiempo.

## Equipos a Estimar:

- **Equipo 1**: 20% del precio está compuesto por la materia prima X, y el 80% restante por la materia prima Y.
- **Equipo 2**: El precio está compuesto en partes iguales por las materias primas X, Y y Z.

### Beneficios Esperados:

- Mejora de la planificación financiera del proyecto al proporcionar estimaciones de costos más precisas.
- Optimización de los recursos al seleccionar proveedores con la mejor relación costo-beneficio.

## 2. Supuestos

- Se cuenta con datos históricos diarios de precios de las materias primas X, Y y Z durante los últimos 5000 días.
- Se asume que los precios históricos reflejan los posibles valores futuros de las materias primas.
- Este enfoque ha demostrado ser efectivo en el contexto de la gestión de proyectos [^1].
- El modelo de forecasting utilizado se basa en el análisis de series temporales, considerando la estacionalidad y tendencias de los precios.
- Las predicciones de los precios de las materias primas a futuro serán utilizadas para estimar los costos de los equipos.

## 3. Formas para Resolver el Caso y Opción Tomada

$$
\int_{-\infty}^{\infty} H_n(x) H_m(x) e^{-x^2} \, dx = 0 \quad \text{para } n \neq m.
$$

Para resolver el caso, se utilizó el **modelo ARIMA** para el forecasting de los precios de las materias primas. El proceso se desglosó en los siguientes pasos:

1. **Preprocesamiento de los datos**: Los datos históricos fueron analizados y preparados, asegurando que no hubiese valores faltantes ni inconsistencias. Se verificó la estacionariedad de las series temporales de los precios mediante la prueba de Dickey-Fuller.
   
2. **Selección del Modelo de Forecasting**: Se eligió el modelo **ARIMA** para modelar los precios de las materias primas debido a su capacidad para capturar tendencias y patrones en series temporales. El modelo fue ajustado con los parámetros más adecuados tras realizar pruebas y análisis de autocorrelación.

3. **Pronóstico de Precios Futuros**: Utilizando el modelo ARIMA, se realizaron predicciones para los próximos 12 meses de precios de las materias primas X, Y y Z.

4. **Estimación de Costos**: Los precios pronosticados fueron utilizados para calcular el costo de los dos equipos en base a las proporciones de cada materia prima especificadas.

### Código utilizado:

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


**Modelos de Longo Plazo y Ciclos**

- En algunos casos, las series temporales presentan **ciclos** o tendencias de largo plazo que no son estrictamente estacionales, sino influenciados por factores macroeconómicos o industriales. Los **modelos de ciclos económicos** o **ciclos de mercado** pueden ser útiles para capturar estos patrones.

## Conclusión

El análisis de series temporales abarca una amplia variedad de métodos, desde los tradicionales enfoques estadísticos (como ARIMA o GARCH) hasta técnicas más modernas de aprendizaje automático (como redes neuronales y árboles de decisión). La elección del enfoque depende de la naturaleza de los datos, los objetivos del análisis (predicción, comprensión de patrones) y las características específicas de la serie temporal que se está analizando (estacionalidad, volatilidad, tendencia, etc.).

Si tienes una serie temporal específica en mente o deseas profundizar en algún método en particular, puedo proporcionarte más detalles o ejemplos prácticos sobre cómo implementar estos enfoques.

# Bibliografía
\nocite{*}

[^1]: Pérez, Juan. *Gestión de Proyectos Eficaz*. Editorial Academia, 2019.
