---
title: "Análisis de Políticas Públicas"
author: "Tu Nombre"
date: "2024-12-05"
bibliography: "referencias.bib"
---


## 1. Caso de Negocio

Se ha requerido estimar los costos de dos equipos esenciales para un proyecto de construcción, con una duración de 36 meses. El cliente (empresa constructora) debe proporcionar los equipos necesarios, y el análisis se centra en la estimación del precio de los equipos __1__ y __2__. Dichos precios dependen directamente del valor de mercado de las materias primas $X$, $Y$, $Z$. El equipo 1 está compuesto en un 20% por la materia prima $X$ y un 80% por la materia prima $Y$. Por otro lado, el equipo 2 está compuesto por iguales proporciones de las materias primas $X$, $Y$ y $Z$.
El objetivo de este estudio es optimizar el monto de inversión que la empresa constructora debe realizar en la adquisición de los equipos en el futuro.

## 2. Supuestos

- Se cuenta con datos históricos diarios de precios de las materias primas $X$, $Y$ y $Z$.
- Los datos históricos se recopilan desde junio 1988, noviembre 2006 y enero 2010 para $X$, $Y$ y $Z$, respectivamente.
- Los datos históricos se recopilan hasta abril 2024, diciembre 2023 y agosto 2023 para $X$, $Y$ y $Z$, respectivamente.
- Cada materia prima es susceptible a distinta volatilidad [^1].

## 3. Formas para Resolver el Caso y Opción Tomada

$$
\int_{-\infty}^{\infty} H_n(x) H_m(x) e^{-x^2} \, dx = 0 \quad \text{para } n \neq m.
$$



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

% Bibliografía
% \nocite{*}
% 
% [^1]:  Eficaz*. Editorial Academia, 2019.
