# Neurona artificial para el riego de plantas

Este proyecto es una neurona artificial hecha en Python con NumPy que decide si una planta necesita riego. Recibe la humedad del suelo (0 a 100 %) y la temperatura ambiental (0 a 50 °C aproximadamente) y responde 1 si hay que regar o 0 si no. Los datos son de práctica y no sirven como recomendación real para ninguna planta.

La neurona tiene dos entradas, dos pesos y un sesgo, usa la función sigmoide y convierte la probabilidad en 0 o 1 con un umbral de 0.5. Antes de entrenar, las entradas se dividen por la escala [100, 50], y esa misma escala se usa para los datos nuevos.

## Cómo ejecutarlo

uv sync
uv run main.py

## Resultado del entrenamiento base

Con tasa de aprendizaje 0.5 y 10000 épocas:

- Peso de la humedad: -33.2571
- Peso de la temperatura: 0.1955
- Sesgo: 14.8320
- Error final (ECM): 0.005361
- Aciertos en los 10 datos de entrenamiento: 10 de 10

## Experimentos

En cada prueba cambié solo un parámetro. En todas la neurona acertó los 10 datos de entrenamiento, así que lo que cambia es el error y la rapidez del aprendizaje.

- Base (tasa 0.5, 10000 épocas): error 0.005361, probabilidad para 45 % y 34 °C de 0.4998. Aprendizaje bueno.
- Pocas épocas (tasa 0.5, 100 épocas): error 0.089490, probabilidad 0.5451. Insuficiente.
- Intermedia (tasa 0.5, 1000 épocas): error 0.030038, probabilidad 0.5603. Todavía estaba mejorando.
- Más épocas (tasa 0.5, 20000 épocas): error 0.002795, probabilidad 0.4856. Mejora pequeña frente a la base.
- Tasa pequeña (0.01, 10000 épocas): error 0.066192, probabilidad 0.5621. Aprendizaje lento.
- Tasa moderada (0.1, 10000 épocas): error 0.019328, probabilidad 0.5411. Aprendizaje moderado.
- Tasa alta (1.0, 10000 épocas): error 0.002795, probabilidad 0.4857. Rápido y estable.
- Tasa muy alta (2.0, 10000 épocas): error 0.001396, probabilidad 0.4713. Muy rápido y estable.

## Predicciones con condiciones nuevas

Con el modelo base y umbral 0.5:

- Humedad 75 %, temperatura 30 °C: probabilidad 0.0000, decisión 0 (no regar).
- Humedad 45 %, temperatura 34 °C: probabilidad 0.4998, decisión 0 (no regar).
- Humedad 25 %, temperatura 22 °C: probabilidad 0.9986, decisión 1 (regar).
- Humedad 50 %, temperatura 25 °C: probabilidad 0.1546, decisión 0 (no regar).
- Humedad 30 %, temperatura 40 °C: probabilidad 0.9934, decisión 1 (regar).

## Prueba con umbrales

Sin volver a entrenar, comparé los umbrales 0.4, 0.5 y 0.6 sobre las cinco condiciones nuevas:

- Umbral 0.4: [0, 1, 1, 0, 1]
- Umbral 0.5: [0, 0, 1, 0, 1]
- Umbral 0.6: [0, 0, 1, 0, 1]

En los datos de entrenamiento no cambió nada. Solo cambia el caso de 45 % y 34 °C, porque su probabilidad (0.4998) queda justo entre 0.4 y 0.5, y con umbral 0.4 pasa a "regar". Cambiar el umbral no modifica los pesos porque se aplica después, sobre la probabilidad ya calculada, y no participa en el entrenamiento.

## Análisis

1. ¿Por qué normalizar? La humedad llega hasta 100 y la temperatura hasta 50. Sin normalizar, la humedad pesaría mucho más en los cálculos y el entrenamiento sería más lento o inestable. Al dividir por la escala, las dos entradas quedan en un rango parecido.

2. ¿Dónde se usó X_normalizado y para qué se guardó X? X_normalizado se usó en la suma ponderada (z) y en el gradiente de los pesos. X se guardó solo para mostrar los valores originales y poder interpretar los resultados.

3. ¿Qué pasó con 100 épocas? El error fue de 0.0895, el más alto con tasa 0.5. Aunque acertó los 10 casos, las probabilidades eran poco seguras. Necesita más entrenamiento.

4. ¿Más épocas siempre mejoran mucho? No. De 100 a 10000 épocas el error bajó bastante, pero de 10000 a 20000 solo pasó de 0.0054 a 0.0028 y la clasificación fue la misma.

5. ¿Qué pasa con una tasa muy pequeña? Con 0.01 el error seguía en 0.066 después de 10000 épocas. Aprende, pero muy despacio.

6. ¿Qué pasa con una tasa alta o muy alta? Con 1.0 y 2.0 el error bajó más rápido y llegó más lejos (0.0028 y 0.0014). En este caso no hubo inestabilidad, probablemente porque son pocos datos y se separan bien. En otros problemas una tasa alta sí podría hacer que el entrenamiento oscile.

7. ¿Qué significa el signo del peso de la humedad? Es negativo (-33.26): a más humedad, menos probabilidad de regar. Tiene sentido, porque un suelo húmedo no necesita agua.

8. ¿Qué significa el signo del peso de la temperatura? Es positivo (0.1955): a más calor, un poco más de probabilidad de regar. Pero es muy pequeño comparado con el de la humedad, así que la neurona decide casi solo por la humedad. En los datos, todos los casos con 50 % o más de humedad son "no regar" y todos los de 40 % o menos son "regar".

9. ¿Por qué usar un umbral? La sigmoide da un número entre 0 y 1, pero al final hay que tomar una decisión: regar o no regar. El umbral convierte la probabilidad en esa decisión.

10. ¿Qué limitaciones tiene? Usa solo 10 datos y 2 variables. No tiene en cuenta el tipo de planta, el tipo de suelo, la lluvia, la hora del día ni la época del año. Además, se probó con los mismos datos con los que se entrenó, sin un conjunto aparte para validar, y los datos son inventados para la práctica.
