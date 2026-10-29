# Clasificación de letras de puño cerrado en lenguaje de señas (ASL) con MLP
Evaluación Parcial N°1 – TLY1101 Técnicas Avanzadas de Machine Learning I

**Integrantes:** Bastian Sepulveda, Nicolas Bravo, Marcelo Obreque

## Descripción del problema de negocio
Las personas con discapacidad auditiva enfrentan barreras de comunicación. Se busca un sistema de visión computacional que traduzca letras del alfabeto ASL a texto, facilitando la inclusión.

## Objetivos
- Implementar, entrenar y evaluar un MLP que clasifique imágenes 28×28 en 6 letras ASL de puño cerrado: A, E, M, N, S, T.
- Comparar 3 arquitecturas y seleccionar la mejor con un conjunto de validación independiente del entrenamiento.
- Analizar errores y limitaciones del MLP en imágenes.

## KPIs
Fijados antes de entrenar la variante y sin ajustarlos después de ver resultados.

| KPI | Meta | Resultado (test final) |
|---|---|---|
| Accuracy | ≥ 80% | 79,4% (no cumplido) |
| F1-Score macro | ≥ 75% | 0,784 (cumplido) |

## Fuentes de datos
Sign Language MNIST (Kaggle). Archivos en `data/`. Descarga:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

**Variante propia:** se redefine el problema y no solo el tamaño. Se clasifican únicamente las letras de puño cerrado o dedos recogidos (A, E, M, N, S, T), que se distinguen casi solo por la posición del pulgar y la forma del puño. Se usa el 100% de las imágenes de esas 6 clases: 6.674 de train y 2.008 de test.

## Preparación y EDA
Sin nulos ni duplicados. El train está razonablemente balanceado (957 a 1.199 imágenes por clase), pero el test no lo está (246 a 498 por clase), por lo que además del accuracy se reportan Precision, Recall y F1 por clase. Normalización /255 y one-hot.

**Validación independiente:** las imágenes del train son aumentaciones de pocas manos originales, así que una validación sacada del train da valores irreales. Por eso el train se usa solo para entrenar y el archivo de test (otras manos) se divide 50/50 de forma estratificada (semilla 42): 1.004 imágenes para **validación** (elección de modelo y early stopping) y 1.004 para el **test final** (reporte). Limitación: el dataset no trae identificador de persona, por lo que las dos mitades podrían compartir manos entre sí.

## Metodología (CRISP-DM)
1. Comprensión del negocio  2. Comprensión de los datos (EDA)  3. Preparación de datos  4. Modelado (3 MLP)  5. Evaluación (métricas, matriz de confusión, errores)  6. Conclusiones / despliegue futuro.

## Resultados
| Modelo | Épocas | Val Loss | Val Acc | Test final Acc |
|---|---|---|---|---|
| Baseline (128) | 18 | 0.574 | 81,1% | 79,4% |
| Profundo (256-128) | 17 | 0.709 | 84,0% | 83,4% |
| Profundo + Dropout | 13 | 0.620 | 78,0% | 73,9% |

Mejor modelo: Baseline (128), elegido por val_loss en la validación (criterio definido de antemano; el test final no se usa para decidir). El Profundo tiene el mayor accuracy en validación y en test final, pero también el mayor val_loss. Las diferencias pequeñas entre arquitecturas son sensibles al ruido numérico de cada entorno. El modelo elegido sobreajusta (accuracy de entrenamiento 99,3% frente a 81,1% en validación). Las letras más difíciles son S (F1 0,51), M (0,70) y N (0,72). Ver `images/` para curvas, matriz de confusión y ejemplos.

## Estructura del proyecto
data/  models/  images/  train_sign.ipynb  requirements.txt  README.md

## Reproducibilidad
Probado con Python 3.13 y las versiones de requirements.txt

pip install -r requirements.txt
Ejecutar `train_sign.ipynb` de arriba a abajo (semilla global = 42). Los resultados pueden variar levemente según la versión de TensorFlow y el hardware..

## Figuras
![Curvas](images/curvas_mejor_modelo.png)

![Matriz de confusión](images/matriz_confusion_test.png)

![Ejemplos](images/ejemplos_clasificacion.png)
