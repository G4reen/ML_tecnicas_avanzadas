# Clasificación de Lenguaje de Señas (ASL) con MLP
Evaluación Parcial N°1 – TLY1101 Técnicas Avanzadas de Machine Learning I

## Descripción del problema de negocio
Las personas con discapacidad auditiva enfrentan barreras de comunicación. Se busca un sistema de visión computacional que traduzca letras del alfabeto ASL a texto, facilitando la inclusión.

## Objetivos
- Implementar, entrenar y evaluar un MLP que clasifique imágenes 28×28 en 24 letras ASL (sin J ni Z).
- Comparar 3 arquitecturas y seleccionar la mejor con el conjunto de validación.
- Analizar errores y limitaciones del MLP en imágenes.

## KPIs
| KPI | Meta | Resultado |
|---|---|---|
| Accuracy en test | ≥ 80% | <X>% |
| F1-Score macro en test | ≥ 75% | <Y> |

## Fuentes de datos
Sign Language MNIST (Kaggle). 27.455 imágenes de entrenamiento y 7.172 de test tras filtrar etiquetas válidas, 24 clases, 784 píxeles (28×28, escala de grises). Archivos en `data/`. Descarga: <URL de Kaggle>.
**Variante propia:** 50% del train con muestreo estratificado (semilla 42).

## Preparación y EDA
Sin nulos ni duplicados. Clases balanceadas. Normalización /255, one-hot, split estratificado 80/20 train/val. Test aislado para evitar data leakage (las imágenes del dataset son aumentaciones de pocas manos originales).

## Metodología (CRISP-DM)
1. Comprensión del negocio  2. Comprensión de los datos (EDA)  3. Preparación de datos  4. Modelado (3 MLP)  5. Evaluación (métricas, matriz de confusión, errores)  6. Conclusiones / despliegue futuro.

## Resultados
| Modelo | Épocas | Val Loss | Val Acc |
|---|---|---|---|
| Baseline (128) | <> | <> | <> |
| Profundo (256-128) | <> | <> | <> |
| Profundo + Dropout | <> | <> | <> |

Mejor modelo: <NOMBRE>. Ver `images/` para curvas, matriz de confusión y ejemplos.

## Estructura del proyecto
data/  models/  images/  train_sign.ipynb  requirements.txt  README.md

## Reproducibilidad
pip install -r requirements.txt
Ejecutar `train_sign.ipynb` de arriba a abajo (semilla global = 42).