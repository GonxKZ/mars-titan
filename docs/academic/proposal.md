# Propuesta presentada de MARS-TITAN

Autor: **Gonzalo García Lama**. País: España. Modalidad: individual. Tipo de trabajo: **Tipo 3, comparativa de soluciones**. Participación en Tech4Good: no.

Título provisional presentado: **MARS-TITAN: memoria causal adaptativa para predicción bursátil multimodal**.

Este documento recoge el contenido y los seis objetivos comunicados por el estudiante el 18 de septiembre de 2026, con la denominación del proyecto unificada. No constituye una resolución de aprobación ni sustituye al formulario presentado en la facultad.

## Problema y justificación

El proyecto propone diseñar, implementar y evaluar experimentalmente una arquitectura de inteligencia artificial para predicción bursátil multimodal basada en memoria neural. El problema es la dificultad de predecir la evolución de activos con datos ruidosos, no estacionarios y procedentes de precios, noticias, información fundamental y patrones visuales, en presencia de cambios de régimen.

La motivación es que el aprendizaje de patrones locales puede resultar insuficiente para gestionar dependencias largas, acontecimientos inesperados e incertidumbre. Un modelo aparentemente preciso en validaciones simples puede fallar cuando cambia el mercado o cuando se han alineado incorrectamente sus fuentes de información.

MARS-TITAN estudiará representaciones multimodales, selección de eventos que merecen conservarse, distinción entre contexto de mercado, sector y activo, e incertidumbre. Se trabajará con un subconjunto reproducible de FinMultiTime y una comparación experimental controlada, compatible con un proyecto individual y unos 8 GB de VRAM. La contribución prevista es comprobar si la memoria orientada a eventos mejora la predicción frente a referencias y variantes más sencillas.

## Objetivos específicos

1. **Preparación de datos.** Construir un pipeline sobre FinMultiTime que integre precios, noticias, datos fundamentales y representaciones visuales o derivadas de gráficos, considerando la disponibilidad temporal y el riesgo de fuga de información.
2. **Problema predictivo.** Estudiar retornos residuales ajustados por mercado y, cuando sea posible, sector, para separar señales del activo de movimientos generales.
3. **Arquitectura.** Diseñar MARS-TITAN con memoria de eventos, escritura por sorpresa causal, representación de regímenes y estimación de incertidumbre. La sorpresa combinará error predictivo, anomalía y relevancia económica.
4. **Comparativa.** Implementar referencias estadísticas, de aprendizaje automático clásico y neuronales, junto con ablaciones sin memoria, con memoria global o sin sorpresa causal.
5. **Evaluación.** Medir error, dirección y calibración, además de ranking y simulaciones financieras con rentabilidad, drawdown, rotación y costes, dentro de un marco experimental.
6. **Análisis crítico.** Estudiar mejoras, fallos, contribución de modalidades y restricciones de cómputo para valorar la utilidad de la memoria adaptativa.

Cada objetivo corresponde a una fase de seguimiento. El orden de ejecución puede solapar actividades: los modelos base son necesarios antes de valorar la arquitectura propia.

## Metodología prevista

La metodología es experimental. Se revisará literatura sobre predicción financiera, series temporales, multimodalidad y memoria para concretar las preguntas e hipótesis. Después se preparará una tabla por activo e instante que relacione las entradas disponibles con el objetivo de predicción.

La preparación incluirá normalización, retornos, volatilidad, agregación de noticias, codificación tabular y estudio de ausencias y desalineación. Después se desarrollarán referencias como regresión regularizada, árboles o boosting y modelos de secuencia compactos. La selección definitiva atenderá al tiempo disponible y al hardware, manteniendo la equidad de comparación.

La arquitectura combinará un codificador temporal, representaciones multimodales compactas, memoria, escritura selectiva, régimen de mercado y una salida de retorno e incertidumbre. Esta última permitirá estudiar abstención o ausencia de señal cuando no exista confianza suficiente.

La evaluación será walk-forward, con entrenamiento en el pasado y evaluación en periodos posteriores. Las ablaciones medirán contribución de memoria, sorpresa, texto, fundamentales e incertidumbre. El análisis de errores incluirá alta volatilidad, cambios de régimen y datos incompletos.

Tecnologías previstas: Python, PyTorch, NumPy, pandas, scikit-learn, Polars o DuckDB, Matplotlib o Plotly, Jupyter, Git y LaTeX. El seguimiento podrá ampliarse a MLflow o Weights & Biases si existe una necesidad concreta. Se priorizarán modelos compactos, precisión mixta, embeddings precalculados y experimentos controlados. Las cifras de consumo deberán medirse.

## Impacto esperado y estado

El impacto académico esperado es una comparativa fundamentada sobre memoria adaptativa y predicción financiera multimodal. El técnico es un prototipo reproducible y acotado que facilite estudiar memoria, multimodalidad e incertidumbre. Ambos son objetivos por evaluar. La preparación del repositorio no demuestra que se hayan alcanzado.

Las precisiones sobre disponibilidad, adaptación y alcance se desarrollan en el [protocolo](../research/protocol.md) y la [revisión del documento original](../research/original-review.md). El antecedente PDF propio se conserva localmente en `docs/research/source/initial-proposal.pdf`.
