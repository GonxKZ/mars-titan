# Arquitectura propuesta

MARS-TITAN se diseña como una comparación reproducible de componentes. La estructura separa las responsabilidades necesarias para implementar y comparar cada componente científico.

La [especificación candidata](../research/candidate-architecture.md) desarrolla memoria episódica, retención, recurrencia y actualización coherente. Su [revisión adversarial](../research/adversarial-review.md) recoge objeciones y pruebas pendientes. La comparación principal prioriza una variante compacta sobre 128 activos propuestos, dentro de 32 GB de RAM y 8 GB de VRAM.

## Límites entre componentes

| Componente | Entrada | Salida y responsabilidad |
| --- | --- | --- |
| Datos | Fuentes y reglas de disponibilidad. | Muestras y máscaras auditables. No conoce modelos ni calcula resultados financieros. |
| Objetivos | Precios futuros usados solo como etiquetas y coeficientes temporales. | Retorno bruto/residual, intervalo y fecha de maduración. |
| Codificadores | Ventanas históricas y modalidades disponibles. | Vectores compactos. Sin acceso al objetivo futuro. |
| Memoria | Representaciones observadas, estado anterior y eventos autorizados de actualización. | Estado consultable y registro de lectura/escritura. Sin acceso libre a tablas de etiquetas. |
| Predicción | Representación actual y lectura de memoria. | Estimación puntual y cuantiles o escala. No ejecuta operaciones. |
| Calibración | Predicciones y etiquetas de un tramo reservado. | Intervalos y política de abstención fijados antes de evaluar. |
| Entrenamiento | Configuración, muestras permitidas y particiones. | Parámetros, estado inicial, curvas de ajuste y trazabilidad. |
| Evaluación | Predicciones inmutables, etiquetas maduras y política experimental. | Métricas, simulación y tablas. No reajusta el modelo usando el test. |

## Modelo compacto de partida

El codificador de precios podrá ser una GRU pequeña o una TCN. El texto se representará con embeddings congelados y una proyección de dimensión reducida, con procedencia y fecha del modelo documentadas. La fusión incluirá máscaras de modalidad y un mecanismo sencillo de combinación antes de introducir atención adicional.

La primera memoria será global. Una consulta dependerá de la representación actual y recuperará un vector de contexto que la cabeza combine con el estado temporal. El estado y sus operaciones `read`, `update`, `reset` y `snapshot` deben tener responsabilidades separadas. Estas son interfaces previstas, todavía sin código. Después se podrá comparar la separación mercado/sector/activo manteniendo control de capacidad total.

La actualización se estudiará en dos referencias separadas. Un banco episódico inserta y recupera eventos con lectura aprendida. Una memoria asociativa neural modifica un estado de pesos rápidos mediante una regla explícita. No son mecanismos equivalentes. La [candidata](../research/candidate-architecture.md) especifica esa separación y reserva su combinación para una extensión. En la variante neural se documentará qué gradientes se propagan, cuáles se detienen y qué estado persiste. Una adaptación no se denomina reproducción de Titans sin comprobar su correspondencia con [Titans, TTT y MIRAS](../references/neural-review.md).

## Sorpresa y régimen

La puntuación propuesta combina un error predictivo **maduro**, una anomalía calculada sobre historia y un indicador de relevancia económica cuya fuente sea verificable. Se normaliza cada componente con estadísticas del pasado y se estudia cada uno por separado. Si no hay expectativas de consenso observadas en tiempo real, no se llamará «sorpresa de beneficios» a la diferencia frente a un valor estimado retrospectivamente.

No se fijan pesos supuestamente óptimos sin experimento. El umbral y los pesos se eligen con validación. El registro incluirá escrituras por periodo, magnitud de actualización y saturación. La escritura aleatoria o uniforme con igual presupuesto sirve para comprobar si el efecto proviene de seleccionar eventos o solo de reducir ruido y cómputo.

El régimen inicial puede usar volatilidad y tendencia retrospectivas, con umbrales estimados en entrenamiento. Si se añade un modelo de estados ocultos, se utilizarán probabilidades **filtradas**, no estados suavizados con observaciones futuras. La identificación de un régimen sirve para condicionar o analizar predicciones, sin convertirlo en una explicación causal del mercado.

## Incertidumbre y abstención

La primera salida incierta puede ser una cabeza de cuantiles. Se medirán pérdida de cuantiles, cobertura y anchura. NLL solo será aplicable si otra variante define una densidad predictiva. El núcleo calibra en un tramo distinto del ajuste y del test y después congela el calibrador. Una extensión ACI podrá adaptarse de forma secuencial con etiquetas maduras bajo una política fijada antes de evaluar, sin selección de hiperparámetros durante el test. Ensambles pequeños o MC dropout son alternativas de coste a medir. Ninguna produce por sí sola una garantía de cobertura temporal. La decisión de abstenerse se evalúa a cobertura comparable y con exposición económica explícita.

## Python, C y C++/CUDA

Python será la referencia de corrección. `native/` contiene únicamente la configuración CMake y los límites previstos para código nativo. Un kernel solo se justifica cuando un perfil identifique un coste relevante que no resuelvan operaciones existentes de PyTorch. Se compararán resultados, gradientes, tipos, tamaños vacíos/no contiguos y errores de dispositivo antes de sustituir la referencia.

El diseño evita servicios distribuidos, colas remotas y un despliegue en producción. Los artefactos de experimento son archivos locales con un registro estructurado. Se incorporará un gestor adicional solo si resuelve una necesidad observada.
