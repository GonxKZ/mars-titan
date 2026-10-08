# Arquitectura propuesta

MARS-TITAN se diseña como una comparación reproducible de componentes. La estructura separa las responsabilidades necesarias para implementar y comparar cada componente científico.

La [dirección arquitectónica vigente](../research/titans-mac-architecture.md) separa la referencia GRU con banco episódico, Transformer compacto, Titans-MAC y MARS-TITAN con ampliaciones. La [especificación episódica](../research/candidate-architecture.md) conserva retención, recurrencia y escritura del candidato anterior. Las comparaciones usan todo el universo admisible de su edición con lectura por lotes, sin un límite de 128 activos.

La [revisión de integración](../research/system-integration.md) sitúa las ampliaciones en una vista común de información y un ciclo cronológico de decisión y maduración. Los componentes de lectura, actualización asociativa, escritura episódica y replay conservan responsabilidades distintas. El componente GRU de #296, las referencias existentes y los módulos aislados no acreditan la integración de Titans-MAC ni una comparación científica de las nuevas variantes.

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

La GRU y el Transformer compacto compartirán proyecciones, fusión y cabeza cuando el contraste aísle el codificador de precios. Titans-MAC añade atención cercana, memoria neuronal actualizable y parámetros persistentes aprendidos. El texto conserva embeddings congelados con procedencia y fecha documentadas. Las ausencias de la edición histórica mantienen sus máscaras, sin cambiar las filas por arquitectura.

La primera memoria del candidato será global. Una consulta dependerá de la representación actual y recuperará un vector de contexto que la cabeza combine con el estado temporal. El estado y sus operaciones `read`, `update`, `reset` y `snapshot` deben tener responsabilidades separadas. Estas son interfaces previstas para el candidato. La memoria nativa ya implementada pertenece a los comparadores financieros y tiene otro ámbito y dimensiones. Después se podrá comparar la separación mercado/sector/activo manteniendo control de capacidad total.

La memoria neuronal aplica la pérdida asociativa, gradientes, momentum y olvido dependientes de la entrada del núcleo Titans-MAC. El banco episódico sigue siendo una ampliación separada. Los pesos rápidos, los parámetros persistentes de Titans, los episodios y el estado de trabajo tienen ciclos y checkpoints distintos. Se documentará qué gradientes se propagan y qué estado se desacopla al persistirlo. Una regla delta aislada no sustituye esa correspondencia con el artículo.

## Sorpresa y régimen

La puntuación propuesta combina un error predictivo **maduro**, una anomalía calculada sobre historia y un indicador de relevancia económica cuya fuente sea verificable. Se normaliza cada componente con estadísticas del pasado y se estudia cada uno por separado. Si no hay expectativas de consenso observadas en tiempo real, no se llamará «sorpresa de beneficios» a la diferencia frente a un valor estimado retrospectivamente.

No se fijan pesos supuestamente óptimos sin experimento. El umbral y los pesos se eligen con validación. El registro incluirá escrituras por periodo, magnitud de actualización y saturación. La escritura aleatoria o uniforme con igual presupuesto sirve para comprobar si el efecto proviene de seleccionar eventos o solo de reducir ruido y cómputo.

El régimen inicial puede usar volatilidad y tendencia retrospectivas, con umbrales estimados en entrenamiento. Si se añade un modelo de estados ocultos, se utilizarán probabilidades **filtradas**, no estados suavizados con observaciones futuras. La identificación de un régimen sirve para condicionar o analizar predicciones, sin convertirlo en una explicación causal del mercado.

## Incertidumbre y abstención

La primera salida incierta puede ser una cabeza de cuantiles. Se medirán pérdida de cuantiles, cobertura y anchura. NLL solo será aplicable si otra variante define una densidad predictiva. El núcleo calibra en un tramo distinto del ajuste y del test y después congela el calibrador. Una extensión ACI podrá adaptarse de forma secuencial con etiquetas maduras bajo una política fijada antes de evaluar, sin selección de hiperparámetros durante el test. Ensambles pequeños o MC dropout son alternativas de coste a medir. Ninguna produce por sí sola una garantía de cobertura temporal. La decisión de abstenerse se evalúa a cobertura comparable y con exposición económica explícita.

## Python, C y C++/CUDA

Python mantiene las referencias científicas y parte de la preparación. `native/` contiene C++20 para contabilidad, simulación por lotes, filtrado HMM, memoria episódica, replay, políticas y recuperación, además de los ejecutables `mars-titan-sim` y `mars-titan-ppo`. Estos componentes no constituyen una implementación del candidato. La revisión de integración distingue qué contratos reutilizar y qué semánticas financieras deben permanecer separadas.

La ruta intensiva futura del candidato se propone como un ejecutable C++20 con LibTorch y bibliotecas matemáticas existentes, con una referencia de corrección y artefactos comparables. Un kernel propio solo se justifica cuando un perfil identifique un coste que no resuelvan mejor esas primitivas. Se compararán resultados, gradientes, tipos, formas, recuperación y errores de dispositivo antes de sustituir una referencia. El uso de C++ no acredita por sí solo una aceleración de operaciones que ya ejecutan código nativo.

El diseño evita servicios distribuidos, colas remotas y un despliegue en producción. Los artefactos de experimento son archivos locales con un registro estructurado. Se incorporará un gestor adicional solo si resuelve una necesidad observada.
