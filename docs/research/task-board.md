# Catálogo del tablero de MARS-TITAN

El [Project](https://github.com/users/GonxKZ/projects/4) organiza tareas revisables, no porcentajes de éxito científico. Cada issue contiene herramientas por función, entradas, pasos, artefactos previstos y comprobaciones. El [catálogo JSON](../../.github/planning/issues.json) conserva sus cuerpos y el [mapa remoto](../../.github/planning/remote-map.json) registra la verificación de GitHub.

El catálogo conserva **64 tareas canónicas y 183 dependencias**. Los estados de sus tablas son la instantánea de planificación inicial, no un contador en directo. El estado operativo se consulta en el Project y la evidencia posterior en [preparación](../data/preparation.md) y [mediciones](../../reports/resources/campaign-budget.md). Las cinco duplicadas permanecen cerradas como no planificadas y fuera del tablero.

## Cómo utilizar una issue

Leer primero su contexto, guía y criterios. Las rutas de implementación son previstas salvo evidencia expresa de existencia. Crear la rama vinculada, mover la tarjeta a En curso y conservar comprobaciones y commits. Pasar a En revisión con evidencia y a Hecho solo tras comprobar sus criterios. No ejecutar por anticipado órdenes que dependen de componentes aún no implementados.

La [guía de implementación](../engineering/implementation-guide.md) explica la elección de lenguajes y bibliotecas. El núcleo usa Python/PyTorch y bibliotecas numéricas. C++/CUDA propios dependen de perfilado. Go no es una dependencia del estudio. Las comprobaciones son locales y el único workflow propio publica GitHub Pages. La [limitación de los automatismos internos](../engineering/observatory.md#límite-de-los-automatismos-de-github) permanece pendiente de decisión en MT-068.

## Alcance de las tareas

| Función | Tareas | Criterio |
| --- | ---: | --- |
| Núcleo científico | 39 | Produce la evidencia de los seis objetivos, incluido el contexto macro obligatorio. |
| Apoyo | 15 | Entorno, ejecución, trazabilidad y entrega. |
| Extensión | 10 | Se activa con pregunta, datos y presupuesto. No bloquea el mínimo científico. |

Una prioridad P1 no significa opcional. La etiqueta `opcional` señala extensiones. Las tareas del observatorio y de estas guías son resultados de preparación distintos de los modelos y sus experimentos. La auditoría original de duplicados se conserva en [revisión del catálogo](backlog-review.md).

## Tareas por objetivo

### Trabajo transversal

| Tarea | Función | Prioridad | Estado | Dependencias |
| --- | --- | --- | --- | --- |
| [MT-001 · Preparar requisitos, rúbrica y revisión del original](https://github.com/GonxKZ/mars-titan/issues/1) | Apoyo | P0 | Hecho | Ninguna |
| [MT-002 · Confirmar requisitos y calendario académico](https://github.com/GonxKZ/mars-titan/issues/2) | Apoyo | P0 | Pendiente | MT-001 |
| [MT-004 · Verificar el entorno con uv y CUDA](https://github.com/GonxKZ/mars-titan/issues/4) | Apoyo | P0 | Pendiente | Ninguna |
| [MT-069 · Concretar las guías de ejecución y el alcance](https://github.com/GonxKZ/mars-titan/issues/71) | Apoyo | P0 | Hecho | MT-001 |

### O1 · Datos multimodales y disponibilidad temporal

| Tarea | Función | Prioridad | Estado | Dependencias |
| --- | --- | --- | --- | --- |
| [MT-005 · Inventariar FinMultiTime](https://github.com/GonxKZ/mars-titan/issues/5) | Núcleo | P0 | Pendiente | Ninguna |
| [MT-006 · Auditar precios y universo histórico](https://github.com/GonxKZ/mars-titan/issues/6) | Núcleo | P0 | Pendiente | MT-005 |
| [MT-007 · Normalizar noticias y su disponibilidad](https://github.com/GonxKZ/mars-titan/issues/7) | Núcleo | P0 | Pendiente | MT-005 |
| [MT-008 · Reconstruir fundamentales por publicación](https://github.com/GonxKZ/mars-titan/issues/8) | Núcleo | P0 | Pendiente | MT-005 |
| [MT-009 · Generar gráficos con historia disponible](https://github.com/GonxKZ/mars-titan/issues/9) | Núcleo | P1 | Pendiente | MT-006 |
| [MT-010 · Auditar y versionar codificadores congelados](https://github.com/GonxKZ/mars-titan/issues/10) | Núcleo | P1 | Pendiente | MT-007, MT-009, MT-004 |
| [MT-011 · Construir el contrato temporal de datos](https://github.com/GonxKZ/mars-titan/issues/11) | Núcleo | P0 | Pendiente | MT-006, MT-007, MT-008 |
| [MT-012 · Definir la selección y publicar el piloto](https://github.com/GonxKZ/mars-titan/issues/12) | Núcleo | P0 | Pendiente | MT-005, MT-006, MT-007, MT-008, MT-011 |
| [MT-048 · Preparar datos por bloques y ventanas bajo demanda](https://github.com/GonxKZ/mars-titan/issues/50) | Apoyo | P1 | Pendiente | MT-004, MT-005, MT-011, MT-012 |
| [MT-056 · Auditar indicadores macro y sus versiones](https://github.com/GonxKZ/mars-titan/issues/58) | Núcleo | P0 | Pendiente | MT-003, MT-005 |
| [MT-057 · Integrar macro con versiones y disponibilidad](https://github.com/GonxKZ/mars-titan/issues/59) | Núcleo | P0 | Pendiente | MT-011, MT-056 |
| [MT-066 · Verificar fuentes públicas complementarias](https://github.com/GonxKZ/mars-titan/issues/68) | Extensión | P2 | Pendiente | MT-005 |
| [MT-067 · Implementar actualizaciones con instantáneas inmutables](https://github.com/GonxKZ/mars-titan/issues/69) | Extensión | P2 | Pendiente | MT-011, MT-066 |

### O2 · Retornos residuales y etiquetas

| Tarea | Función | Prioridad | Estado | Dependencias |
| --- | --- | --- | --- | --- |
| [MT-013 · Definir objetivo y reloj de decisión](https://github.com/GonxKZ/mars-titan/issues/13) | Núcleo | P0 | Pendiente | MT-001 |
| [MT-014 · Versionar factores de mercado y sector](https://github.com/GonxKZ/mars-titan/issues/14) | Núcleo | P0 | Pendiente | MT-005, MT-006, MT-013 |
| [MT-015 · Calcular residuales y maduración de etiquetas](https://github.com/GonxKZ/mars-titan/issues/15) | Núcleo | P0 | Pendiente | MT-011, MT-013, MT-014 |
| [MT-016 · Diagnosticar cobertura y estabilidad del residual](https://github.com/GonxKZ/mars-titan/issues/16) | Núcleo | P1 | Pendiente | MT-012, MT-015 |
| [MT-058 · Versionar calendarios y reglas de mercado](https://github.com/GonxKZ/mars-titan/issues/60) | Núcleo | P1 | Pendiente | MT-005, MT-013 |

### O3 · Memoria de eventos e incertidumbre

| Tarea | Función | Prioridad | Estado | Dependencias |
| --- | --- | --- | --- | --- |
| [MT-017 · Implementar codificación y fusión de eventos](https://github.com/GonxKZ/mars-titan/issues/17) | Núcleo | P1 | Pendiente | MT-004, MT-011, MT-012 |
| [MT-018 · Implementar memoria global acotada](https://github.com/GonxKZ/mars-titan/issues/18) | Núcleo | P1 | Pendiente | MT-017 |
| [MT-019 · Comparar políticas de escritura](https://github.com/GonxKZ/mars-titan/issues/19) | Núcleo | P1 | Pendiente | MT-015, MT-018, MT-021, MT-028 |
| [MT-020 · Comparar enrutamiento por regímenes](https://github.com/GonxKZ/mars-titan/issues/20) | Núcleo | P1 | Pendiente | MT-018, MT-019 |
| [MT-021 · Verificar orden temporal y aislamiento](https://github.com/GonxKZ/mars-titan/issues/21) | Núcleo | P0 | Pendiente | MT-015, MT-018 |
| [MT-022 · Implementar cuantiles y calibración](https://github.com/GonxKZ/mars-titan/issues/22) | Núcleo | P1 | Pendiente | MT-013, MT-017, MT-021, MT-030 |
| [MT-023 · Integrar el modelo y ejecutar un ensayo mínimo](https://github.com/GonxKZ/mars-titan/issues/23) | Núcleo | P1 | Pendiente | MT-004, MT-017, MT-018, MT-019, MT-020, MT-021, MT-022 |
| [MT-052 · Medir retención e interferencia](https://github.com/GonxKZ/mars-titan/issues/54) | Núcleo | P1 | Pendiente | MT-024, MT-030, MT-019 |
| [MT-053 · Evaluar consolidación y replay](https://github.com/GonxKZ/mars-titan/issues/55) | Extensión | P2 | Pendiente | MT-019, MT-052, MT-059 |
| [MT-059 · Comprobar compatibilidad de las versiones](https://github.com/GonxKZ/mars-titan/issues/61) | Apoyo | P0 | Pendiente | MT-010, MT-021, MT-022 |
| [MT-063 · Medir concurrencia y antigüedad del estado](https://github.com/GonxKZ/mars-titan/issues/65) | Extensión | P2 | Pendiente | MT-053, MT-059, MT-060 |

### O4 · Referencias y ablaciones

| Tarea | Función | Prioridad | Estado | Dependencias |
| --- | --- | --- | --- | --- |
| [MT-024 · Validar cero, Ridge y el cálculo por bloques](https://github.com/GonxKZ/mars-titan/issues/24) | Núcleo | P0 | Pendiente | MT-015, MT-031, MT-032 |
| [MT-025 · Evaluar la referencia de boosting](https://github.com/GonxKZ/mars-titan/issues/25) | Núcleo | P1 | Pendiente | MT-024 |
| [MT-026 · Evaluar GRU y decidir sobre DLinear](https://github.com/GonxKZ/mars-titan/issues/26) | Núcleo | P1 | Pendiente | MT-004, MT-024 |
| [MT-027 · Evaluar una referencia de memoria identificable](https://github.com/GonxKZ/mars-titan/issues/27) | Núcleo | P1 | Pendiente | MT-003, MT-018, MT-021, MT-031, MT-032 |
| [MT-028 · Diseñar ablaciones y falsaciones](https://github.com/GonxKZ/mars-titan/issues/28) | Núcleo | P0 | Pendiente | MT-013, MT-017, MT-030, MT-003 |
| [MT-029 · Ejecutar ablaciones y falsaciones](https://github.com/GonxKZ/mars-titan/issues/29) | Núcleo | P1 | Pendiente | MT-010, MT-023, MT-027, MT-028, MT-031, MT-032 |
| [MT-050 · Contrastar familias secuenciales adicionales](https://github.com/GonxKZ/mars-titan/issues/52) | Extensión | P2 | Pendiente | MT-026, MT-028, MT-048 |
| [MT-054 · Evaluar pasos recurrentes y parada](https://github.com/GonxKZ/mars-titan/issues/56) | Extensión | P2 | Pendiente | MT-023, MT-028, MT-032 |
| [MT-055 · Evaluar destilación temporalmente válida](https://github.com/GonxKZ/mars-titan/issues/57) | Extensión | P2 | Pendiente | MT-030, MT-054, MT-059 |
| [MT-064 · Evaluar ampliaciones del universo](https://github.com/GonxKZ/mars-titan/issues/66) | Extensión | P2 | Pendiente | MT-030, MT-048, MT-060, MT-024, MT-025 |

### O5 · Evaluación temporal y económica

| Tarea | Función | Prioridad | Estado | Dependencias |
| --- | --- | --- | --- | --- |
| [MT-030 · Fijar el protocolo y la reserva final](https://github.com/GonxKZ/mars-titan/issues/30) | Núcleo | P0 | Pendiente | MT-013 |
| [MT-031 · Implementar ejecución y trazabilidad](https://github.com/GonxKZ/mars-titan/issues/31) | Apoyo | P0 | Pendiente | MT-004, MT-012, MT-015, MT-030, MT-065, MT-048 |
| [MT-032 · Verificar métricas y agregaciones](https://github.com/GonxKZ/mars-titan/issues/32) | Núcleo | P0 | Pendiente | MT-013, MT-030 |
| [MT-033 · Implementar la simulación con costes](https://github.com/GonxKZ/mars-titan/issues/33) | Núcleo | P1 | Pendiente | MT-006, MT-013, MT-030, MT-058 |
| [MT-034 · Fijar muestra principal y finalistas](https://github.com/GonxKZ/mars-titan/issues/34) | Núcleo | P1 | Pendiente | MT-023, MT-024, MT-025, MT-026, MT-027, MT-028, MT-031, MT-032, MT-033, MT-059, MT-060, MT-029 |
| [MT-035 · Ejecutar la comparación fuera de muestra](https://github.com/GonxKZ/mars-titan/issues/35) | Núcleo | P0 | Pendiente | MT-029, MT-034 |
| [MT-036 · Estimar incertidumbre y efecto de la selección](https://github.com/GonxKZ/mars-titan/issues/36) | Núcleo | P1 | Pendiente | MT-035 |
| [MT-037 · Evaluar robustez y abstención](https://github.com/GonxKZ/mars-titan/issues/37) | Núcleo | P1 | Pendiente | MT-035 |
| [MT-065 · Verificar checkpoints y reanudación](https://github.com/GonxKZ/mars-titan/issues/67) | Apoyo | P0 | Pendiente | MT-004, MT-030 |

### O6 · Análisis, memoria y defensa

| Tarea | Función | Prioridad | Estado | Dependencias |
| --- | --- | --- | --- | --- |
| [MT-003 · Completar el estado del arte y las referencias](https://github.com/GonxKZ/mars-titan/issues/3) | Núcleo | P1 | Pendiente | MT-001 |
| [MT-038 · Informar recursos de los experimentos finales](https://github.com/GonxKZ/mars-titan/issues/38) | Núcleo | P0 | Pendiente | MT-023, MT-026, MT-027, MT-035, MT-060 |
| [MT-039 · Decidir sobre optimización nativa](https://github.com/GonxKZ/mars-titan/issues/39) | Extensión | P2 | Pendiente | MT-038 |
| [MT-040 · Redactar conclusiones de los seis objetivos](https://github.com/GonxKZ/mars-titan/issues/40) | Núcleo | P1 | Pendiente | MT-016, MT-029, MT-036, MT-037, MT-038, MT-052 |
| [MT-041 · Preparar y entregar el primer borrador](https://github.com/GonxKZ/mars-titan/issues/41) | Apoyo | P1 | Pendiente | MT-001, MT-002, MT-003, MT-013, MT-030 |
| [MT-042 · Preparar y entregar el segundo borrador](https://github.com/GonxKZ/mars-titan/issues/42) | Apoyo | P1 | Pendiente | MT-041, MT-012, MT-015, MT-024, MT-023, MT-028 |
| [MT-043 · Completar y entregar el tercer borrador](https://github.com/GonxKZ/mars-titan/issues/43) | Apoyo | P1 | Pendiente | MT-042, MT-040 |
| [MT-044 · Reproducir figuras y revisar las evidencias](https://github.com/GonxKZ/mars-titan/issues/44) | Apoyo | P0 | Pendiente | MT-003, MT-035, MT-036, MT-037, MT-038, MT-043 |
| [MT-045 · Cerrar predepósito y tramitar la autorización](https://github.com/GonxKZ/mars-titan/issues/45) | Apoyo | P0 | Pendiente | MT-002, MT-043, MT-044 |
| [MT-046 · Preparar y realizar la defensa](https://github.com/GonxKZ/mars-titan/issues/46) | Apoyo | P0 | Pendiente | MT-040, MT-044, MT-045 |
| [MT-060 · Medir el piloto y fijar el presupuesto](https://github.com/GonxKZ/mars-titan/issues/62) | Apoyo | P0 | Pendiente | MT-004, MT-023, MT-026, MT-027 |
| [MT-068 · Publicar el observatorio de experimentos](https://github.com/GonxKZ/mars-titan/issues/70) | Extensión | P1 | Bloqueado | MT-001 |

## Cierre temporal y dependencias

MT-030 fija reglas y reserva final. MT-012 publica el piloto. MT-060 mide desde ensayos mínimos. MT-034 congela muestra y receta. MT-035 sella la versión ajustada y calibrada antes del test. MT-036 aplica el método estadístico ya predefinido. La retención de MT-052 alimenta conclusiones y las reglas de MT-058 condicionan la simulación estadounidense.

La lectura por bloques de MT-048 alimenta el ejecutor, sin duplicar canalizaciones. Ninguna ruta necesaria para núcleo o apoyo depende de una extensión. El observatorio consume el contrato de estado que integrará MT-031, pero su interfaz puede verificarse antes con casos controlados y un registro inicial vacío.

Los números de issue no siempre coinciden con los identificadores MT. Deben utilizarse los enlaces del mapa, no construir números por suposición. No hay fechas de entrega inventadas. El calendario se incorporará cuando esté confirmado.
