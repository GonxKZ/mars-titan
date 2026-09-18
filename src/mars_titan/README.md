# Estructura científica prevista

Este directorio reserva el espacio para la lógica reutilizable de MARS-TITAN. Aún no contiene módulos implementados ni un paquete instalable. La configuración actual de uv utiliza `package = false`.

Cuando comience la implementación, la organización prevista será:

| Módulo futuro | Responsabilidad |
| --- | --- |
| `data` | Leer fuentes, comprobar disponibilidad temporal y preparar muestras y máscaras |
| `targets` | Construir etiquetas y retornos residuales con fechas de maduración explícitas |
| `models` | Codificar modalidades, combinarlas y obtener predicciones |
| `memory` | Consultar, actualizar, reiniciar y conservar el estado de memoria |
| `calibration` | Calibrar incertidumbre usando únicamente etiquetas ya disponibles |
| `training` | Ajustar parámetros y registrar configuración, semillas y estado inicial |
| `evaluation` | Evaluar predicciones guardadas y simular la política financiera especificada |

Esta tabla define responsabilidades, no interfaces aprobadas ni funcionalidad existente. Cada módulo se concretará con el [protocolo de investigación](../../docs/research/protocol.md) y la [arquitectura propuesta](../../docs/engineering/architecture.md).

La evaluación no reajustará modelos a partir del test. La memoria solo recibirá la información autorizada por la política temporal. La integración con código nativo dependerá de evidencia de perfilado y de una comparación numérica con la referencia Python.
