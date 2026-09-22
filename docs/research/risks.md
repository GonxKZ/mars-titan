# Riesgos y decisiones de reducción de alcance

Los riesgos se priorizan por su capacidad para invalidar la comparación. No se asignan puntuaciones numéricas de probabilidad sin observaciones que las justifiquen.

| Riesgo | Señal que lo detecta | Medida y evidencia exigida |
| --- | --- | --- |
| Fuga por informes contables | Cifras disponibles antes de `filed` o mezcla de versiones. | Unir por versión/publicación. Excluir registros sin fecha defendible. Pruebas de contaminación futura. |
| Noticias o imágenes con información posterior | Texto sin hora. Imagen cubre el futuro respecto a t. | Retardo conservador, contraste de relevancia y regeneración de gráficos históricos. |
| Contaminación de memoria | Las predicciones cambian al permutar activos del mismo instante o al cargar otra ventana de evaluación. | Instantánea por sesión, etiquetas maduras, reinicio del estado y pruebas de independencia. |
| Supervivencia y universo retrospectivo | Selección por constituyentes actuales o por datos hasta el final. | Pertenencia temporal cuando exista. Registrar exclusiones y limitar inferencias. |
| Conocimiento retrospectivo del codificador | Modelo externo entrenado/publicado después del test histórico. | Registrar corpus y fecha. Análisis separado y evitar afirmar disponibilidad histórica estricta. |
| Complejidad sin señal | El modelo completo mejora solo en validación o depende de una semilla. | Referencias sencillas, ablaciones emparejadas, intervalos por bloques y evaluación final reservada. |
| Agotamiento de VRAM | Picos al escribir memoria o calcular gradientes internos. | Medir, reducir dimensión, lote o contexto y precalcular representaciones. No ocultar un cambio de dispositivo. |
| Rendimiento económico aparente | Los costes omiten cierre, préstamo o rotación. Los residuos se tratan como beneficios. | Registro de operaciones con retornos reales, costes por lado y escenarios de sensibilidad. |
| Incertidumbre mal interpretada | Buena cobertura global y mala por régimen. | Desgloses, anchura, riesgo-cobertura y limitaciones conformales explícitas. |
| Búsqueda oportunista | Se prueban variantes hasta obtener una mejora sin contar intentos. | Registro de todas las configuraciones y congelación de comparaciones principales. |
| Referencias o licencias incorrectas | Un blog se presenta como artículo científico o un repositorio público carece de licencia. | Fuente primaria, nivel de evidencia y registro de derechos. |
| Informe final incoherente | Las conclusiones no responden a los seis objetivos. | Correspondencia entre cada objetivo, su evidencia y la conclusión que permite sostener. |
| Explicación insuficiente | No se puede explicar una figura, una ecuación o un fallo. | Revisar cada resultado junto a sus datos, método, supuestos y limitaciones. |

Si un riesgo temporal no puede resolverse, se acota la modalidad o la afirmación que depende de ella. Si una extensión consume el tiempo reservado a referencias, evaluación o redacción, se mantiene como trabajo futuro. El registro de decisiones debe conservar el motivo y el impacto sobre la propuesta presentada.
