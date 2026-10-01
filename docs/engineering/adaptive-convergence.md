# Parada de la comparación financiera

`configs/simulation/adaptive-convergence-campaign.json` define la edición 2 de la campaña y utiliza `adaptive-ppo-convergence.json`, de esquema nativo 3. Se activa mediante `--config` en `scripts/run_adaptive_campaign.py`. Las configuraciones anteriores conservan sus formatos y sus presupuestos fijos.

La comparación principal y los casos auxiliares tienen un techo de 1.048.576 transiciones. La selección empieza con la política inicial y compara primero el número de ruinas y después el crecimiento logarítmico medio, con mejora mínima de 0,0001. Se conserva el mejor checkpoint aunque corresponda al estado inicial.

La paciencia cuenta ocho validaciones completas sin mejora después de 131.072 transiciones. La validación que coincide con ese mínimo deja el contador a cero. Una mejora posterior también lo reinicia. El intervalo es de 16.384 transiciones. En DoubleDQN, el mínimo debe cubrir su calentamiento de 256 transiciones y una parada exige al menos una actualización del optimizador. En esta edición, las validaciones se programan por transiciones, incluidas las del calentamiento.

El piloto mantiene sus 8.192 transiciones con parada desactivada y mínimo cero. Solo mide el coste del trabajo. La cuadrícula de esta edición contiene el techo de 1.048.576 transiciones. Si la previsión no cabe en el presupuesto temporal, la campaña queda bloqueada. No reduce el mínimo ni utiliza los resultados de validación para escoger el presupuesto.

El estado interno `early_stopped` se publica como `status=completed` y `stopping_reason=early_stop`. Alcanzar el techo publica `budget_exhausted`, incluso si la última validación también agota la paciencia. Una pausa conserva el cursor y los contadores, pero no acredita una meseta.

La recuperación y la auditoría comprueban el cursor de la última validación completa, el número de evaluaciones y la paciencia desde el mínimo o desde la última mejora. La campaña contrasta el recibo con el último checkpoint confirmado y sus hashes. La retención sigue limitada al mejor estado y los dos recientes.

Los informes registran las transiciones y actualizaciones efectivas. Esta edición admite presupuestos adaptativos y no presupone igualdad de actualizaciones entre variantes. La selección utiliza validación, la auditoría se abre después de congelarla y el test final sigue cerrado. Una meseta bajo este criterio no demuestra convergencia a un óptimo global.
