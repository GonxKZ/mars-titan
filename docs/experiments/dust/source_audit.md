# Dust como variante de aprendizaje

La fuente consultada el 23 de octubre de 2026 es [Dust: Pretraining Transformers Without
Backpropagation](https://qlabs.sh/research/dust/), de Samip Dahal, Bishwas Mandal, Serdar
Gülbahar y Akshay Vegesna. La página indica octubre de 2026. El código mínimo se fija en
[qlabs-eng/dust, b20f7c03eac630b6441ba6f254128c1761dc45ad](https://github.com/qlabs-eng/dust/tree/b20f7c03eac630b6441ba6f254128c1761dc45ad).
Se conserva [su licencia MIT](../../licenses/dust-mit.txt).

Dust perturba activaciones, obtiene señales de diferencias de pérdida y reconstruye
gradientes de parámetros mediante productos exteriores locales. La atención usa una señal
en su salida y perturbaciones locales de consultas, claves y valores. El código mínimo
vuelve a ejecutar el modelo completo en sus probes directos. No se le atribuye una
reutilización de prefijos que esa implementación no realiza.

La adaptación financiera es una variante experimental no canónica. Su compatibilidad
inicial comprende el predictor `transformer_direct` completo, con las cinco entradas,
máscaras, cabeza escalar y todos sus parámetros entrenables. Incluye sesgos, LayerNorm
afín y QKV empaquetadas. Rechaza MAC, ampliaciones, cuantiles y ajustes parciales con
parámetros congelados. No cambia la arquitectura ni el estimador por defecto.

La pérdida por fila puede ser MAE, MSE o Huber. Se normaliza una sola vez sobre las filas
financieras. La adaptación usa centrado sobre la población completa con corrección
`P/(P-1)` y crédito de atención con `gamma=1`. Estas decisiones difieren de las recetas
lingüísticas publicadas y forman parte de su identidad. Una fila con varios probes sigue
siendo una sola observación. El estimador no ejecuta `backward`, no modifica pesos o
`.grad` y deja el paso del optimizador al llamante.

Los resultados de Noise requieren otra lectura. En el [README anterior al anuncio del
22 de octubre](https://github.com/qlabs-eng/noise/blob/45b771df4f8f63e9254ae33ad36848dcb28a44ef/README.md),
el registro rápido de 10M declara 1,58 segundos y pérdida 4,1859. El registro de mejor
pérdida declara 3,6758 y 2.100,6 segundos. Son ejecuciones distintas. La receta rápida
utiliza ocho H100 y excluye preparación, compilación y validación del cronómetro.
Estos resultados no establecen una ventaja financiera ni el coste total en una RTX 4070.

Las comparaciones locales deben mantener las mismas ventanas, filas, entradas e
inicializaciones. BP y Dust se comparan con el mismo optimizador y presupuesto de
actualizaciones. SGD y AdamW son controles separados. Los experimentos de desarrollo
utilizan segmentos de meses y, como máximo, un año total de ajuste y validación.
El test final no interviene. No hay en esta auditoría una mejora predictiva medida.
