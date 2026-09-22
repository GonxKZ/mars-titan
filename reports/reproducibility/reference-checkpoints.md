# Recuperación de las referencias

La persistencia local confirma un estado después de escribirlo, sincronizarlo y
comprobar que admite una carga restringida. La identidad incluye el contrato que
declara el consumidor. No se acepta continuar con otros datos, optimizador o
precisión. El límite de identidad se comprueba antes de crear el directorio.

Una prueba termina un proceso mientras escribe el archivo provisional. La carga
posterior recupera el estado anterior y permite guardar otro. También se prueban
falta de espacio simulada, archivo truncado, identidad distinta y retención de los
dos estados recientes, el mejor y las referencias fijadas.

La comprobación neuronal ejecuta una GRU, guarda pesos, AdamW, cursor y generadores
de Python, NumPy y PyTorch. Después compara una actualización continua con su
repetición desde el estado guardado. Pesos y pérdida coinciden exactamente tanto
en CPU como en `cuda:0`. No basta con comparar únicamente los pesos recién cargados.

La revisión encontró que un estado serializable podía contener claves o clases
que `weights_only=True` no admite. La regresión se reprodujo y ahora ese estado
se rechaza antes de publicar el manifiesto, conservando el anterior. También se
reprodujo y corrigió una identidad que el escritor aceptaba y el lector rechazaba
por tamaño.

## Coste observado

Una GRU de 52.609 parámetros con cuatro modalidades y contexto macro genera un
archivo de 664.931 bytes, incluidos AdamW y los generadores. El primer guardado
medido tarda 18,55 ms. Cinco repeticiones tardan entre 14,81 y 16,76 ms. La carga
tarda entre 4,07 y 4,30 ms en esas repeticiones.

La medida incluye la comprobación de carga previa a publicar. No incluye capturar
el RNG ni iniciar Python. Se ejecutó con una campaña neuronal concurrente y sin
controlar la caché del sistema. No se presenta como coste máximo ni como mejora
de velocidad frente a otra versión.

Pasan 15 pruebas específicas y tres mutaciones dirigidas. La batería completa
pasa con 1.004 pruebas sin omisiones. El [recibo](../resources/reference-checkpoints-quality.json)
conserva medidas, huella del código, cobertura y CRAP. La integración del cursor
en épocas completas se comprueba con el ejecutor, no se deduce de este ensayo.
