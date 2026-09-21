# Referencias cero y Ridge

Se implementa Ridge con suma de errores cuadrados más `alpha × ||coeficientes||²`.
El intercepto no se penaliza. La media y la varianza poblacional se calculan solo
con entrenamiento mediante estadísticas por bloques en `float64`. Las columnas
numéricamente constantes mantienen escala uno.

La segunda pasada acumula el sistema regularizado en `cuda:0`, sin formar una
inversa ni almacenar todas las ventanas. Se corrige la media residual de las
características estandarizadas, que puede no ser exactamente cero por redondeo.
Un hash comprueba que las dos pasadas leen los mismos valores. Estadísticas o
sistemas no finitos producen un error explícito.

Las entradas conservan precios, noticias, gráficos, fundamentales y macro en un
orden fijo. La ventana temporal no se resume ni se elimina para favorecer el
modelo lineal. La referencia cero no aprende parámetros y se evalúa sobre las
mismas filas.

## Equivalencia y límites

Se contrastaron bloques de 1, 7 y 40 filas frente a `StandardScaler` y `Ridge` de
scikit-learn, con columnas constantes y colineales. Las predicciones coinciden
dentro de `rtol=10⁻⁸` y `atol=10⁻¹⁰`. Una prueba independiente comprueba el
intercepto cuando las entradas tienen un desplazamiento de `10¹⁶`. Los valores
que desbordan las estadísticas se rechazan, aunque los datos originales sean
finitos.

La matriz del sistema requiere memoria cuadrática en el número de características.
Por eso se limita a 4.096 características y 4.096 filas por bloque. La sonda se
limita a 64 activos y 100.000 muestras entre entrenamiento y validación. Estos
límites no acreditan rendimiento del universo completo.

El modelo se guarda como NPZ sin objetos serializados y se publica de forma
atómica sin sobrescribir archivos existentes. La restauración conserva las
predicciones. El ajuste lineal se repite desde el principio si se interrumpe
antes de guardar. No se presenta como checkpoint a mitad de una factorización.

## Ensayo sobre datos estrictos

La [sonda ejecutada](../resources/strict-ridge-probe.json) usa 50 ejemplos anteriores
a 2023 y 15 de validación de 2023, de MNST y DECK. Cada fila contiene las cuatro
modalidades y macro, con 1.660 características. Se fija `alpha=1`, sin búsqueda de
hiperparámetros y sin abrir el test final.

Ridge obtiene un MAE diagnóstico de 0,018010 y la referencia cero, 0,009620. En
esta muestra Ridge es peor. No se elimina el resultado ni se cambia la muestra
para mejorarlo. Quince filas de validación no permiten sostener una conclusión
sobre eficacia predictiva, y los errores se agregan por fila para esta comprobación,
no como resultado confirmatorio por sesión.

Los tiempos y bytes exactos están en el recibo. La GPU estuvo compartida con otra
aplicación, por lo que no se presenta la medida como capacidad máxima ni se extrapola
una campaña. Los sensores antes y después registran el estado observado.

```bash
uv run --locked --extra research --extra cuda --extra encoders \
  python -m mars_titan.reference_probe \
  --prepared data/processed/news-expansion-20260921 \
  --samples data/processed/bounded-samples-20260921-v2/US \
  --output data/interim/linear-probe-new \
  --report reports/resources/linear-probe-new.json
```

Usar destinos nuevos. El informe no puede ocupar una ruta interna de la ejecución
ni sobrescribir modelo, etiquetas o predicciones.

## Verificación

La suite habitual pasó con 431 pruebas y una integración CUDA omitida por activación
explícita. La ejecución adicional con esa integración activada falló por falta de
VRAM en el codificador, mientras otra aplicación ocupaba 6,53 GiB. Ese fallo se
conserva como límite de la verificación, no como prueba superada ni como alternativa
en CPU. Las pruebas numéricas de Ridge y su ensayo real sí ejecutaron CUDA.

La revisión independiente detectó colisiones de rutas, un problema de centrado
con valores grandes y desbordamientos. Los tres casos se reprodujeron con pruebas
fallidas antes de corregirlos. Una mutación que omite el hash entre pasadas se detectó.
En los módulos de referencias y la sonda, la cobertura combinada fue del 84,44 %.
Ridge tiene complejidad 16 y CRAP 16,164, con Radon 6.0.1 y cobertura de sentencias
por función. Son evidencias acotadas, no garantías de ausencia de errores.

#24 sigue abierta para la campaña común por ventanas y la selección cronológica de
regularización sobre una cohorte suficiente. Esta entrega prepara y mide la referencia.
