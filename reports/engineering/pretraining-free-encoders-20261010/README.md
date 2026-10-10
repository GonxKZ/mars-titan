# Codificador sin preentrenamiento posterior: comprobación con datos reales y mutación

Recibos del 10 de octubre de 2026 para [#10](https://github.com/GonxKZ/mars-titan/issues/10).
No se ha ajustado ni evaluado ningún modelo de la campaña, no se ha usado la GPU y no se ha
ejecutado ningún paso de optimizador. El control, la sensibilidad declarada y la procedencia
de MiniLM y ResNet18 están en la [auditoría de los codificadores](../../../docs/data/pretraining-audit.md).
La sensibilidad queda pendiente de la edición de control y de los ajustes de la campaña.

## Comprobación con datos reales

[`real-data.json`](real-data.json) es el recibo de `benchmarks/pretraining_free_encoders_real.py`
sobre la edición preparada v3.1 desde 2000, con el código del commit `6af4a9c4`. Los commits
posteriores añaden pruebas y documentación y, en el último, cambian el vocabulario de
docstrings, mensajes y una línea de texto de `encoder_sensitivity.py` y del propio script. La
huella del codificador coincide con la del commit final, pero las de esos dos archivos ya no.
La misma corrección se aplicó a mano a la lista de tareas pendientes del recibo, sin tocar
ninguna cifra.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run --no-sync python \
  benchmarks/pretraining_free_encoders_real.py --prepared <edición preparada v3.1> \
  --output <recibo nuevo>.json --weights-history --scratch <carpeta temporal>
```

- **Noticias.** 2.000 textos reales de 40 activos de US (9,1 millones de caracteres) y 2.000
  de 40 activos de CN (176.622 caracteres), elegidos con semilla fija. Dos instancias
  distintas dan vectores idénticos bit a bit, todos finitos y con norma 1 (error máximo de
  5·10⁻⁸).
- **Gráficos.** 927 gráficos de US y 952 de CN dibujados con `charts.chart_png` sobre 64 filas
  de precios reales. Otros 73 y 48 intentos se descartaron porque el dibujo rechaza velas
  incoherentes sin la tolerancia de la auditoría de precios, que la canalización sí aplica.
  Los vectores coinciden bit a bit entre instancias y quedan entre 0 y 1.
- **Pesos de MiniLM.** Se descargó el `pytorch_model.bin` del commit `e62509716f` del 23 de
  junio de 2021 (SHA-256 `16cc9e54df6e0832…ac59`). Sus 200 tensores son idénticos a los del
  `model.safetensors` fijado (`eaa086f0ffee582a…917b`), sin tensores de más ni de menos. La
  descarga se borró al terminar.

## Coste

| Medida | US | CN |
| --- | ---: | ---: |
| Texto, por noticia | 4,68 ms | 0,36 ms |
| Texto, por mil caracteres | 1,02 ms | 4,02 ms |
| Gráfico, codificador | 1,93 ms | 2,02 ms |
| Gráfico, dibujo (lo paga cualquier edición) | 2,23 ms | 2,33 ms |
| Noticias de la edición | 2.156.832 | 553.392 |
| Filas de precios de la edición | 16.149.645 | 2.895.378 |
| Horas de CPU del texto para toda la edición | 2,8 h | 0,05 h |
| Cota de horas de CPU de los gráficos | 8,6 h | 1,6 h |

Mejor de tres repeticiones en CPU AMD Ryzen 9 8945HS, un solo proceso, dos hilos y una carga
media del equipo de 18,9 por otros procesos al empezar, así que las cifras solo ordenan el
coste. Las noticias de US son artículos largos (4.570 caracteres de media) y las de CN
resúmenes. La cota de los gráficos cuenta una ventana por cada fila de precios, más de las
que tienen muestra. El resto de una edición de control (materialización, dibujo de gráficos,
fundamentales y macro) cuesta lo mismo que en la edición congelada y no se ha medido aquí. La
sensibilidad con la forma de US (4.955 sesiones, dos modelos, tres semillas y 2.000 réplicas)
tardó entre 0,72 y 1,04 s. El pico de RSS del proceso fue de 2.172 MiB, sobre todo por cargar
los pesos de MiniLM para compararlos. No se ha medido la energía por falta de instrumento.

## Mutación dirigida

[`mutations.json`](mutations.json) recoge 39 mutaciones aplicadas de una en una sobre una
copia aislada de `src`, `tests` y `configs` en el commit `37523a0e`, con las pruebas importando
la copia. Cada entrada guarda el fragmento original y el mutado. Las 39 hicieron fallar
`tests/data/test_pretraining_free_encoders.py` o `tests/evaluation/test_encoder_sensitivity.py`,
con la copia sin mutar en verde (57 pruebas):

- En el texto: n-gramas desde un carácter, hashing sin signo, norma L1, sin NFKC, sin pasar a
  minúsculas y admitir textos en blanco.
- En los gráficos: el valor del canal en lugar de la tinta, los canales verde y azul, bloques
  por columnas, no comprobar el tamaño y lotes sin límite.
- En la identidad: una entrada de control sin gráfico, marcarlo como no histórico y TF32 desconocido.
- En la codificación: ignorar la opción, admitir las pasadas de GPU y pasar opciones CUDA.
- En la sensibilidad: un tramo posterior el doble de largo, los signos del salto, del placebo y
  de Δ, una media de semillas mal dividida, conservar las filas calibradas, admitir sesiones
  sin todas sus semillas, no comparar recuentos de filas, admitir la misma edición, ignorar la
  huella de la tabla, el test final y que los modelos compartan sesiones, no exigir el mínimo al
  tramo placebo, tratar el cero como positivo, confirmar con un solo intervalo, intervalos
  marginales, otra semilla, un bloque tan largo como la serie, un corte anterior a la
  publicación o fuera del día 1, pocas réplicas y sobrescribir el informe.

Una primera pasada en el commit `aa4dce8a` dejó vivas dos: no exigir el mínimo de sesiones al
tramo placebo y tratar un extremo igual a cero como positivo. El commit `37523a0e` añade una
prueba para cada una y la segunda pasada las detecta todas. La mutación cubre los fragmentos
elegidos, no garantiza que cualquier otro cambio se detecte.
