# Guía de implementación y límites tecnológicos

La propuesta de MARS-TITAN gobierna el trabajo: preparar FinMultiTime con disponibilidad temporal, predecir retornos residuales, comparar memoria adaptativa con referencias y analizar resultados y recursos. La biblioteca amplia sirve para elegir y descartar, no para convertir todos sus mecanismos en requisitos de implementación.

Las issues contienen la receta de cada tarea. Su sección «Guía de ejecución» concreta herramientas, entradas, pasos, artefactos, comprobaciones y límites. Los archivos previstos se crearán al implementar la tarea. Que aparezca una orden de pruebas no significa que esas pruebas existan o hayan pasado.

## Dónde utilizar cada tecnología

| Responsabilidad | Elección inicial y uso concreto | Frontera que se mantiene |
| --- | --- | --- |
| Inventario y adquisición | Python estándar con uv para manifiestos, hashes, peticiones acotadas y validación. | No cargar archivos completos de gran tamaño ni convertir una descarga actual en evidencia histórica. |
| Tablas experimentales | Polars con ejecución diferida y PyArrow para bloques Parquet. DuckDB cuando facilite joins temporales o auditorías SQL. | No mantener tres copias equivalentes en pandas, Polars y DuckDB. Medir materializaciones e índices. |
| Etiquetas y referencias numéricas | NumPy y SciPy para retornos, ajuste lineal, cuantiles y comprobaciones de precisión. | El retorno futuro de mercado forma parte de la etiqueta, nunca de las entradas. |
| Referencias clásicas | scikit-learn para Ridge y HistGradientBoosting, con dispositivo CPU declarado y validación externa temporal. | No interpretar warm_start como aprendizaje incremental ni activar particiones aleatorias ocultas. |
| Ampliación de boosting | XGBoost con CUDA o memoria externa solo si una decisión de cobertura o coste lo justifica. | La memoria externa también consume RAM, caché y transferencia. No garantiza cabida en el equipo. |
| Codificadores y modelos | PyTorch en cuda:0 para codificación compacta, GRU, memoria, cabeza de cuantiles y entrenamiento. | Compartir parámetros entre activos y separar parámetros, estado persistente y estado de consulta. |
| Texto y gráficos | Codificadores congelados y cachés por contenido, revisión y regla temporal. Una biblioteca adicional se fija cuando se elija el modelo. | No entrenar un modelo fundacional ni ocultar conocimiento retrospectivo del preentrenamiento. |
| Evaluación | Python, NumPy, SciPy y lectura columnar para métricas por sesión, simulación, bootstrap y agregaciones. | No usar residuos como beneficios ni tratar semillas como mercados independientes. |
| Código nativo | C17/C++20 con CMake y CUDA solo para un cuello medido que no resuelvan las operaciones existentes. | Conservar referencia PyTorch, precisión, gradientes, errores y medida de mejora real. |
| Seguimiento y web | Exportador Python sin GPU, instantáneas JSON permitidas y HTML/CSS/JavaScript estático para GitHub Pages. | No consultar tensores CUDA desde el navegador ni acoplar la publicación al paso de entrenamiento. |
| Go | No forma parte del núcleo ni de la web estática inicial. | Considerarlo solo para un auxiliar de E/S identificado y medido que compense otro lenguaje y otra cadena de compilación. |

Las versiones se fijan mediante uv.lock al incorporar dependencias. Una librería citada en una issue no se instala por adelantado si la variante sigue descartada o pendiente. C++ y CUDA propios no son una condición para que el proyecto sea eficiente.

## Núcleo, apoyo y extensiones

El núcleo conserva auditoría de precios, texto, fundamentales y gráficos. Empezar la comparación con precios y texto no permite omitir silenciosamente las otras modalidades. Cada integración o exclusión necesita evidencia. El objetivo principal es residual de mercado y el sector se añade solo con pertenencia y factores históricos defendibles.

Las referencias mínimas son cero/media histórica, Ridge, boosting, GRU y una memoria neural asociativa identificable. Las ablaciones comparan ausencia de memoria, escritura uniforme, error maduro y sorpresa completa, con régimen e incertidumbre. La sorpresa completa incluye anomalía de mercado y relevancia económica. La diversidad puede añadirse, pero no sustituye esos términos.

La comparación principal usa K = 1 e incluye las cuatro modalidades y el contexto macroeconómico. Más pasos de lectura, replay paramétrico, destilación, familias secuenciales adicionales, China y optimizaciones nativas requieren una pregunta, datos adecuados y presupuesto medido. El [plan por familia](full-dataset-training.md) explica cómo recorrer todos los registros elegibles y qué estimadores no ofrecen aprendizaje por bloques. La [campaña de referencias](comparison-campaign.md) concreta la ampliación al corpus admisible. «Todo el dataset» nunca incluye entrenar con el test reservado.

Las tareas de apoyo, como entorno, checkpoints, ejecución, versiones y redacción técnica, siguen siendo necesarias. Una prioridad P1 no convierte automáticamente una tarea en opcional. La etiqueta `opcional` identifica extensiones prescindibles para responder a los seis objetivos. El observatorio presenta el trabajo y no sustituye sus resultados.

## Orden de congelación

1. MT-030 fija reloj, objetivo, regla de selección, particiones, reserva final, métricas y método estadístico antes de comparar resultados. También predefine bloques de remuestreo y tratamiento de las semillas que MT-036 aplicará después.
2. MT-012 publica la regla y el piloto. MT-060 empieza a medir desde los ensayos mínimos, sin esperar a toda la campaña para estimar coste.
3. MT-034 congela el manifiesto principal y la receta de los finalistas usando solo desarrollo y recursos medidos. No selecciona activos o fechas por su resultado en el test.
4. MT-035 ajusta y calibra según esa receta, sella pesos y calibrador y ejecuta el test una vez por configuración y semilla comprometidas. La adaptación de estado sigue una política ya fijada.

Los factores FF3/FF5 y retornos en exceso de la revisión financiera son antecedentes y posibles sensibilidades. No reemplazan el objetivo apertura-cierre con proxy de mercado del protocolo sin registrar otro experimento. El contrato canónico usa `event_at`, `available_at`, `prediction_at`, `entry_at`, `exit_at` y `label_available_at`.

## Principios y comprobaciones

SOLID, DRY, KISS, YAGNI, Open/Closed, Dependency Inversion, composición, separación de responsabilidades, Fail Fast y Measure First guían las decisiones. La arquitectura debe expresar datos, objetivos, memoria, modelos, entrenamiento y evaluación, no una colección de capas vacías. Una abstracción necesita una variación real que proteger. Minimizar líneas no justifica ocultar errores ni reducir legibilidad.

Las pruebas unitarias cubren invariantes y casos límite. Las de integración comprueban contratos entre componentes, archivos y órdenes reales. Las pruebas temporales alteran información futura y verifican que no cambien predicciones anteriores. La recuperación compara una ejecución continua con otra interrumpida. El código nativo exige concordancia numérica con su referencia.

La mutación debe demostrar que las pruebas detectan cambios relevantes, como permitir una etiqueta inmadura, filtrar un test sellado, alterar costes o ignorar un hash. Se distingue una selección dirigida de mutantes de una campaña completa de una herramienta como [mutmut](https://mutmut.readthedocs.io/en/latest/). Registrar mutantes ejecutados, muertos, supervivientes, errores y alcance, sin inferir una cobertura universal a partir de unas muestras.

Cobertura y CRAP localizan funciones complejas poco comprobadas. El informe indicará herramienta, revisión y convención de cobertura. No se fija un umbral arbitrario ni se cambia código correcto solo para optimizar una cifra. La calidad también depende de invariantes, revisión, datos y comportamiento observado.

## Trabajo con GitHub

Antes de una modificación se consulta o crea la issue correspondiente, se añade al Project si falta y se mueve a En curso. Se trabaja en una rama breve asociada, como `docs/mt-069-execution-guides` o `feat/mt-068-observatory`. Las ramas paralelas se aíslan cuando lo necesitan y no comparten modificaciones sin coordinación.

Los commits son atómicos, en inglés y con Conventional Commits. Se enlaza la issue en el commit o la revisión. La evidencia preparada permite pasar a En revisión. Tras verificar la integración, el artefacto y sus límites, se cierra la tarea y se mueve a Hecho. Los estados registran trabajo real, no una cadencia simulada.

Pruebas y entrenamientos se ejecutan localmente. La única excepción autorizada de Actions es desplegar GitHub Pages. La publicación estática no valida por sí misma el código ni autoriza difundir archivos del entrenamiento. El repositorio es público, mientras datos, biblioteca de PDF, pesos y registros completos permanecen excluidos de Git.
