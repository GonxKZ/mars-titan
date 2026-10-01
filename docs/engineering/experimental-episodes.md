# Episodios y mundos sintéticos

Los episodios identifican una fuente, un bloque cronológico, el calentamiento y el
límite de maduración de etiquetas. Mantienen juntas todas las empresas de cada
cohorte. Cada bloque se reinicia y no se concatena con otro como si ambos formasen
una historia continua. `EpisodePredictionEnv` reutiliza las 21 acciones y la
recompensa de error absoluto normalizado del entorno existente. Su estado añade la
identidad del episodio a la cola de etiquetas y al cursor recuperable.

El remuestreo solo admite entrenamiento. Cada época recorre todas las cohortes
reales y después las adicionales. El presupuesto inicial es el 25 % de sus filas,
redondeado hacia la siguiente cohorte completa. El exceso queda por debajo de una
cohorte y se debe informar como proporción observada. `paired_world` conserva los
tamaños de esas cohortes, también cuando cambia el número de empresas, de modo que
los brazos remuestreado y sintético tienen el mismo número de filas adicionales.
El entrenamiento que consume estos recorridos se registra por separado.

## Mecanismo generador

[`scenarios.json`](../../configs/episodes/scenarios.json) fija dos controles. Uno
carece de término de señal explícita. El otro añade al retorno siguiente un término
proporcional al evento que ya estaba disponible. Los retornos comparten un factor
de mercado, factores por sector ficticio y componentes individuales. La varianza
del factor de mercado depende de su estado anterior, hay saltos simétricos y la
carga de mercado cambia por regímenes.

Cada apertura parte del cierre anterior con un desplazamiento nocturno acotado.
El cierre aplica el retorno generado. Máximos y mínimos contienen apertura y
cierre. La etiqueta se recalcula como cierre siguiente dividido por apertura
siguiente, menos uno y menos la contribución conocida del factor de mercado. Esa
definición identifica un residual de un mecanismo ficticio. No acredita una
reproducción del mercado real.

Los balances cumplen activo igual a pasivo más patrimonio. Hay un saldo inicial
disponible y publicaciones posteriores cada veinte sesiones, con el retraso
configurado. Los eventos describen empresas ficticias y no contienen el resultado
futuro. Los gráficos usan únicamente la ventana anterior a la decisión. El corpus
real, sus exclusiones y los informes de la campaña activa permanecen independientes.

La configuración y las semillas definen la generación. `fit_volatility` permite
ajustar una escala ilustrativa a las etiquetas de entrenamiento de una partición.
Registra su origen, tamaño y desviación observada. No ajusta con validación ni
afirma reproducir la distribución conjunta de las modalidades.

## Codificación y almacenamiento

La codificación analítica `synthetic-analytic-v1` sirve para comprobar contratos.
Identifica sus propias dimensiones y no se presenta como MiniLM o ResNet18.
`EncodedWorld` recibe los codificadores congelados existentes y exige su versión
completa. Usa MiniLM para los eventos y ResNet18 para los gráficos de velas. Los
gráficos se codifican en lotes de hasta 64, sin dividir la cohorte lógica.

El vector de fundamentales conserva los ocho conceptos de balance y los siete
ratios de las fórmulas existentes, con sus máscaras y antigüedad. El contexto macro
simula los tipos a dos y diez años y calcula su diferencia con la fórmula del
catálogo. Los demás indicadores quedan sin observar y con máscara cero. No se les
asigna una fórmula ni un valor inventado. Los tipos pertenecen al mundo ficticio,
no son publicaciones de una fuente económica real.

Parquet guarda grupos por cohorte y archivos por bloques. El manifiesto registra
semilla, configuración, codificación, transformaciones, unidades, partición,
dimensiones y huellas. La recuperación comprueba posiciones, grupos y filas antes
de continuar. Un archivo corrupto o una transformación distinta impiden reanudar.
El lector admite selección de modalidades y mantiene como máximo dos grupos en
una caché de hasta 64 MiB.

La caché del padre guarda escalares. Su clave incluye los pesos identificados del
padre, codificación, activos ordenados y entradas efectivamente utilizadas. Cambiar
una trayectoria o un evento fuerza una predicción nueva. Cambiar únicamente una
etiqueta no convierte esa etiqueta en entrada del padre.

## Ejecución y límites

```bash
uv run --locked python scripts/generate_episode_worlds.py \
  --config configs/episodes/scenarios.json \
  --output data/interim/synthetic-scenarios-v1
```

`--resume` comprueba la identidad antes de continuar. La opción
`--reference-encoding` recibe el manifiesto de codificación de la referencia para
utilizar sus versiones congeladas. Esa ruta requiere CUDA y no cambia a CPU si no
puede admitirse. La generación analítica es una comprobación independiente.

La admisión CUDA usa `cuda:0`, excluye otras cargas de cómputo y conserva al menos
1 GiB de margen libre. El presupuesto es como máximo 6 GiB de VRAM y 12 GiB de RAM
por proceso. Un control temporal limita las llamadas que los consumidores
anteriores hacen al configurador de memoria de PyTorch. La función original se
restaura al salir, sin modificar sus archivos científicos. El presupuesto se
comprueba durante la escritura de bloques.

Las pruebas verifican causalidad, identidades contables, reconstrucción de
etiquetas, semillas, tamaños variables, maduración, recuperación, corrupción y
caché del padre. Los codificadores de las pruebas de contrato son sustitutos
controlados. Esa verificación no equivale a haber ejecutado MiniLM y ResNet18 en
CUDA ni a demostrar utilidad del aumento sintético sobre validación real.
