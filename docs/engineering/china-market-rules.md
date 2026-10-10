# Reglas de negociación de acciones A en la simulación

La simulación financiera ejecuta las órdenes en la apertura siguiente con lotes de una acción, sin límites diarios de precio y con un coste único en puntos básicos. En China ese supuesto sobrestima lo que puede ejecutarse. `simulation/market_rules.py` declara, para cada activo de Shanghái o Shenzhen, las reglas que he podido comprobar en fuentes primarias entre 2006 y 2023. `china_a_share_instrument(asset)` devuelve un `Instrument` con identidad `cn_a_share_v1_main`, `cn_a_share_v1_chinext` o `cn_a_share_v1_star`, y `FinancialEnv(..., instruments=...)` las aplica en el motor Python y en el nativo.

Las reglas son una identidad nueva del entorno (`instruments_sha256`). Sin `instruments`, la contabilidad, las identidades y las trayectorias no cambian. El motor nativo añade además el contrato con el que las aplica, como se explica en [Motor nativo](#motor-nativo).

## Tablero

El tablero se deduce del código de seis cifras y de la bolsa (`.SS` o `.SH` para Shanghái y `.SZ` para Shenzhen). En la población preparada desde 2000 hay 449 activos del tablero principal de Shanghái, 282 del principal de Shenzhen, 59 de ChiNext y 20 de STAR. Las acciones B, la bolsa de Pekín y cualquier prefijo sin reglas comprobadas se rechazan.

| Prefijos | Tablero |
| --- | --- |
| 600, 601, 603 y 605 (Shanghái), 000, 001, 002 y 003 (Shenzhen) | Principal |
| 300 y 301 (Shenzhen) | ChiNext |
| 688 y 689 (Shanghái) | STAR |

## Reglas implementadas

**T+1.** Las acciones compradas no pueden venderse antes de la liquidación del día siguiente. Lo establecen los artículos 3.1.4 y 3.1.5 de las reglas de negociación de Shanghái y Shenzhen de [2006](https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20230418_5720136.shtml) y de 2023. Ningún artículo dice literalmente «T+1 para acciones A». La lectura sale de esos artículos y del ciclo de liquidación. La simulación lo respeta por construcción. Hay una ejecución por sesión, en la apertura posterior al cierre de decisión, y `Portfolio.advance` rechaza una segunda ejecución antes del siguiente cierre.

**Lotes y resto impar.** En el tablero principal y en ChiNext las compras se hacen en múltiplos de 100 acciones y el resto inferior a 100 se vende en una sola orden. Así lo fijan el artículo 3.4.7 de las reglas de Shanghái de 2006, el 3.3.8 de las de 2023 y el 3.3.8 de las de Shenzhen de 2006 y 2023. La reforma de ChiNext de 2020 (深证上〔2020〕515号) conservó el lote. En STAR cada orden necesita al menos 200 acciones y el saldo inferior a 200 se vende de una vez. Lo fijan el artículo 20 de las [reglas especiales de 2019](https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/10118601/files/f6fc4a1d4c1f469183a013c4dc36a535.pdf) (上证发〔2019〕23号) y el 6.1.7 de las reglas de Shanghái de 2023. El incremento de una acción por encima de 200 no aparece en esas normas. Solo lo recoge una [respuesta de formación al inversor de la bolsa de Shanghái](https://edu.sse.com.cn/tib/qa/c/4866268.shtml) del 19 de julio de 2019.

La venta admitida es la mayor que no supera la deseada: la posición completa, un múltiplo del lote que respete el mínimo o el resto impar entero más lotes completos. Un saldo de 150 acciones del tablero principal permite vender 50, 100 o 150, pero no 130.

**Límites diarios de precio.** El límite es el cierre anterior por 1 ± la banda, redondeado por la mitad hacia arriba a 0,01 yuanes (artículos 3.4.11 y 3.4.13 de Shanghái 2006, 3.3.11, 3.3.13 y 3.3.17 de Shanghái 2023, 3.3.14 de Shenzhen 2006 y 3.3.14 y 3.3.19 de Shenzhen 2023). En días de exdividendo o exderecho la base es el precio de referencia ajustado (artículo 4.3.3). La simulación resta el dividendo por acción y divide por el factor del split de esa apertura.

| Tablero | Banda | Vigencia aplicada | Fuente |
| --- | --- | --- | --- |
| Principal | 10 % | 2006-07-01 a 2023-12-31 | Reglas de 2006 y 2023 de ambas bolsas |
| ChiNext | 10 % | 2006-07-01 a 2020-08-23 | Reglas de Shenzhen de 2006 |
| ChiNext | 20 % | Desde 2020-08-24 | [深证上〔2020〕515号](http://www.szse.cn/lawrules/rule/repeal/rules/P020231230545310237980.pdf), artículo 2.1 |
| STAR | 20 % | Desde 2019-07-22 | 上证发〔2019〕23号, artículo 18, y [anuncio del inicio de negociación](http://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20190705_4858654.shtml) |

Las reglas de 2006 entraron en vigor el 1 de julio de 2006. La banda del 10 % existía antes, pero su inicio en 1996 solo consta en fuentes secundarias y no se aplica a fechas anteriores.

Una orden de compra no se ejecuta si la apertura está en el límite superior o por encima, y una venta no se ejecuta si está en el límite inferior o por debajo. Es un supuesto conservador. En el mercado puede haber ejecuciones parciales en el límite, pero una barra diaria no indica la cola de órdenes. La orden queda pendiente con el motivo `limit_up` o `limit_down`.

**Impuesto de timbre.** Se cobra sobre el efectivo negociado, además de `cost_bps`, y se reserva al dimensionar las compras.

| Desde | Tipo | Fuente |
| --- | --- | --- |
| 2005-01-24 | 1 ‰ comprador y vendedor | [财税[2005]11号](http://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/2005zcfb/200805/t20080524_34819.htm) |
| 2007-05-30 | 3 ‰ comprador y vendedor | [财税〔2007〕84号](https://www.chinatax.gov.cn/chinatax/n810341/n810765/n812176/200705/c1194505/content.html) |
| 2008-04-24 | 1 ‰ comprador y vendedor | [Ministerio de Hacienda, 23-4-2008](http://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200805/t20080519_29133.htm) |
| 2008-09-19 | 1 ‰ solo vendedor | [Ministerio de Hacienda, 19-9-2008](http://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm). La [Ley del impuesto de timbre](https://shanghai.chinatax.gov.cn/gate/big5/shanghai.chinatax.gov.cn/zcfw/zcfgk/yhs/202106/t458595.html) lo mantiene desde el 1-7-2022 |
| 2023-08-28 | 0,5 ‰ solo vendedor | [财政部 税务总局公告2023年第39号](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html) |

Las fechas se interpretan como días de Pekín. El anuncio de 2023 dice «reducir a la mitad», y el 0,5 ‰ resulta de aplicarlo al 1 ‰ de la ley. Los dos comunicados de 2008 no muestran número de documento.

La liquidación final hipotética de una evaluación también paga el timbre de venta vigente en la fecha del último cierre, además de `cost_bps`, en `evaluate` y en `liquidated_nav` de la sesión C++, con paridad exacta entre ambos (`test_terminal_liquidation_pays_the_sell_tax_in_force_at_the_last_close` y la prueba C++ homónima).

## Estado de cotización

Las bandas del tablero no bastan para una acción con advertencia de riesgo, sin reforma accionarial o recién admitida. Desde #365 la cinta china lleva también el estado acreditado de cada activo. `simulation/listing_status.py` define el contrato de una tabla privada, identificada por la huella de sus bytes (`listing_status_sha256` en las [políticas](../../configs/simulation/historical-masked-rl-policies.json)), y `china_a_share_instrument(asset, status)` corta los periodos del tablero en cada cambio de estado. Las fechas se interpretan como días de Pekín.

| Estado | Regla aplicada | Origen de las fechas |
| --- | --- | --- |
| Advertencia de riesgo (ST o *ST) | Banda del 5 % en el tablero principal. ChiNext no tuvo advertencias antes de su reforma de 2020 y desde ella aplica el 20 % también a esas acciones, como STAR | Shenzhen: informe oficial de cambios de nombre abreviado. Shanghái: anuncios de implantación, cambio y retirada en CNINFO |
| Sin reforma accionarial (prefijo S) | Banda del 5 % en el tablero principal, como confirman los anuncios de esas acciones (por ejemplo el 临2013-008 de 600733) y sus precios | Shenzhen: el mismo informe de nombres. Shanghái: desde 2010 hasta el día de la reforma |
| Primer día tras la reforma o tras reanudar la cotización | Sin banda ese día | Texto del anuncio de ejecución de la reforma o de reanudación, que debe decir «不设涨跌幅限制» o una variante |
| Cinco primeras sesiones tras la salida a bolsa | Sin banda en STAR desde 2019-07-22, ChiNext desde 2020-08-24 y el principal desde 2023-04-10 | Fecha de admisión de las listas oficiales de ambas bolsas |

Fuera de estos estados se aplica la banda del tablero. La tabla empieza el 1 de enero de 2010 (`COVERAGE_FROM`), porque las cintas chinas empiezan en 2011, y Python y C++ rechazan una cinta china anterior. Un estado que ya estaba vigente en esa fecha empieza en ella. Ninguna fecha posterior al 31 de diciembre de 2023 entra en la tabla, y un estado vigente en el corte queda abierto, de modo que la tabla no revela nada de 2024.

### Construcción

`data/china_listing_status.py` construye la tabla y su informe solo con capturas ya guardadas, cada una con su recibo y su huella, sin hacer ninguna petición. Shenzhen publica el historial de nombres abreviados y de él salen sus tramos ST y S. Shanghái no lo publica, así que sus tramos ST se leen de los anuncios de CNINFO. El extractor clasifica cada título (implantación, retirada, cambio, continuación o reanudación), localiza en el texto la fecha efectiva dentro de la ventana de publicación y recorre los anuncios de cada código con una máquina de estados. El nombre que CNINFO asocia a cada anuncio en su fecha de publicación sirve de observación independiente. Una retirada sin inicio previo, un nombre que contradice el estado o un anuncio de estado sin texto o sin fecha desde 2010 detienen la construcción en lugar de producir una tabla dudosa.

La comprobación principal aplica el mismo extractor a los anuncios de Shenzhen y compara el resultado, día a día, con el informe oficial. Entre 2010 y 2023 coinciden los 24.753 días con advertencia y no hay ningún día que aparezca solo en una de las dos fuentes. Los días sin límite tras la reforma que salen de los anuncios de 000750 y 000776 coinciden con el fin oficial de su prefijo S. La segunda comprobación contrasta la tabla con los precios sin ajustar. En pares de sesiones consecutivas del calendario, ambas verificadas, con negociación y sin acción corporativa, solo 4 de los 42.320 cierres con banda del 5 % la superan. Tres lo hacen por entre 2 y 4 céntimos y el cuarto, 600733 el 31 de julio de 2015, cae un 5,65 % sin explicación en las fuentes capturadas. Con la banda del 10 % la superan 178 de 2.090.717 cierres, que no dependen de la tabla.

La tabla resultante tiene 810 activos con su fecha de admisión, 127 tramos ST en 107 activos (64 de Shanghái y 63 de Shenzhen, 9 todavía abiertos en el corte), seis acciones con tramo S (000750, 000776, 600688, 600705, 600733 y 600871), 17 activos con un día sin límite tras su reforma o reanudación y 27 con exención tras la salida a bolsa. Procede de 877 fuentes y 419 anuncios leídos. Las salidas con precio quedan vacías, porque ninguna fuente capturada fija un importe de salida para los activos de la edición ([integridad del entorno](rl-environment-integrity.md#bajas-y-universo-por-fecha)).

El efecto sobre la etapa es pequeño. Con los 128 primeros activos por efectivo negociado mediano de cada año entre 2011 y 2023, sin exigir predicciones (una cota superior del universo real), hay 835 sesiones con banda reducida, todas entre 2016 y 2019 y en cinco activos (000410, 000831, 600150, 600518 y 601918). Ningún día sin límite ni ninguna exención inicial cae dentro de esos universos. Una salida a bolsa o una reanudación tras una suspensión larga no pueden entrar, porque la clasificación exige negociación verificada en más de la mitad de las 252 sesiones anteriores, y las seis acciones sin reforma no llegan a ellos.

## Motor nativo

La misma semántica está implementada en C++20 para la cartera nativa (`NativePortfolio` y `FinancialEnv(backend="native")`) y para la sesión C++ que usan PPO y KLPO nativos. El cálculo vive en el núcleo existente, `native/src/simulation.cpp`, sin capas nuevas.

- **Contrato binario.** `mt_simulation_step_v2` recibe, además de los argumentos de v1, una fila `mt_rules_v1` por activo con el mínimo por orden, el resto impar, la banda vigente, el cierre de referencia y el timbre de compra y de venta de esa apertura. Con `rules = NULL` reproduce v1 byte a byte. Las órdenes bloqueadas devuelven los motivos `MT_ORDER_LIMIT_UP` y `MT_ORDER_LIMIT_DOWN`, que la cartera traduce a `limit_up` y `limit_down`.
- **Redondeo de los límites.** Python calcula `Decimal(repr(referencia)) * (1 ± Decimal(repr(banda)))` y redondea a 0,01 por la mitad hacia arriba. C++ toma la representación decimal más corta del `double` con `std::to_chars`, que coincide con `repr`, multiplica de forma exacta con enteros de 128 bits y redondea igual. El producto cabe en las 28 cifras del contexto decimal de Python si la banda tiene como mucho diez decimales. Una banda con más decimales o un límite por encima de 2^53 céntimos se rechazan de forma explícita. `mt_simulation_price_limits_v1` expone el cálculo para contrastarlo. En cada paso, el núcleo solo calcula el límite exacto cuando la apertura está cerca del producto binario de referencia y banda. Fuera de esa franja, la cota del error de redondeo asegura la misma decisión, como se explica con sus medidas en la [simulación C++](native-financial-simulation.md#reglas-de-mercado).
- **Precio de referencia.** La cartera nativa usa el último cierre. Las sesiones con eventos o cobros siguen pasando por la contabilidad Python, como antes. La sesión C++ resta los dividendos y divide por los splits de la apertura, en ese orden, igual que `Portfolio._references`.
- **T+1.** Se conserva por construcción. Hay una ejecución por sesión, el núcleo rechaza una apertura que no sea posterior al cierre anterior y cada activo tiene una sola dirección neta por ejecución. Las ventas solo usan la posición que había al cierre de decisión.
- **Cintas para la sesión C++.** `write_tape(tape, destino, instruments=...)` escribe la identidad de cada `Instrument` en un manifiesto de versión 2. El lector C++ exige que la versión y las reglas aparezcan juntas, de modo que un ejecutable anterior rechaza la cinta en lugar de ignorar sus reglas. PPO exige además las mismas reglas en entrenamiento y validación. La cinta reconstruida real también se admite en `mars-titan-sim`, `mars-titan-ppo` (esquema 4) y `mars-titan-klpo`, cuyo lector compara las reglas declaradas con la tabla acreditada de cada tablero ([políticas nativas sobre cintas reconstruidas](native-policy-real-tapes.md)). Ningún ajuste se ha ejecutado todavía sobre ella.
- **Estado de cotización.** El lector C++ de la cinta reconstruida lee la misma tabla, exige los cinco campos de cada activo chino y la cobertura desde 2010, y deriva las bandas con el mismo algoritmo que `status_price_limits`. `test_listing_status_bands_block_and_free_orders_in_both_engines` recorre en ambos motores una orden bloqueada por la banda del 5 % que la del 10 % habría ejecutado y una compra en un día sin límite.
- **Identidad.** Un entorno nativo con reglas añade `native_market_rules = "mt_simulation_step_v2/mt_rules_v1"` y los recibos de `mars-titan-sim` añaden `market_rules` con el mismo valor. Sin reglas, la llamada, la identidad y la trayectoria US son las anteriores.

### Paridad y comprobaciones

`tests/simulation/test_native_china_rules.py` compara, paso a paso, observación, recompensa, final, `info` y estado completo de la cartera entre ambos motores. Usa tres tipos de cinta sintética, identificada como tal. Una se construye desde una edición con el formato real (aperturas en los límites con el residuo de la reconstrucción, fuera de rejilla, suspensiones, filas ausentes, split con resto impar, dividendo y evento ambiguo). Otra recorre los cambios de timbre de 2008 y 2023, el inicio de STAR y la reforma de ChiNext, con aperturas exactamente en el límite, dentro, por fuera y ausentes. La tercera repite esa cinta en la sesión C++ mediante `mars-titan-sim` y compara sus métricas exactas con `evaluate`. Las 30 trayectorias de las diez cintas fechadas incluyen compras, ventas, bloqueos al alza y a la baja, aperturas ausentes, órdenes limitadas por efectivo o lote y ventas del resto impar, de modo que la paridad no es vacía. El cálculo de los límites coincide con `Instrument.limits` en 18.009 precios por banda, con seis bandas, incluidos los casos de medio céntimo.

Con la edición real declarada, `test_reconstructed_tape_edition.py` recorre la cinta CN de 2023 con cinco activos y tres planes de acciones fijas en ambos motores. Los 241 pasos de cada plan coinciden. En dos de ellos hay órdenes bloqueadas al alza.

La ruta US se ha contrastado con la biblioteca y el código anteriores. Once trayectorias nativas (mundos sintéticos y cinta US reconstruida, 1.174 pasos) dan los mismos bytes de observación, recompensas, `info`, estado e identidad salvo la huella de la biblioteca, y quince recibos de `mars-titan-sim` coinciden salvo las huellas nativas.

He introducido 24 defectos dirigidos en bandas por fecha, lote de compra, mínimo, resto impar, timbres, reserva del timbre de compra, referencia ajustada por dividendos y splits, T+1, redondeo por la mitad, igualdad en el límite, lectura del exponente y decisión acotada. Las pruebas Python y CTest detectan los 24. Solo sobrevive una mutación equivalente, que mueve el umbral de la decisión acotada dentro de su propio margen. CTest pasa también con ASan y UBSan, y libFuzzer ejecutó 6,8 millones de entradas del núcleo con reglas opcionales sin fallos.

## Lo que falta

Solo se implementa lo que tiene una regla y unos datos fechados. La revisión del 10 de octubre de 2026 incorporó el [estado de cotización](#estado-de-cotización) con fuentes oficiales. Lo demás queda como simplificación declarada, con el efecto que cabe esperar sobre los resultados.

- **Primer día de una salida a bolsa del tablero principal antes de 2023.** Antes del registro, el primer día del tablero principal no tenía la banda habitual (sin banda diaria hasta 2013 y, desde 2014, con un tope del 44 % sobre el precio de emisión que limitaba las órdenes y no la variación). La tabla no lo modela y la cinta aplica la banda desde la segunda sesión con cierre anterior. No afecta a la etapa, porque esos días nunca entran en un universo por la regla de clasificación.
- **Estado anterior a 2010.** La tabla no cubre fechas anteriores y las cintas chinas que empiezan antes se rechazan.
- **Retenciones sobre dividendos.** Desde el 8 de septiembre de 2015, el dividendo que cobra un inversor individual tributa según el tiempo de tenencia: el 20 % hasta un mes, el 10 % entre un mes y un año y nada por encima de un año ([财税〔2015〕101号](http://m.mof.gov.cn/czxw/201509/t20150907_1452683.htm)). Con tenencias de hasta un año, la sociedad no retiene al pagar y el impuesto se calcula y se descuenta cuando el inversor vende, según el tiempo que mantuvo las acciones. Los periodos anteriores tenían otras reglas (财税〔2012〕85号 desde 2013) que no he revisado. No se implementa porque depende del tipo de inversor, que la simulación no declara, y exigiría lotes fechados en los dos motores. La contabilidad abona el bruto. Para un inversor individual sobrestima el rendimiento neto entre el 10 % y el 20 % de los dividendos cobrados, porque ninguna posición de un episodio anual supera el año.
- **Fechas reales de pago.** La edición solo guarda la fecha ex. El plazo de pago sigue siendo el supuesto declarado `dividend_payment_lag_sessions`. Las únicas fechas de pago encontradas son capturas de prueba de proveedores para unos pocos valores de EE. UU. Solo cambia cuándo llega el efectivo. Con el plazo cero el efectivo se puede reinvertir antes que en el mercado, un efecto pequeño frente al importe del dividendo.
- **Ampliaciones de capital.** El precio de referencia de un exderecho con suscripción depende del precio y la proporción de la ampliación, que la cinta no representa. Si la fecha cae dentro de un tramo verificado, la caída exderecho aparece sin el derecho que la compensa. Penaliza mantener la posición y no permite ganar.
- **Precios muy bajos.** Desde 2023, si la banda queda a menos de un céntimo del cierre anterior, el límite se desplaza un céntimo. No afecta a precios por encima de 0,10 yuanes y no está implementado.
- **Topes por orden.** ChiNext limita una orden a 300.000 acciones a precio limitado y 150.000 a mercado, y STAR a 100.000 y 50.000. El tope se aplica a cada orden y no a la cantidad diaria, que puede repartirse en varias órdenes. La simulación agrupa en una ejecución por sesión, acotada por el 1 % del volumen, lo que en el mercado serían varias órdenes, así que no se espera efecto sobre lo ejecutable.
- **Comisiones de bolsa y de registro.** La de negociación bajó a 0,0487 ‰ en 2015 y a 0,0341 ‰ en agosto de 2023, y la de transferencia a 0,02 ‰ en 2015 y a 0,01 ‰ en 2022. Quedan dentro de `cost_bps`, que no varía por fecha. Juntas suman menos de 0,1 ‰ por lado, por debajo de un punto básico, frente a los 10 pb del ajuste y la rejilla de evaluación de 0, 5, 10 y 20 pb. No he enlazado aún las fuentes primarias de estas cifras, por eso no se aplican por fecha.
- **Vigencia.** Las reglas aplicadas terminan el 31 de diciembre de 2023. Las reglas de 2026 sustituyeron a las de 2023 desde el 6 de julio de 2026 y no se han revisado.

## Precios ajustados

Las reglas de lotes y límites se refieren a precios negociados sin ajustar. En los Parquet de la población preparada, con una tolerancia de 0,05 céntimos (muy superior al redondeo de `float32`), el 84 % de los 2.891.924 cierres chinos y el 69 % de los 16.090.522 estadounidenses no caen en un múltiplo de céntimo. En 2023 siguen fuera el 75 % y el 62 %. El resultado es coherente con precios ajustados por acciones corporativas posteriores, aunque la medida no identifica el método de ajuste. Con esa base, el redondeo del límite y los lotes no reproducen el mercado. La cinta real exige `price_basis="unadjusted"` y estas reglas solo tienen sentido con esa base.
