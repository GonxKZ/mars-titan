# Reglas de negociación de acciones A en la simulación

La simulación financiera ejecuta las órdenes en la apertura siguiente con lotes de una acción, sin límites diarios de precio y con un coste único en puntos básicos. En China ese supuesto sobrestima lo que puede ejecutarse. `simulation/market_rules.py` declara, para cada activo de Shanghái o Shenzhen, las reglas que he podido comprobar en fuentes primarias entre 2006 y 2023. `china_a_share_instrument(asset)` devuelve un `Instrument` con identidad `cn_a_share_v1_main`, `cn_a_share_v1_chinext` o `cn_a_share_v1_star`, y `FinancialEnv(..., instruments=...)` las aplica en el motor Python.

Las reglas son una identidad nueva del entorno (`instruments_sha256`). Sin `instruments`, la contabilidad, las identidades y las trayectorias no cambian. El motor nativo rechaza instrumentos con reglas hasta que las implemente, en lugar de ignorarlas.

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

## Lo que falta

- **Estado ST.** Las acciones con advertencia de riesgo tienen una banda del 5 % en el tablero principal (artículo 3.4.13 de Shanghái 2006, reglas del tablero de advertencia de riesgo desde 2013 y artículo 4.4.10 de 2023, artículos 3.3.14 de 2006 y 4.5.5 de 2021 y 2023 en Shenzhen). La población preparada no conserva ese estado por fecha. Aplicar el 10 % a un valor ST permite ejecuciones que el mercado habría bloqueado.
- **Primeros días tras la salida a bolsa.** STAR, ChiNext desde 2020 y el principal desde el 10 de abril de 2023 no tienen límites durante cinco sesiones. Antes, el primer día del principal limitaba el precio de las órdenes, no la variación. Sin fecha de admisión verificada, la simulación aplica la banda desde la segunda sesión con cierre anterior.
- **Ampliaciones de capital.** El precio de referencia de un exderecho con suscripción depende del precio y la proporción de la ampliación, que la cinta no representa.
- **Precios muy bajos.** Desde 2023, si la banda queda a menos de un céntimo del cierre anterior, el límite se desplaza un céntimo. No afecta a precios por encima de 0,10 yuanes y no está implementado.
- **Topes por orden.** ChiNext limita una orden a 300.000 acciones a precio limitado y 150.000 a mercado, y STAR a 100.000 y 50.000. La participación del 1 % del volumen suele quedar por debajo, pero no se comprueba.
- **Comisiones de bolsa y de registro.** La de negociación bajó a 0,0487 ‰ en 2015 y a 0,0341 ‰ en agosto de 2023, y la de transferencia a 0,02 ‰ en 2015 y a 0,01 ‰ en 2022. Quedan dentro de `cost_bps`, que no varía por fecha.
- **Vigencia.** Las reglas aplicadas terminan el 31 de diciembre de 2023. Las reglas de 2026 sustituyeron a las de 2023 desde el 6 de julio de 2026 y no se han revisado.
- **Motor nativo.** La sesión C++ y PPO nativo no aplican estas reglas. Rechazan instrumentos que las declaren.

## Precios ajustados

Las reglas de lotes y límites se refieren a precios negociados sin ajustar. En los Parquet de la población preparada, con una tolerancia de 0,05 céntimos (muy superior al redondeo de `float32`), el 84 % de los 2.891.924 cierres chinos y el 69 % de los 16.090.522 estadounidenses no caen en un múltiplo de céntimo. En 2023 siguen fuera el 75 % y el 62 %. El resultado es coherente con precios ajustados por acciones corporativas posteriores, aunque la medida no identifica el método de ajuste. Con esa base, el redondeo del límite y los lotes no reproducen el mercado. La cinta real exige `price_basis="unadjusted"` y estas reglas solo tienen sentido con esa base.
