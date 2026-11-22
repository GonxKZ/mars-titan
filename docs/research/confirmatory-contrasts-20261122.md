# Contrastes confirmatorios de la campaña DGX, fijados antes de ajustar los orígenes limpios

Fijado el 22 de noviembre de 2026 a las 18:30 UTC. En ese momento solo existen ajustes del origen
2013 (semilla 42 y, en curso, 43 y 44). No se ha ajustado ningún modelo en los orígenes 2014 a 2023.
Este documento no se edita después: cualquier cambio se añade al final con su fecha y su motivo.

## Qué se decidió con el origen 2013 y con qué datos

El origen 2013 es el de desarrollo. La receta que se confirma se eligió con su tramo de decisión
(validación y calibración), no con su evaluación:

- **Modelo confirmado:** `titans_mac_nogain` (memoria en línea, rama densa con rangos por sesión,
  término de IC, sin ganancia de la mediana). Es el de mayor IC de decisión de los 29 ajustados
  (0,1201).
- **Controles:** `titans_mac_frozen_nogain` y `titans_mac_disabled_nogain` (la misma receta con la
  memoria congelada y sin memoria) y `lightgbm_technical` (árboles con los mismos 46 rasgos).

## Familia primaria

Evaluación de los nueve orígenes limpios (2014 a 2022), semilla 42. La unidad es el origen: de cada
uno se toma la diferencia media pareada por sesión del IC de rangos, y el contraste es la t de esas
nueve medias, con 8 grados de libertad, bilateral. Corrección de Holm al 5 % dentro de la familia.

| Contraste | Qué afirma |
| --- | --- |
| C1: `titans_mac_nogain` − `lightgbm_technical` | El modelo supera a la referencia tabular fuerte |
| C2: `titans_mac_nogain` − `titans_mac_frozen_nogain` | Escribir en la memoria aporta |
| C3: `titans_mac_nogain` − `titans_mac_disabled_nogain` | La memoria aporta |
| C4: órganos de memoria (brazo combinado) sobre `titans_mac_nogain` − su padre | Las memorias supervisadas con etiquetas maduras aportan |

Los cuatro se informan sobre dos medidas, cada una con su propio Holm: el IC de la etiqueta, que es
la métrica de la campaña, y el IC operable (etiqueta más alfa frente a la predicción por un factor
elegido en calibración más alfa). El alfa cambiado de signo tiene por sí solo un IC de 0,0813 con
la etiqueta en 2013, así que la segunda medida es la que dice si lo ganado se puede operar.

## Qué se informará siempre, salga lo que salga

- Los nueve valores de cada contraste, su media, su error entre orígenes, la t, la p sin corregir y
  la corregida, y cuántos orígenes van a favor. Herramienta: `scripts/dgx_seed_contrasts.py` con
  varios `--year`, `--offset` y la referencia de árboles.
- El resultado por sesión con error con autocorrelación, como medida secundaria.
- El origen 2023 y el 2013 se informan aparte y no entran en la familia: 2013 se usó para elegir y
  2023 tiene tres semillas con otro propósito (la varianza entre semillas).

## Secundarios, declarados como tales

El efecto de la ganancia de la mediana y de los rangos por sesión en cada modo de memoria, el núcleo
de la v10 con ganancia frente a sus controles, los lectores episódicos y el núcleo cognitivo frente a
su núcleo. Lo que salga de la criba de la base v11 (término de IC sobre lo operable, atención después
de la rama técnica y el resto de opciones) es exploratorio en 2013 y solo pasa a confirmatorio en un
ajuste posterior con su propia declaración.

## Potencia

Con nueve orígenes y 8 grados de libertad, la diferencia mínima detectable al 80 % es unas 3,2 veces
el error entre orígenes sin corregir y unas 4,1 veces para el contraste más exigente de Holm. Si la
desviación entre orígenes de la diferencia es de 0,0015 a 0,0020 (la de una sesión a otra dentro de
2013 da un error por origen de 0,0010 a 0,0015), el error entre orígenes es de 0,0005 a 0,0007 y se
detectan diferencias de 0,002 a 0,003. Los efectos medidos en 2013 son +0,0041 (C1), +0,0065 (C2),
+0,0104 (C3) y +0,0023 (C4, medido sobre el núcleo con ganancia). C4 queda al límite de lo detectable.

## Selección entre variantes: el listón del azar

En 2013 se compararon 25 variantes con su control. El mayor de 25 estadísticos sin efecto se espera
en torno a 2,0 desviaciones. La ventaja del modelo sin ganancia sobre el núcleo de la v10 (t = 3,0)
supera ese listón, pero con una sola semilla. Por eso el trío sin ganancia se ajusta también con las
semillas 43 y 44 en 2013 y en 2023 y entra en todos los orígenes limpios.
