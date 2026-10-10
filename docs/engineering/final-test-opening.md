# Apertura única del test final de 2024

Estado: procedimiento declarado e implementado sin abridor. Nada en el repositorio
prepara los objetivos de 2024 ni lee sus filas, y la orden `open` se niega antes de
registrar nada. Las pruebas usan un repositorio temporal y un abridor simulado que
solo escribe un archivo de texto.

El [protocolo](../research/protocol.md#reglas-para-cerrar-el-estudio) exige cerrar
antes de abrir el test el registro de experimentos, las exclusiones, las fórmulas,
el número de ensayos y la política de memoria. La
[declaración](../../configs/evaluation/final-test-2024-opening.json) convierte esa
exigencia en un orden comprobable y `evaluation/final_test_opening.py` lo ejecuta.

## Orden declarado

1. **Congelar** (`freeze`). Escribe `frozen.json` en un directorio de estado
   privado, fuera del repositorio. Contiene la huella de la comparación declarada
   y de los protocolos de cada ámbito, la ruta y huella de cada artefacto congelado
   (registro de ensayos, exclusiones, política de memoria, recibos de selección y
   edición de datos), el número de ensayos, las familias confirmatorias (las demás
   quedan como exploratorias) y el commit. Exige un repositorio limpio, familias que
   existan en la comparación, al menos un ensayo y exactamente los cinco artefactos.
   Se puede volver a congelar mientras no haya apertura.
2. **Comprobar** (`check`). Evalúa las condiciones en el orden declarado y no
   escribe nada.
3. **Confirmar.** La apertura exige la huella del manifiesto congelado, que solo se
   conoce después de congelar.
4. **Registrar la apertura.** Crea el registro privado con creación exclusiva
   (`O_CREAT | O_EXCL`) antes de llamar al abridor. Desde ese momento no hay otra
   apertura, aunque el abridor falle.
5. a 7. **Abridor.** Preparar los objetivos de 2024, predecir con los estados
   congelados y evaluar la comparación declarada. No está conectado.
8. **Publicar el registro.** Al terminar, con éxito o con fallo, el registro se
   reescribe con su estado, sus salidas y sus huellas y se copia en
   `reports/evaluation/final-test-2024/opening.json` dentro del repositorio.

## Condiciones

| Condición | Se cumple si |
| --- | --- |
| `frozen_manifest_unchanged` | Hay manifiesto congelado de este procedimiento y siguen iguales la comparación, sus protocolos y cada artefacto. |
| `repository_clean_at_frozen_commit` | El repositorio está limpio y en el commit congelado. |
| `learning_hold_lifted` | La protección local del aprendizaje no está vigente. Solo se lee, nunca se modifica. |
| `no_previous_opening` | No existe el registro privado ni el del repositorio. |

Basta con uno de los dos registros para impedir otra apertura, así que borrar solo
uno no permite repetir. La declaración no se puede relajar: el validador rechaza
otro orden de condiciones, un abridor declarado como conectado, otra regla de
repetición, otras fechas del test y rutas de registro que salgan del directorio.

## Uso previsto

```bash
uv run python -m mars_titan.evaluation.final_test_opening check \
  --procedure configs/evaluation/final-test-2024-opening.json --state <estado>
uv run python -m mars_titan.evaluation.final_test_opening freeze \
  --procedure configs/evaluation/final-test-2024-opening.json --state <estado> \
  --artefact trial_registry=<ruta> --artefact exclusions=<ruta> \
  --artefact memory_policy=<ruta> --artefact selection_receipts=<ruta> \
  --artefact data_edition=<ruta> --trials <n> --confirmatory references_vs_zero
```

`open --confirm <huella>` comprueba todo y se niega mientras el abridor no exista.
Conectarlo será un cambio revisado aparte, con su propia evidencia, y deberá
llamarse solo desde esta función.

## Comprobaciones

`tests/evaluation/test_final_test_opening.py` cubre el orden y las condiciones de
la declaración, el contenido del manifiesto, cada condición por separado
(artefacto cambiado, archivos sin versionar, otro commit, protección vigente,
comparación cambiada), la negativa sin abridor y con otra confirmación, la apertura
con éxito y con fallo, la imposibilidad de repetir incluso borrando un registro y
la creación exclusiva cuando otro proceso registra entre la comprobación y la
apertura. La mutación dirigida de 18 defectos los mató todos tras añadir la prueba
de la comparación cambiada.

## Límites

El procedimiento no protege frente a quien borre los dos registros a la vez o
cambie el código. Su función es dejar constancia verificable y hacer que repetir
exija un acto deliberado y visible en el historial. Tampoco decide qué familias son
confirmatorias. Esa elección debe fijarse y revisarse antes de congelar.
