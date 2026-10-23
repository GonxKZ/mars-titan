# Registro finito de configuraciones

`mars_titan.proposals.contracts` identifica recetas financieras sin construir modelos, copiar pesos o autorizar su despliegue. El módulo implementa dos tipos: `RegisteredConfiguration` y `ProposalRegistry`. Su creación no activa una ejecución ni acredita un ciclo exterior de propuestas.

`RegisteredConfiguration` recibe un `FinancialConfig`, un `EpisodicReadoutConfig` opcional, una `MechanismAction` fija, un `MechanismCatalog` y una precisión explícita. Solo admite `"torch.float32"` y `"torch.float64"`. La acción debe conservar `calibration="none"`. La identidad registra `policy=None` y `calibration=None`, sin incorporar un selector aprendido o un calibrador.

El contrato comprueba que el programa regional del predictor coincide con el del catálogo y que el contrato del lector también corresponde al catálogo. Si hay lector, exige `refinements=1` y la dimensión oculta del predictor. Las rutas K mantienen su significado de refinamiento episódico. J pertenece a la recurrencia del puente y conserva su configuración independiente. Registrar una receta no aprende ni cambia esos valores.

`ProposalRegistry` recibe una tupla de entre una y ocho configuraciones distintas y un `baseline_id` que debe pertenecer a ella. Conserva el orden declarado. La identidad de cada configuración y la del registro completo tienen un límite de 1 MiB de JSON canónico. Ese límite no representa el pico de RAM ni la cuota global del futuro ciclo. Un identificador desconocido, una referencia ausente, duplicados o tipos incompatibles producen `ValueError`.

El ejemplo parte de `predictor_config`, un `FinancialConfig` ya construido con el contrato de entradas del proyecto. Registra una referencia sin lector y sin calibración:

```python
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.proposals.contracts import ProposalRegistry, RegisteredConfiguration

reference = RegisteredConfiguration(
    predictor=predictor_config,
    readout=None,
    action=MechanismAction(),
    catalog=MechanismCatalog(
        regional_execution=predictor_config.regional_execution,
    ),
    dtype="torch.float32",
)
registry = ProposalRegistry(
    entries=(reference,),
    baseline_id=reference.configuration_id,
)
registered = registry.resolve(reference.configuration_id)
```

`configuration_id` y `registry_id` son huellas SHA-256 de las identidades completas selladas. `identity()` devuelve una copia y comprueba que los objetos recibidos siguen correspondiendo al sello, incluso si se alteran campos internos. `resolve()` valida el registro y devuelve únicamente la receta solicitada. No crea ni sustituye un consumidor.

Estas huellas identifican configuraciones, no los pesos de un modelo entrenado ni su estado rápido, snapshot o historial de resultados. La compatibilidad estructural del registro tampoco acredita disponibilidad de recursos durante una ejecución. El módulo no calcula pérdidas, compara candidatas, selecciona una configuración por su utilidad ni confirma cambios. No ejecuta aprendizaje o evaluación predictiva y no modifica el bloqueo de la campaña ni del test final.

La comparación posterior deberá verificar también filas, ventanas, emparejamiento de pesos y presupuestos. La pertenencia al registro no demuestra por sí sola que dos ejecuciones sean comparables. `baseline_id` identifica una configuración del registro y no sustituye el manifiesto experimental de B.

El [informe técnico](../../reports/engineering/proposal-registry-20261023/README.md) recoge 27 casos, cobertura, CRAP y cuatro mutantes dirigidos, junto con los intentos previos no aprobados.
