# Recibos operativos de campañas adaptativas

`simulation/campaign_receipts.py` normaliza el progreso sin importar el runtime científico. Solo utiliza la biblioteca estándar. Recibe la ruta de `registry.json` y lee ese archivo, `run.json`, `campaign.json` e `identity.json`. No abre pesos, datasets ni los recibos de auditoría individuales. Devuelve estado, fase, recuentos, identidad, huella del diario y reserva. No devuelve métricas científicas.

El diario conserva el estado canónico en un sobre `payload` y `sha256`. El lector verifica su huella con la serialización de `Campaign.publish`, la identidad asociada y las declaraciones explícitas `final_test_opened=false` de identidad, configuración base, configuración de campaña e informe. La ausencia de una declaración no equivale a una reserva cerrada. El informe debe conservar padre congelado, dominio sintético y propósito técnico.

Los casos se identifican por etapa, variante y semilla. Se rechazan duplicados, rutas ajenas, estados desconocidos, recuentos incompatibles y archivos vacíos, truncados o con campos JSON repetidos. Cada archivo admite como máximo 32 MiB y el diario admite 75 casos. El plan se obtiene de los casos declarados, incluyendo sus ampliaciones principal, auxiliar y de auditoría. Un parámetro `expected` distinto de `None` exige que coincida con el plan observado.

El cierre exige estado y fase `completed`, todos los casos completados, las etapas declaradas por variantes, semillas y decisión auxiliar, auditoría abierta y selección congelada, presupuesto completo y ningún caso activo. `audit_opened` y `final_test_opened` tienen funciones distintas. Abrir la auditoría técnica no acredita ni autoriza abrir el test final.

## Publicación por generaciones

La campaña publica diario, informe y registro en ese orden. Cada archivo es atómico, pero los tres no se sustituyen como una única transacción. El lector vuelve a leer el diario después de los otros archivos y admite hasta tres intentos si cambia durante la lectura.

Un informe anterior o un registro retrasado con las mismas identidades y sin atribuir progreso futuro producen `BlockingIOError`. La fecha vincula versiones del informe, pero no se supone que el reloj de pared avance de forma monótona. Una huella incorrecta, un duplicado o una reserva inválida producen `ValueError` y no se ocultan como una publicación pendiente.

`posttraining.completion.run_child` admite `receipt_reader` como argumento nombrado. Su valor por defecto conserva `_receipt`, el lector predictivo. Durante la ejecución omite únicamente `BlockingIOError` al observar progreso. Después de una salida de código cero exige que el lector confirme un recibo completo. Un código de pausa no convierte un recibo `blocked`, `failed` o `completed` en una pausa recuperable. Los fallos confirmados tampoco se interpretan como pausa aunque el proceso termine con código cero.

## Carga operativa separada

Se pueden copiar únicamente `completion.py` y `campaign_receipts.py` a un directorio operativo externo y cargarlos con nombres propios. La copia de `completion.py` presta `_receipt` y `run_child`. Sus imports se resuelven contra el paquete científico que permanezca en `PYTHONPATH`. Los subprocesos y `FrozenInputs` siguen utilizando ese paquete original.

```python
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_receipts(directory):
    directory = Path(directory)

    def load(name, filename):
        spec = spec_from_file_location(name, directory / filename)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    completion = load("operational_completion", "completion.py")
    adaptive = load("operational_receipts", "campaign_receipts.py")

    def receipt(path, expected=None):
        if Path(path).name == "registry.json":
            return adaptive.read_adaptive_receipt(path, expected)
        return completion._receipt(path, expected)

    return receipt, partial(completion.run_child, receipt_reader=receipt)
```

El coordinador debe utilizar el mismo lector al comprobar etapas ya terminadas y al observar el hijo. El adaptador no escribe los cuatro artefactos ni reconstruye casos. La recuperación y la verificación científica de cada caso permanecen en `Campaign`.

Se comprobaron 115 pruebas CPU de recibos, ejecución por procesos, finalización y campaña adaptativa. Incluyen esquemas sintéticos reducidos, planes auxiliares, publicaciones entre generaciones, retroceso del reloj, pausa, fallo y cierre. El helper se cargó en un intérprete aislado sin paquetes externos. También se comprobó la carga operativa junto a los imports de la revisión científica congelada, sin lanzar trabajos ni cambiar sus archivos.

Tres mutaciones dirigidas se detectaron al retirar la reserva de la identidad, la comprobación de la huella del diario y la exigencia de completar todos los casos. No se midió cobertura en el entorno compartido, que no tiene instalado `coverage`.
