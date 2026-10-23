"""Etapas de salida del estudio de variantes que se aplican a predicciones ya emitidas (#748).

Una etapa de salida transforma las predicciones y cuantiles guardados de un ajuste. Se ajusta
con la partición de calibración (G) y se aplica sin cambios a los tramos posteriores. No
entrena pesos ni cambia la identidad del predictor, así que no es un `predictor_changes`:
con un campo nuevo en `FinancialConfig`, una variante que solo actúa sobre la salida pediría
otro ajuste idéntico del modelo con otra huella. La criba combina en su lugar cada trabajo
(receta, `predictor_changes`, semilla y origen) con las etapas declaradas aquí, que se
validan con `checked_output_stages`. Sin etapas, la salida es la del modelo, bit a bit.
"""

from mars_titan.evaluation import neutral_abstention

# Nombre de cada etapa y la función que valida su configuración declarada.
OUTPUT_STAGES = {neutral_abstention.IDENTITY: neutral_abstention.checked_config}


def checked_output_stages(stages):
    """Validar las etapas de salida declaradas, o devolver None si no se declara ninguna."""
    if stages is None:
        return None
    if not (isinstance(stages, dict) and stages and set(stages) <= set(OUTPUT_STAGES)):
        raise ValueError(
            f"Las etapas de salida admiten {', '.join(OUTPUT_STAGES)} con su configuración"
        )
    return {name: OUTPUT_STAGES[name](config) for name, config in stages.items()}
