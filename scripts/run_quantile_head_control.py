"""Orden del control de la cabeza de cuantiles de #22: comprobar, ejecutar y decidir.

`check` valida el plan, el protocolo, la comparación y las ventanas, y cuenta los
trabajos sin leer datos. `run` ajusta o reanuda cada trabajo con recibos y se detiene si
rige el bloqueo de aprendizaje. `decide` calcula el contraste por índice de diseño con
las predicciones de validación confirmadas y aplica la regla de retroceso declarada.
"""

from mars_titan.training import quantile_head_control

if __name__ == "__main__":
    raise SystemExit(quantile_head_control.main())
