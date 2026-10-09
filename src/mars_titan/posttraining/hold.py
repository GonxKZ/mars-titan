"""Rechazar los ajustes del postentrenamiento mientras rija la protección del aprendizaje."""

from mars_titan.training.learning_hold import learning_blocked


def refuse_while_blocked(action):
    """Fallar antes de abrir fuentes o crear salidas si la protección local está vigente.

    La comprobación se repite en cada punto de entrada que puede ajustar parámetros.
    Una protección ausente, o con `training_allowed` verdadero, no detiene el recorrido.
    """
    if learning_blocked():
        raise RuntimeError(f"Bloqueo de aprendizaje vigente: {action} no puede ajustar parámetros")
