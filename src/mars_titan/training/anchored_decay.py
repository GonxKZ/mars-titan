"""Decaimiento desacoplado de AdamW hacia los valores iniciales de los parámetros (#444).

AdamW multiplica cada peso por (1 − ηλ) antes de su actualización y lo lleva hacia cero.
En un adaptador ese factor actúa sobre la corrección δ, así que el peso efectivo θ₀ + δ
tiende al padre. En la continuación completa contrae en cambio los pesos del propio padre.
Este módulo da a la continuación el mismo ancla que tienen los adaptadores:

    θ ← θ₀ + (θ − θ₀)(1 − ηλ),  y después la actualización de Adam sin decaimiento,

con θ₀ el valor de cada parámetro al construir el optimizador, que en el postentrenamiento
es el padre. Es la forma desacoplada de L2-SP (Li et al., 2018), que suma (λ/2)‖θ − θ₀‖²
a la pérdida. Con Adam esa penalización pasaría por los momentos y se reescalaría por
coordenada. La forma desacoplada no, igual que AdamW frente a L2 (Loshchilov y Hutter,
2019). En aritmética exacta equivale a AdamW sobre un adaptador residual de rango completo
θ = θ₀ + δ en todos los parámetros: el gradiente respecto a δ es el de θ, así que Adam
recibe los mismos gradientes y δ decae con el mismo factor.

Detalles numéricos, comprobados en `tests/training/test_anchored_decay.py`:

- El factor se calcula como en AdamW, `1 - lr * weight_decay` con floats de Python.
- Con θ₀ = 0 la resta y la suma son exactas, así que el resultado coincide bit a bit con
  `param.mul_(1 - lr * weight_decay)` de AdamW, salvo el signo de un cero.
- Con θ = θ₀ la desviación es cero y el peso no cambia, sin deriva por redondeo.
- Si θ y θ₀ tienen el mismo signo y no se separan más de un factor dos, θ − θ₀ es exacta
  (lema de Sterbenz) y el decaimiento de ese paso es el de la corrección de un adaptador.
- Con λ = 0 no se toca ningún peso y el paso es el de AdamW sin decaimiento.
- Solo decaen los parámetros con gradiente, como en AdamW.
"""

import math
from pathlib import Path

import torch

from mars_titan.data.storage import sha256

INITIAL = "initial_parameters"
ANCHORS = (None, INITIAL)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_anchor(anchor):
    """Ancla declarada del decaimiento: None (hacia cero, AdamW) o los valores iniciales."""
    _require(
        anchor is None or (isinstance(anchor, str) and anchor == INITIAL),
        "El decaimiento solo se ancla en cero, como AdamW, o en los valores iniciales",
    )
    return anchor


def _rate(value):
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        "El decaimiento anclado necesita un λ y una tasa finitos y no negativos",
    )
    return value


def _decay_(values, anchors, rate):
    """θ ← θ₀ + (θ − θ₀)(1 − ρ) en sitio, con un núcleo por operación para toda la lista."""
    torch._foreach_sub_(values, anchors)
    torch._foreach_mul_(values, 1 - rate)
    torch._foreach_add_(values, anchors)


class AnchoredAdamW(torch.optim.AdamW):
    """AdamW cuyo decaimiento desacoplado lleva cada peso a su valor inicial.

    El decaimiento se aplica en un gancho previo al paso, que corre después de los ganchos
    globales (entre ellos la protección del aprendizaje), y AdamW recibe `weight_decay=0`.
    Las anclas viajan en `param_groups`, así que el estado del optimizador las guarda y la
    reanudación recupera las del padre aunque el modelo ya no esté en su valor inicial.
    """

    def __init__(self, params, *, lr, weight_decay, **options):
        _require(type(lr) in (int, float), "La tasa de aprendizaje anclada es un número")
        self._anchored_weight_decay = float(_rate(weight_decay))
        super().__init__(params, lr=lr, weight_decay=0.0, **options)
        self.register_step_pre_hook(_apply_decay)
        self.register_load_state_dict_post_hook(_place_anchors)

    def add_param_group(self, param_group):
        _require(
            param_group.get("weight_decay", 0.0) == 0,
            "Un grupo del decaimiento anclado no declara además un decaimiento hacia cero",
        )
        _rate(param_group.setdefault("anchored_weight_decay", self._anchored_weight_decay))
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        group["anchors"] = [value.detach().clone() for value in group["params"]]

    def _decaying(self):
        """Por grupo, tasa, parámetros con gradiente y sus anclas. Sin tasa no hay nada."""
        for group in self.param_groups:
            rate = group["lr"] * group["anchored_weight_decay"]
            pairs = [
                (value, anchor)
                for value, anchor in zip(group["params"], group["anchors"], strict=True)
                if value.grad is not None
            ]
            if rate != 0 and pairs:
                yield rate, [value for value, _ in pairs], [anchor for _, anchor in pairs]

    def decayed_parameters(self):
        """Parámetros que decaerían en el próximo paso y su valor tras el decaimiento.

        Calcula sobre copias con la misma operación del paso y no modifica nada, así que
        sirve para comprobar la contracción sin pasos de optimizador.
        """
        result = []
        for rate, values, anchors in self._decaying():
            copies = [value.detach().clone() for value in values]
            _decay_(copies, anchors, rate)
            result += list(zip(values, copies, strict=True))
        return result


def _apply_decay(optimizer, args, kwargs):
    # Con una clausura, AdamW recalcularía los gradientes después de este gancho, sobre
    # pesos ya decaídos. Los entrenadores del proyecto no la usan.
    closure = args[0] if args else kwargs.get("closure")
    _require(
        closure is None and len(args) <= 1 and set(kwargs) <= {"closure"},
        "El decaimiento anclado no admite clausura en el paso",
    )
    with torch.no_grad():
        for rate, values, anchors in optimizer._decaying():
            _decay_(values, anchors, rate)


def _place_anchors(optimizer):
    # La carga restringida de los puntos de control deja las anclas en CPU y se llevan junto a
    # su parámetro. `Optimizer.load_state_dict` ya copia los grupos guardados, así que no
    # comparten memoria con el estado leído. Un tipo distinto no se convierte: indicaría un
    # estado de otro modelo o de otra precisión.
    for group in optimizer.param_groups:
        anchors = group.get("anchors")
        _require(
            isinstance(anchors, list)
            and len(anchors) == len(group["params"])
            and all(
                isinstance(anchor, torch.Tensor)
                and anchor.shape == value.shape
                and anchor.dtype == value.dtype
                for anchor, value in zip(anchors, group["params"], strict=True)
            ),
            "Las anclas guardadas no corresponden a los parámetros del optimizador",
        )
        _rate(group.get("anchored_weight_decay"))
        group["anchors"] = [
            anchor.detach().to(device=value.device)
            for anchor, value in zip(anchors, group["params"], strict=True)
        ]


def adamw(parameters, *, learning_rate, weight_decay, anchor=None):
    """AdamW de una receta o un caso: hacia cero, como hasta ahora, o anclado al inicio.

    Sin ancla devuelve el mismo `torch.optim.AdamW` que construían los entrenadores.
    """
    if checked_anchor(anchor) is None:
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    return AnchoredAdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def recipe_factory(recipe):
    """Fábrica por defecto de los entrenadores cronológicos para su receta."""
    return lambda values: adamw(
        values,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        anchor=recipe.weight_decay_anchor,
    )


def describe(optimizer, anchor):
    """Identidad del decaimiento anclado que declara una receta, o None sin ancla.

    Incluye la huella de este módulo, de modo que una reanudación con otro código se rechaza.
    Un optimizador de PyTorch debe aplicar exactamente el ancla declarada. Los dobles de las
    pruebas, que no heredan de `torch.optim.Optimizer`, no se comprueban.
    """
    if isinstance(optimizer, torch.optim.Optimizer):
        _require(
            isinstance(optimizer, AnchoredAdamW) == (checked_anchor(anchor) is not None),
            "El optimizador no aplica el ancla del decaimiento que declara la receta",
        )
    if checked_anchor(anchor) is None:
        return None
    return dict(
        anchor=anchor,
        rule="theta_anchor_plus_deviation_times_one_minus_lr_weight_decay_v1",
        implementation_sha256=sha256(Path(__file__)),
    )
