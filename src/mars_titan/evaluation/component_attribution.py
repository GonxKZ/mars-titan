"""Atribuir una mejora a los componentes de una arquitectura con los brazos que existen.

Un linaje declara componentes binarios, las dependencias estructurales de cada uno
(``requires``) y el conjunto de componentes que activa cada brazo con nombre. Con esa tabla,
cada pregunta de atribución se convierte en una combinación lineal de brazos que
``compare_series`` estima con el mismo remuestreo por bloques que el resto de la
comparación. Así la atribución no necesita un estimador propio y comparte la corrección
por familia de las demás preguntas.

Se compilan cinco tipos de contraste. La escalera acumulada mide v(S_{i+1}) − v(S_i) en el
orden declarado, y su total es exactamente la suma de los pasos. Dejar uno fuera compara el
conjunto completo con el completo sin el componente ni lo que depende de él, porque quitar
una pieza y conservar a sus dependientes daría un brazo que no puede existir. Los efectos
condicionados v(S ∪ {c}) − v(S) aparecen para cada par de brazos con nombre que solo
difiere en c y muestran cuánto aporta c según lo que ya hay. La interacción 2×2 en un
contexto declarado es v(S+a+b) − v(S+a) − v(S+b) + v(S). El valor de Shapley de un juego
declarado (jugadores y contexto) es
φ_i = Σ_{T ⊆ P∖{i}} |T|!(n−|T|−1)!/n! · [v(S ∪ T ∪ {i}) − v(S ∪ T)].

Shapley solo está identificado si todas las coaliciones respetan las dependencias. Si
alguna activa una pieza sin aquella de la que depende, su valor no existe por construcción
y el juego se declara como limitación en lugar de completarlo con supuestos. Un conjunto
sin brazo con nombre deja su contraste pendiente con lo que falta. Este módulo no lee
datos ni predicciones.
"""

import math
from dataclasses import dataclass, replace
from itertools import combinations

MAX_COMPONENTS = 16
MAX_PLAYERS = 10
_LINEAGE_FIELDS = {"question", "components", "arms", "ladder", "full", "interactions", "games"}
_INTERACTION_FIELDS = {"first", "second", "context"}
_GAME_FIELDS = {"players", "context"}
ANALYSES = ("ladder", "leave_one_out", "conditional_effects", "interactions", "shapley")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 64 and value.replace("_", "").isalnum()


@dataclass(frozen=True)
class Lineage:
    """Linaje ya validado, con sus componentes, dependencias y brazos con nombre.

    Es inmutable para que las familias compiladas a partir de él no cambien después de la
    validación.
    """

    name: str
    question: str
    components: tuple
    requires: dict
    labels: dict
    arms: dict
    ladder: tuple
    full: frozenset
    interactions: tuple
    games: tuple

    def closure(self, components):
        """Devuelve los componentes junto con todas sus dependencias transitivas.

        Es la base para decidir si un conjunto describe un brazo que se puede construir.
        """
        result, pending = set(), list(components)
        while pending:
            component = pending.pop()
            if component not in result:
                result.add(component)
                pending.extend(self.requires[component])
        return frozenset(result)

    def closed(self, components):
        """Indica si el conjunto ya incluye todas sus dependencias."""
        return self.closure(components) == frozenset(components)

    def dependents(self, component, within):
        """Devuelve el componente y los de ``within`` que dependen de él.

        La dependencia puede ser indirecta. Son las piezas que desaparecen con el componente
        cuando se deja fuera.
        """
        return frozenset(c for c in within if component in self.closure([c]))

    def arm(self, components):
        """Devuelve el brazo que activa exactamente ese conjunto, o None si no hay ninguno.

        Un conjunto sin brazo deja pendiente cualquier contraste que lo use.
        """
        return self._names.get(frozenset(components))

    @property
    def _names(self):
        return {components: arm for arm, components in self.arms.items()}

    def order(self, components):
        """Ordena los componentes como en la declaración.

        Así los nombres de los contrastes no dependen del orden interno de los conjuntos.
        """
        return [c for c in self.components if c in components]


def load_lineage(name, declared):
    """Valida un linaje declarado antes de compilar ningún contraste.

    Exige dependencias sin ciclos, brazos y pasos de la escalera que incluyan sus
    dependencias, y contextos de interacción y de juego que no contengan ya las piezas
    estudiadas. Un error de declaración se detecta aquí y no aparece después como un
    contraste sin sentido.
    """
    _require(_identifier(name), f"El linaje {name} necesita un nombre válido")
    _require(
        isinstance(declared, dict) and set(declared) == _LINEAGE_FIELDS,
        f"El linaje {name} no declara exactamente {sorted(_LINEAGE_FIELDS)}",
    )
    _require(
        isinstance(declared["question"], str) and declared["question"],
        f"El linaje {name} necesita su pregunta",
    )
    components = declared["components"]
    _require(
        isinstance(components, dict) and 1 <= len(components) <= MAX_COMPONENTS,
        f"El linaje {name} declara entre 1 y {MAX_COMPONENTS} componentes",
    )
    requires, labels = {}, {}
    for component, entry in components.items():
        _require(
            _identifier(component)
            and isinstance(entry, dict)
            and set(entry) == {"label", "requires"}
            and isinstance(entry["label"], str)
            and entry["label"]
            and isinstance(entry["requires"], list)
            and len(set(entry["requires"])) == len(entry["requires"])
            and all(other in components and other != component for other in entry["requires"]),
            f"El componente {component} de {name} no declara etiqueta y dependencias válidas",
        )
        requires[component] = tuple(entry["requires"])
        labels[component] = entry["label"]
    _acyclic(name, requires)
    arms = {}
    for arm, members in _mapping(declared["arms"], f"Los brazos de {name}").items():
        _require(
            isinstance(members, list)
            and len(set(members)) == len(members)
            and all(member in components for member in members),
            f"El brazo {arm} de {name} usa componentes no declarados",
        )
        arms[arm] = frozenset(members)
    lineage = Lineage(
        name=name,
        question=declared["question"],
        components=tuple(components),
        requires=requires,
        labels=labels,
        arms=arms,
        ladder=(),
        full=frozenset(),
        interactions=(),
        games=(),
    )
    for arm, members in arms.items():
        _require(lineage.closed(members), f"El brazo {arm} de {name} no incluye sus dependencias")
    _require(
        len(set(arms.values())) == len(arms),
        f"Dos brazos de {name} activan los mismos componentes",
    )
    ladder = _components(lineage, declared["ladder"], f"La escalera de {name}")
    for size in range(len(ladder) + 1):
        _require(
            lineage.closed(ladder[:size]),
            f"La escalera de {name} añade un componente antes de sus dependencias",
        )
    full = frozenset(_components(lineage, declared["full"], f"El conjunto completo de {name}"))
    _require(lineage.closed(full), f"El conjunto completo de {name} no está cerrado")
    interactions = tuple(_interaction(lineage, entry) for entry in declared["interactions"])
    games = tuple(_game(lineage, entry) for entry in declared["games"])
    return replace(lineage, ladder=tuple(ladder), full=full, interactions=interactions, games=games)


def _mapping(value, label):
    _require(
        isinstance(value, dict) and value and all(_identifier(key) for key in value),
        f"{label} deben ser un diccionario con nombres válidos",
    )
    return value


def _components(lineage, value, label):
    _require(
        isinstance(value, list)
        and len(set(value)) == len(value)
        and all(component in lineage.requires for component in value),
        f"{label} debe listar componentes declarados sin repetir",
    )
    return list(value)


def _acyclic(name, requires):
    state = {}

    def visit(component):
        _require(state.get(component) != "open", f"Las dependencias de {name} forman un ciclo")
        if state.get(component) == "done":
            return
        state[component] = "open"
        for other in requires[component]:
            visit(other)
        state[component] = "done"

    for component in requires:
        visit(component)


def _context(lineage, value, label):
    _require(value in lineage.arms, f"{label} necesita un contexto con nombre del linaje")
    return value


def _interaction(lineage, entry):
    _require(
        isinstance(entry, dict) and set(entry) == _INTERACTION_FIELDS,
        f"Cada interacción de {lineage.name} declara {sorted(_INTERACTION_FIELDS)}",
    )
    first, second = entry["first"], entry["second"]
    context = _context(lineage, entry["context"], f"La interacción {first}x{second}")
    base = lineage.arms[context]
    _require(
        first in lineage.requires and second in lineage.requires and first != second,
        f"La interacción de {lineage.name} necesita dos componentes distintos",
    )
    _require(
        first not in base and second not in base,
        f"El contexto {context} ya contiene {first} o {second}",
    )
    for members in (base | {first}, base | {second}, base | {first, second}):
        _require(
            lineage.closed(members),
            f"La interacción {first}x{second} en {context} no respeta las dependencias",
        )
    return (first, second, context)


def _game(lineage, entry):
    _require(
        isinstance(entry, dict) and set(entry) == _GAME_FIELDS,
        f"Cada juego de Shapley de {lineage.name} declara {sorted(_GAME_FIELDS)}",
    )
    players = _components(lineage, entry["players"], f"Los jugadores de {lineage.name}")
    _require(
        2 <= len(players) <= MAX_PLAYERS,
        f"Un juego de Shapley tiene entre 2 y {MAX_PLAYERS} jugadores",
    )
    context = _context(lineage, entry["context"], "El juego de Shapley")
    _require(
        not set(players) & lineage.arms[context],
        f"El contexto {context} ya contiene algún jugador",
    )
    return (tuple(players), context)


def _term(*pairs):
    """Agrupa los coeficientes por conjunto de componentes.

    Un conjunto repetido indicaría un contraste mal construido, así que se rechaza en lugar
    de sumar sus coeficientes en silencio.
    """
    terms = {frozenset(members): coefficient for members, coefficient in pairs}
    _require(len(terms) == len(pairs), "Un contraste repite un conjunto de componentes")
    return terms


def ladder(lineage):
    """Compila los pasos de la escalera acumulada y su total.

    El total coincide con la suma de los pasos, pero cada paso depende del orden declarado.
    Por eso la escalera describe un camino concreto y no reparte por sí sola el mérito entre
    componentes.
    """
    steps = {}
    for index, component in enumerate(lineage.ladder):
        before = lineage.ladder[:index]
        steps[f"+{component}"] = _term((before, -1.0), ((*before, component), 1.0))
    if lineage.ladder:
        steps["total"] = _term(((), -1.0), (lineage.ladder, 1.0))
    return steps


def leave_one_out(lineage):
    """Compara el conjunto completo con el completo sin cada componente.

    También se retira lo que depende del componente, porque el brazo sin él y con sus
    dependientes no existe. El efecto resultante incluye, por tanto, a esos dependientes.
    """
    result = {}
    for component in lineage.order(lineage.full):
        removed = lineage.dependents(component, lineage.full)
        result[f"-{component}"] = _term((lineage.full, 1.0), (lineage.full - removed, -1.0))
    return result


def removed_with(lineage, component):
    """Devuelve lo que retira de verdad el contraste de dejar fuera ``component``.

    El informe lo muestra junto al efecto para no atribuir al componente lo que también
    aportan sus dependientes.
    """
    return lineage.order(lineage.dependents(component, lineage.full))


def conditional_effects(lineage):
    """Compila el efecto de cada componente sobre cada brazo con nombre que lo admite.

    Solo aparecen los pares en los que también existe el brazo con el componente añadido.
    Cuando un componente tiene efecto en varios contextos, la diferencia entre ellos indica
    si su aportación depende de lo que ya hay.
    """
    names = lineage._names
    result = {}
    for arm, members in lineage.arms.items():
        for component in lineage.components:
            if component in members:
                continue
            larger = members | {component}
            if larger in names:
                result[f"{component}@{arm}"] = _term((members, -1.0), (larger, 1.0))
    return result


def interactions(lineage):
    """Compila las interacciones 2×2 declaradas.

    Una interacción vale cero cuando los dos efectos se suman sin más. Un valor distinto de
    cero indica solape o complementariedad en ese contexto, no en cualquier otro.
    """
    result = {}
    for first, second, context in lineage.interactions:
        base = lineage.arms[context]
        result[f"{first}x{second}@{context}"] = _term(
            (base | {first, second}, 1.0),
            (base | {first}, -1.0),
            (base | {second}, -1.0),
            (base, 1.0),
        )
    return result


def coalitions(players, context):
    """Enumera todas las coaliciones del juego unidas al contexto.

    Con n jugadores salen 2^n conjuntos, y por eso el número de jugadores está acotado.
    """
    return [
        frozenset(context) | frozenset(subset)
        for size in range(len(players) + 1)
        for subset in combinations(players, size)
    ]


def shapley(lineage):
    """Calcula los valores de Shapley de cada juego identificado y la limitación de los demás.

    Un juego solo se resuelve si todas sus coaliciones respetan las dependencias. Los demás
    se devuelven aparte, con el número de coaliciones imposibles, porque completarlos
    exigiría inventar el valor de brazos que no pueden existir. Los valores de un juego
    resuelto suman exactamente v(contexto ∪ P) − v(contexto). Esa diferencia se añade como
    ``total@contexto`` para poder comprobar la eficiencia del reparto con los datos.
    """
    values, limitations = {}, []
    for players, context in lineage.games:
        base = lineage.arms[context]
        every = coalitions(players, base)
        invalid = [members for members in every if not lineage.closed(members)]
        if invalid:
            limitations.append(
                dict(
                    players=list(players),
                    context=context,
                    coalitions=len(every),
                    structurally_invalid=len(invalid),
                    reason=(
                        "Alguna coalición activa un componente sin sus dependencias, así que "
                        "su valor no existe y el reparto de Shapley no está identificado"
                    ),
                )
            )
            continue
        n = len(players)
        for player in players:
            others = [p for p in players if p != player]
            pairs = []
            for size in range(n):
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                for subset in combinations(others, size):
                    members = base | frozenset(subset)
                    pairs += [(members | {player}, weight), (members, -weight)]
            values[f"phi_{player}@{context}"] = _term(*pairs)
        values[f"total@{context}"] = _term((base | frozenset(players), 1.0), (base, -1.0))
    return values, limitations


def families(lineage):
    """Reúne las familias de contrastes del linaje y sus limitaciones.

    Los contrastes siguen expresados por conjuntos de componentes, sin resolver todavía los
    brazos. Se omiten las familias vacías, que no tendrían nada que estimar.
    """
    values, limitations = shapley(lineage)
    compiled = dict(
        ladder=ladder(lineage),
        leave_one_out=leave_one_out(lineage),
        conditional_effects=conditional_effects(lineage),
        interactions=interactions(lineage),
        shapley=values,
    )
    return {kind: terms for kind, terms in compiled.items() if terms}, limitations


def resolve(lineage, terms):
    """Traduce un contraste por conjuntos a coeficientes por brazo.

    Los conjuntos sin brazo con nombre se devuelven aparte. El contraste queda pendiente con
    lo que falta en lugar de estimarse con solo una parte de sus términos.
    """
    coefficients, unnamed = {}, []
    for members, value in terms.items():
        arm = lineage.arm(members)
        if arm is None:
            unnamed.append(lineage.order(members))
        else:
            coefficients[arm] = value
    return coefficients, unnamed
