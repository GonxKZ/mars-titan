"""Grafo estático de llamadas para comprobar que todo ajuste pasa por el bloqueo.

Recorre con `ast` los módulos de `src/mars_titan` y de `scripts` y localiza las llamadas que
ajustan parámetros:

- `x.step()` sin argumentos (o solo con `closure`), el paso de un optimizador.
- La construcción de un optimizador de `torch.optim`.
- `xgboost.train`.
- `fit`, `partial_fit`, `fit_transform` o `fit_predict` sobre un objeto que no es una clase
  ni un módulo del proyecto (estimadores de scikit-learn, hmmlearn y similares).
  `ActionGrid.fit` y las demás estadísticas del proyecto son funciones propias y se siguen
  como llamadas normales.
- Las soluciones cerradas de `linalg` (`solve`, `lstsq`, `cho_solve` y similares), como la
  ecuación normal de la ridge.

Las tres reglas externas valen igual con el módulo delante (`xgb.train`) que con el nombre
importado (`from xgboost import train`). Un `step(closure)` posicional no se reconoce como
paso de optimizador: es un límite conocido y no hay ninguno en el proyecto.

Las aristas van de cada función a las funciones y métodos del proyecto que llama o
referencia. Un nombre se resuelve con las importaciones del módulo, `self` y `cls` se
resuelven con la clase y sus redefiniciones en subclases, y las variables locales o los
atributos de `self` creados con el constructor de una clase del proyecto toman esa clase.
Con un receptor desconocido se enlazan todos los métodos con ese nombre de los módulos que
el llamador importa. Es una aproximación por exceso: puede añadir llamadores que no existen,
pero no pierde los de los módulos importados. Las funciones anidadas y las lambdas cuentan
como parte de la función que las contiene.

Una función está protegida si llama a `require_learning_allowed`. Un sitio de ajuste queda
sin proteger si alguna cadena de llamadas llega a él desde una raíz (función sin llamadores
o código a nivel de módulo) sin pasar por una función protegida ni por un consumidor
exento. Los ejecutables nativos comprueban la protección en C++ y quedan fuera de este
recorrido.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

GUARD = "require_learning_allowed"
HOOK = "register_optimizer_step_pre_hook"
# Pasos que el gancho global de PyTorch puede rechazar. Un estimador, XGBoost o una solución
# cerrada no pasan por él, así que solo los detiene la guarda.
TORCH_KINDS = {"paso de optimizador", "optimizador de torch.optim"}
SOLVERS = {"solve", "lstsq", "cho_solve", "solve_triangular", "cholesky_solve", "lu_solve"}
# Métodos de los estimadores que ajustan sus parámetros, también combinados con otra salida.
FITS = {"fit", "partial_fit", "fit_transform", "fit_predict"}
MODULE_SCOPE = "<module>"


@dataclass(frozen=True)
class Site:
    module: str
    function: str
    line: int
    kind: str

    @property
    def node(self):
        return (self.module, self.function)


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def label(node):
    return f"{node[0]}:{node[1]}"


def is_guard(call):
    func = call.func
    return (isinstance(func, ast.Name) and func.id == GUARD) or (
        isinstance(func, ast.Attribute) and func.attr == GUARD
    )


def guard_lines(path, qualname):
    """Líneas de las guardas de una función de nivel superior o de un método, en orden."""
    module = _Module("", [], Path(path))
    return sorted(
        sub.lineno
        for sub in ast.walk(module.functions[qualname])
        if isinstance(sub, ast.Call) and is_guard(sub)
    )


class _Module:
    def __init__(self, name, package, path):
        self.name, self.path = name, path
        self.tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        self.imports, self.functions, self.classes, self.bases = {}, {}, {}, {}
        # Las importaciones dentro de funciones también cuentan: los medidores importan los
        # entrenadores de forma diferida.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.imports[alias.asname] = ("module", alias.name)
                    else:
                        head = alias.name.split(".")[0]
                        self.imports[head] = ("module", head)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    stem = package[: len(package) - node.level + 1]
                    base = ".".join(stem + ([node.module] if node.module else []))
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = ("symbol", base, alias.name)
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
                self.bases[node.name] = [dotted(base) for base in node.bases]
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.functions[f"{node.name}.{item.name}"] = item
        self.known = {name}
        for target in self.imports.values():
            self.known.add(target[1])
            if target[0] == "symbol":
                self.known.add(f"{target[1]}.{target[2]}")


class CallGraph:
    def __init__(self, root, *, package="mars_titan", scripts="scripts"):
        root = Path(root)
        self.modules = {}
        source = root / "src"
        for path in sorted((source / package).rglob("*.py")):
            parts = list(path.relative_to(source).with_suffix("").parts)
            stem = parts[:-1]
            if parts[-1] == "__init__":
                parts = parts[:-1]
            self.modules[".".join(parts)] = _Module(".".join(parts), stem, path)
        if (root / scripts).is_dir():
            for path in sorted((root / scripts).glob("*.py")):
                name = f"{scripts}.{path.stem}"
                self.modules[name] = _Module(name, [scripts], path)
        self.methods = defaultdict(set)
        for module in self.modules.values():
            for qualname in module.functions:
                if "." in qualname:
                    self.methods[qualname.split(".", 1)[1]].add((module.name, qualname))
        self.subclasses = defaultdict(set)
        for module in self.modules.values():
            for cls, bases in module.bases.items():
                for base in bases:
                    found = self.symbol(module.name, base.split(".")[-1]) if base else None
                    if found and found[0] == "class":
                        self.subclasses[(found[1], found[2])].add((module.name, cls))
        self.sites = []
        self.guards = defaultdict(list)
        self.hooks = set()
        self.edges = defaultdict(set)
        self.calls = defaultdict(list)
        for module in self.modules.values():
            self._scan(module)

    # Resolución de nombres.

    def symbol(self, module_name, name, depth=0):
        """Función, clase o módulo del proyecto al que remite `name` en ese módulo."""
        module = self.modules.get(module_name)
        if module is None or depth > 8:
            return None
        if name in module.functions:
            return ("function", module_name, name)
        if name in module.classes:
            return ("class", module_name, name)
        target = module.imports.get(name)
        if target and target[0] == "symbol":
            if f"{target[1]}.{target[2]}" in self.modules:
                return ("module", f"{target[1]}.{target[2]}")
            return self.symbol(target[1], target[2], depth + 1)
        if target and target[0] == "module" and target[1] in self.modules:
            return ("module", target[1])
        return None

    def method(self, module_name, cls, name, depth=0):
        """Método `name` de la clase o, si no lo define, de sus bases del proyecto."""
        module = self.modules[module_name]
        if f"{cls}.{name}" in module.functions:
            return (module_name, f"{cls}.{name}")
        if depth > 8:
            return None
        for base in module.bases.get(cls, []):
            found = self.symbol(module_name, base.split(".")[-1]) if base else None
            if found and found[0] == "class":
                hit = self.method(found[1], found[2], name, depth + 1)
                if hit:
                    return hit
        return None

    def dispatch(self, module_name, cls, name):
        """Método heredado y sus redefiniciones en las subclases del proyecto."""
        hits = set()
        hit = self.method(module_name, cls, name)
        if hit:
            hits.add(hit)
        pending, seen = [(module_name, cls)], set()
        while pending:
            for sub in self.subclasses.get(pending.pop(), ()):
                if sub not in seen:
                    seen.add(sub)
                    if f"{sub[1]}.{name}" in self.modules[sub[0]].functions:
                        hits.add((sub[0], f"{sub[1]}.{name}"))
                    pending.append(sub)
        return hits

    def _constructed(self, module, call):
        """Clase del proyecto que construye `call`, si se puede saber."""
        name = dotted(call.func)
        if not name:
            return None
        head, *rest = name.split(".")
        found = self.symbol(module.name, head)
        for part in rest:
            if not found or found[0] != "module":
                return None
            found = self.symbol(found[1], part)
        return (found[1], found[2]) if found and found[0] == "class" else None

    # Recorrido.

    def _scan(self, module):
        attributes = defaultdict(dict)
        for cls, node in module.classes.items():
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
                    found = self._constructed(module, sub.value)
                    for target in sub.targets:
                        if (
                            found
                            and isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            attributes[cls][target.attr] = found
        for node in module.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(module, node.name, node, None, {})
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualname = f"{node.name}.{item.name}"
                        self._function(module, qualname, item, node.name, attributes[node.name])
                    else:
                        self._body(module, MODULE_SCOPE, [item], node.name, {}, {})
            else:
                self._body(module, MODULE_SCOPE, [node], None, {}, {})

    def _function(self, module, qualname, node, cls, attributes):
        # Una función que registra un gancho previo al paso que lanza una excepción rechaza
        # cualquier paso de torch.optim mientras dura la medición.
        raising = {
            sub.name
            for sub in ast.walk(node)
            if isinstance(sub, ast.FunctionDef)
            and sub is not node
            and any(isinstance(item, ast.Raise) for item in sub.body)
        }
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and (dotted(sub.func) or "").rsplit(".", 1)[-1] == HOOK
                and sub.args
                and isinstance(sub.args[0], ast.Name)
                and sub.args[0].id in raising
            ):
                self.hooks.add((module.name, qualname))
        local = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
                found = self._constructed(module, sub.value)
                for target in sub.targets:
                    if found and isinstance(target, ast.Name):
                        local[target.id] = found
        roots = [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults, *node.body]
        self._body(module, qualname, roots, cls, attributes, local)

    def _body(self, module, qualname, roots, cls, attributes, local):
        me = (module.name, qualname)
        for root in roots:
            if root is None:
                continue
            # Un nombre que es la base de un atributo (`Clase.metodo`) no construye la clase.
            bases = {id(node.value) for node in ast.walk(root) if isinstance(node, ast.Attribute)}
            for sub in ast.walk(root):
                line = getattr(sub, "lineno", 0)
                if isinstance(sub, ast.Call):
                    kind = self._fitting(module, sub)
                    if kind:
                        self.sites.append(Site(module.name, qualname, sub.lineno, kind))
                    if is_guard(sub):
                        self.guards[me].append(sub.lineno)
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if id(sub) not in bases:
                        self._link(me, self.symbol(module.name, sub.id), line)
                elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                    self._attribute(module, me, sub, cls, attributes, local, line)

    def _add(self, me, callee, line):
        self.edges[me].add(callee)
        self.calls[me].append((line, callee))

    def _link(self, me, found, line):
        if found is None:
            return
        if found[0] == "function":
            self._add(me, (found[1], found[2]), line)
        elif found[0] == "class":
            for name in ("__init__", "__post_init__", "__call__"):
                hit = self.method(found[1], found[2], name)
                if hit:
                    self._add(me, hit, line)

    def _attribute(self, module, me, node, cls, attributes, local, line):
        name, base = node.attr, node.value
        if isinstance(base, ast.Name):
            if base.id in ("self", "cls") and cls:
                hits = self.dispatch(module.name, cls, name)
                if hits:
                    for hit in hits:
                        self._add(me, hit, line)
                    return
            elif base.id in local:
                for hit in self.dispatch(*local[base.id], name):
                    self._add(me, hit, line)
                return
            else:
                found = self.symbol(module.name, base.id)
                if found and found[0] == "module":
                    self._link(me, self.symbol(found[1], name), line)
                    return
                if found and found[0] == "class":
                    hit = self.method(found[1], found[2], name)
                    if hit:
                        self._add(me, hit, line)
                    return
                target = module.imports.get(base.id)
                if target and target[1].split(".")[0] not in ("mars_titan", "scripts"):
                    return
        elif (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "self"
            and base.attr in attributes
        ):
            for hit in self.dispatch(*attributes[base.attr], name):
                self._add(me, hit, line)
            return
        if name.startswith("__") and name.endswith("__"):
            return
        for target in self.methods.get(name, ()):
            if target[0] in module.known:
                self._add(me, target, line)

    @staticmethod
    def _imported(module, func):
        """Nombre completo de lo que se llama, según las importaciones del módulo."""
        name = dotted(func)
        if not name:
            return None
        head, _, tail = name.partition(".")
        target = module.imports.get(head)
        if target is None:
            return None
        full = target[1] if target[0] == "module" else f"{target[1]}.{target[2]}"
        return full + ("." + tail if tail else "")

    def _fitting(self, module, call):
        func = call.func
        if not isinstance(func, (ast.Attribute, ast.Name)):
            return None
        attribute = isinstance(func, ast.Attribute)
        if (
            attribute
            and func.attr == "step"
            and not call.args
            and all(k.arg == "closure" for k in call.keywords)
        ):
            return "paso de optimizador"
        # Con `from xgboost import train` el nombre suelto resuelve al mismo destino.
        full = self._imported(module, func)
        if full:
            leaf = full.rsplit(".", 1)[-1]
            if full.startswith("torch.optim.") and leaf[:1].isupper():
                return "optimizador de torch.optim"
            if full == "xgboost.train":
                return "xgboost.train"
            if leaf in SOLVERS and (".linalg." in full or full.startswith("scipy.")):
                return "solución cerrada"
        if attribute and func.attr in FITS:
            receiver = func.value
            if isinstance(receiver, ast.Name):
                found = self.symbol(module.name, receiver.id)
                if found and found[0] in ("class", "module"):
                    return None
            return "fit de un estimador"
        return None

    # Comprobaciones.

    @property
    def guarded(self):
        return set(self.guards)

    def nodes(self):
        found = set(self.edges)
        for callees in self.edges.values():
            found |= callees
        for module in self.modules.values():
            found |= {(module.name, qualname) for qualname in module.functions}
            found.add((module.name, MODULE_SCOPE))
        return found

    @property
    def without_steps(self):
        """Funciones que instalan un gancho que rechaza los pasos antes de seguir."""
        return {node for node, callees in self.edges.items() if callees & self.hooks}

    def unprotected(self, *, exempt_modules=(), exempt_sites=()):
        """Sitios de ajuste alcanzables sin pasar por una guarda, con una cadena de ejemplo.

        Las funciones que instalan el gancho sin pasos solo detienen los sitios de PyTorch.
        """
        exempt = {node for node in self.nodes() if node[0] in set(exempt_modules)}
        found = {}
        for kinds, stop in (
            (TORCH_KINDS, self.guarded | self.without_steps | exempt),
            (None, self.guarded | exempt),
        ):
            sites = [
                site
                for site in self.sites
                if (site.kind in TORCH_KINDS) == (kinds is not None)
                and label(site.node) not in set(exempt_sites)
            ]
            for key, value in self._reached(stop, sites).items():
                found.setdefault(key, value)
        return found

    def _reached(self, stop, sites):
        callers = defaultdict(set)
        for caller, callees in self.edges.items():
            for callee in callees:
                if callee != caller:
                    callers[callee].add(caller)
        roots = [
            node
            for node in self.nodes()
            if node not in stop and (node[1] == MODULE_SCOPE or not callers[node])
        ]
        parent = dict.fromkeys(roots)
        queue = deque(roots)
        while queue:
            node = queue.popleft()
            for callee in self.edges.get(node, ()):
                if callee not in stop and callee not in parent:
                    parent[callee] = node
                    queue.append(callee)
        found = {}
        for site in sites:
            if site.node in parent:
                chain, step = [], site.node
                while step is not None:
                    chain.append(label(step))
                    step = parent[step]
                found.setdefault(label(site.node), (site, chain))
        return found

    def late_guards(self, *, exempt_sites=()):
        """Llamadas que pueden ajustar antes de la primera guarda de su función."""
        reaching = {site.node for site in self.sites if label(site.node) not in set(exempt_sites)}
        callers = defaultdict(set)
        for caller, callees in self.edges.items():
            for callee in callees:
                callers[callee].add(caller)
        queue = deque(reaching)
        while queue:
            node = queue.popleft()
            if node in self.guarded:
                continue
            for caller in callers[node]:
                if caller not in reaching:
                    reaching.add(caller)
                    queue.append(caller)
        late = []
        for node, lines in self.guards.items():
            first = min(lines)
            for line, callee in self.calls.get(node, ()):
                if callee in reaching and callee not in self.guarded and line < first:
                    late.append((label(node), first, line, label(callee)))
            for site in self.sites:
                if site.node == node and site.line < first:
                    late.append((label(node), first, site.line, site.kind))
        return sorted(set(late))
