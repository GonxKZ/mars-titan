// Estado de la vista en el fragmento de la URL, para que un enlace reproduzca la misma
// vista, la misma ejecución y el mismo tramo ampliado. El fragmento no se envía al
// servidor. Un valor desconocido se descarta en lugar de romper la página, porque un
// enlace puede venir de una versión anterior.

const VIEWS = ["campana", "curvas", "recursos", "memoria", "rl", "registros", "metodo"];
const IDENTIFIER = /^[\w.+:~-]{1,200}$/;
const RANGE = /^-?\d+(\.\d+)?~-?\d+(\.\d+)?$/;

export const DEFAULTS = Object.freeze({
  vista: "campana", serie: "", matriz: "", celda: "", ejecucion: "", medida: "mae", ventana: "", x: "",
  escala: "lin", marcas: "estado", buscar: "", estado: "", actividad: "", traza: "", rango: "1h", comun: "si",
});

const RULES = {
  vista: value => VIEWS.includes(value),
  serie: value => IDENTIFIER.test(value),
  matriz: value => IDENTIFIER.test(value),
  celda: value => /^[\w.+:|~-]{1,200}$/.test(value),
  ejecucion: value => IDENTIFIER.test(value),
  medida: value => ["mae", "session_mae", "train_mae", "loss", "train_samples_per_second"].includes(value),
  ventana: value => IDENTIFIER.test(value),
  x: value => RANGE.test(value) && Number(value.split("~")[0]) < Number(value.split("~")[1]),
  escala: value => ["lin", "log"].includes(value),
  marcas: value => ["estado", "mae", "relativo"].includes(value),
  buscar: value => value.length <= 120,
  estado: value => ["running", "stale", "queued", "paused", "completed", "failed", "cancelled", "blocked", "unknown"].includes(value),
  actividad: value => /^[a-z_]{1,40}$/.test(value),
  traza: value => IDENTIFIER.test(value),
  rango: value => ["15m", "1h", "6h", "24h"].includes(value),
  comun: value => ["si", "no"].includes(value),
};

export function parseHash(hash) {
  const state = { ...DEFAULTS };
  const raw = (hash ?? "").replace(/^#/, "");
  // Los enlaces antiguos apuntaban a secciones con #seguimiento o #historial.
  const legacy = { seguimiento: "campana", comparativa: "curvas", historial: "registros", metodo: "metodo" };
  if (Object.hasOwn(legacy, raw)) return { ...state, vista: legacy[raw] };
  const params = new URLSearchParams(raw);
  for (const [key, value] of params) {
    if (Object.hasOwn(RULES, key) && RULES[key](value)) state[key] = value;
  }
  return state;
}

export function serializeHash(state) {
  const params = new URLSearchParams();
  for (const key of Object.keys(DEFAULTS)) {
    const value = state[key];
    if (value === undefined || value === null || value === DEFAULTS[key] || value === "") continue;
    if (RULES[key](String(value))) params.set(key, String(value));
  }
  const text = params.toString();
  return text ? `#${text}` : "#";
}

export function runToken(run) {
  return `${run.run_id}~${run.attempt_id}`;
}

export function rangeToken(min, max) {
  const round = value => Number(value.toPrecision(10));
  return `${round(min)}~${round(max)}`;
}

export function parseRange(token) {
  if (!token || !RANGE.test(token)) return null;
  const [min, max] = token.split("~").map(Number);
  return min < max ? { min, max } : null;
}
