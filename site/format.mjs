// Formatos de números, fechas y duraciones en español de España. Todas las fechas se
// muestran con su zona horaria explícita, porque un registro sin zona no permite saber
// si una ejecución es reciente.

const cache = new Map();

function formatter(key, options) {
  if (!cache.has(key)) cache.set(key, new Intl.NumberFormat("es-ES", options));
  return cache.get(key);
}

export const MISSING = "Sin dato";

export function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function number(value, digits = 0) {
  if (!isNumber(value)) return MISSING;
  return formatter(`f${digits}`, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

// Cifras significativas para errores pequeños como un MAE de 0,0149. Con un número fijo
// de decimales dos valores distintos podrían parecer iguales.
export function significant(value, digits = 4) {
  if (!isNumber(value)) return MISSING;
  if (value !== 0 && (Math.abs(value) >= 1e6 || Math.abs(value) < 1e-4)) {
    return formatter(`e${digits}`, { notation: "scientific", maximumSignificantDigits: digits }).format(value);
  }
  return formatter(`s${digits}`, { maximumSignificantDigits: digits, minimumSignificantDigits: digits }).format(value);
}

export function percent(value, digits = 1) {
  if (!isNumber(value)) return MISSING;
  return formatter(`p${digits}`, { style: "percent", minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

export function bytes(mib) {
  if (!isNumber(mib)) return MISSING;
  return mib >= 1024 ? `${number(mib / 1024, 1)} GiB` : `${number(mib, 0)} MiB`;
}

export function duration(seconds) {
  if (!isNumber(seconds) || seconds < 0) return MISSING;
  if (seconds < 60) return `${number(seconds, seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60), rest = minutes % 60;
  if (hours < 48) return rest ? `${hours} h ${rest} min` : `${hours} h`;
  const days = Math.floor(hours / 24);
  return `${days} d ${hours % 24} h`;
}

export function age(milliseconds) {
  if (!isNumber(milliseconds)) return MISSING;
  if (milliseconds < 0) return "con fecha futura";
  const seconds = Math.round(milliseconds / 1000);
  if (seconds < 5) return "ahora";
  if (seconds < 60) return `hace ${seconds} s`;
  return `hace ${duration(seconds)}`;
}

const dateFormats = new Map();

function dateFormat(timeZone, withSeconds) {
  const key = `${timeZone}|${withSeconds}`;
  if (!dateFormats.has(key)) {
    dateFormats.set(key, new Intl.DateTimeFormat("es-ES", {
      day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
      ...(withSeconds ? { second: "2-digit" } : {}), timeZone, timeZoneName: "short",
    }));
  }
  return dateFormats.get(key);
}

// Fecha local con su zona y, aparte, la misma fecha en UTC para el atributo title.
export function timestamp(value, { seconds = false, timeZone } = {}) {
  const time = typeof value === "number" ? value : Date.parse(value ?? "");
  if (!Number.isFinite(time)) return { text: "Sin fecha", utc: "" };
  return {
    text: dateFormat(timeZone, seconds).format(time),
    utc: new Date(time).toISOString().replace(".000Z", "Z"),
  };
}

export function shortHash(value, length = 12) {
  return typeof value === "string" && value.length > length ? value.slice(0, length) : value ?? MISSING;
}
