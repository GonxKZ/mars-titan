// Lectura de los paquetes de trazas de aprendizaje (#448).
//
// Un paquete es un manifiesto JSON y un bloque binario little-endian. Las columnas se
// leen como vistas tipadas sobre el bloque, sin copiar ni analizar texto, lo que permite
// abrir millones de puntos. Antes de usar el bloque se comprueban su tamaño y su huella
// SHA-256, de modo que una cifra mostrada corresponde al paquete declarado.

export const TRACE_GROUPS = Object.freeze({
  optimization: "Optimización", titans: "Memoria de Titans", episodic: "Banco episódico",
  rl: "Aprendizaje por refuerzo", session: "Por sesión", adapters: "Adaptadores", modalities: "Modalidades",
});
const X_UNITS = { optimizer_step: "Paso del optimizador", epoch: "Época", session: "Sesión" };
const DTYPES = { f64: [Float64Array, 8], f32: [Float32Array, 4] };
const NAME = /^[A-Za-z0-9][\w.-]{0,80}$/;
const BLOB = /^[A-Za-z0-9][\w.-]{0,80}-[a-f0-9]{16}\.bin$/;
const ID = /^[a-z][a-z0-9_.]{0,95}$/;

function fail(message) {
  throw new TypeError(`Trazas: ${message}.`);
}

function text(value, label, maximum = 120) {
  if (typeof value !== "string" || !value.trim() || value.length > maximum) fail(`${label} no válido`);
  return value;
}

function reference(value, label, bytes, dtype) {
  if (!value || typeof value !== "object") fail(`${label} sin referencia`);
  const { offset, length } = value;
  if (value.dtype !== dtype || !Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 1) fail(`${label} con referencia inválida`);
  if (offset % 8 !== 0 || offset + length * DTYPES[dtype][1] > bytes) fail(`${label} fuera del bloque o sin alinear`);
  return { offset, length, dtype };
}

export function validateManifest(input) {
  if (!input || typeof input !== "object" || input.schema_version !== 1 || input.kind !== "mars_titan_learning_traces") fail("manifiesto de otra versión");
  if (!NAME.test(input.name ?? "") || !BLOB.test(input.blob ?? "")) fail("nombre del paquete o del bloque no admitido");
  if (!["measured", "fixture"].includes(input.provenance)) fail("procedencia desconocida");
  if (!Object.hasOwn(X_UNITS, input.x_unit)) fail("unidad del eje desconocida");
  if (!Number.isSafeInteger(input.blob_bytes) || input.blob_bytes < 0 || !/^[a-f0-9]{64}$/.test(input.blob_sha256 ?? "")) fail("tamaño o huella del bloque");
  const bytes = input.blob_bytes, seen = new Set();
  const head = (item, label) => {
    if (!ID.test(item?.id ?? "") || seen.has(item.id)) fail(`${label} con identificador repetido o inválido`);
    if (!Object.hasOwn(TRACE_GROUPS, item.group)) fail(`${label} con grupo desconocido`);
    seen.add(item.id);
    return { id: item.id, group: item.group, label: text(item.label, `${label}.label`), unit: text(item.unit, `${label}.unit`, 40) };
  };
  const series = (input.series ?? []).map((item, index) => {
    const label = `series[${index}]`;
    const result = { ...head(item, label), x: reference(item.x, `${label}.x`, bytes, "f64"), y: reference(item.y, `${label}.y`, bytes, "f32") };
    if (result.x.length !== result.y.length) fail(`${label} con longitudes distintas`);
    return result;
  });
  const matrices = (input.matrices ?? []).map((item, index) => {
    const label = `matrices[${index}]`;
    if (!Array.isArray(item.rows) || !item.rows.length || item.rows.length > 512) fail(`${label} sin filas`);
    const result = {
      ...head(item, label), rows: item.rows.map((row, i) => text(row, `${label}.rows[${i}]`, 40)),
      x: reference(item.x, `${label}.x`, bytes, "f64"), values: reference(item.values, `${label}.values`, bytes, "f32"),
      range: item.range == null ? null : item.range,
    };
    if (result.values.length !== result.rows.length * result.x.length) fail(`${label} con tamaño incoherente`);
    if (result.range && !(Array.isArray(result.range) && result.range.length === 2 && result.range.every(Number.isFinite) && result.range[0] < result.range[1])) fail(`${label} con rango inválido`);
    return result;
  });
  return {
    name: input.name, blob: input.blob, bytes, sha256: input.blob_sha256, provenance: input.provenance,
    xUnit: input.x_unit, xLabel: X_UNITS[input.x_unit], runId: text(input.run_id, "run_id", 96),
    attemptId: text(input.attempt_id, "attempt_id", 96), modelId: text(input.model_id, "model_id", 96),
    cadence: input.cadence ?? null, series, matrices,
  };
}

export async function sha256Hex(buffer, subtle = globalThis.crypto?.subtle) {
  if (!subtle) return null;
  const digest = new Uint8Array(await subtle.digest("SHA-256", buffer));
  return Array.from(digest, byte => byte.toString(16).padStart(2, "0")).join("");
}

function view(buffer, ref) {
  const [Type] = DTYPES[ref.dtype];
  return new Type(buffer, ref.offset, ref.length);
}

function increasing(x, label) {
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i]) || (i > 0 && x[i] <= x[i - 1])) fail(`${label}: eje x no creciente`);
  }
}

// Une manifiesto y bloque. El bloque debe tener exactamente el tamaño y la huella del
// manifiesto. Sin crypto.subtle (origen no seguro) la huella queda como no comprobada.
export async function decodeBundle(manifest, buffer, { subtle } = {}) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== manifest.bytes) fail("el bloque no tiene el tamaño declarado");
  const digest = await sha256Hex(buffer, subtle);
  if (digest !== null && digest !== manifest.sha256) fail("la huella del bloque no coincide");
  const series = manifest.series.map(item => {
    const x = view(buffer, item.x);
    increasing(x, item.id);
    const y = view(buffer, item.y);
    for (const value of y) if (value === Infinity || value === -Infinity) fail(`${item.id}: valor infinito`);
    return { ...item, x, y };
  });
  const matrices = manifest.matrices.map(item => {
    const x = view(buffer, item.x);
    increasing(x, item.id);
    return { ...item, x, values: view(buffer, item.values) };
  });
  return { ...manifest, verified: digest !== null, series, matrices, points: series.reduce((sum, s) => sum + s.x.length, 0) + matrices.reduce((sum, m) => sum + m.values.length, 0) };
}

export function validateIndex(input) {
  if (!input || input.schema_version !== 1 || !Array.isArray(input.bundles) || input.bundles.length > 4096) fail("índice de trazas no válido");
  return input.bundles.map((entry, index) => {
    if (!NAME.test(entry?.name ?? "") || entry.manifest !== `${entry.name}.json`) fail(`bundles[${index}] no válido`);
    return { name: entry.name, manifest: entry.manifest, runId: entry.run_id ?? null, attemptId: entry.attempt_id ?? null, modelId: entry.model_id ?? null, provenance: entry.provenance ?? null };
  });
}
