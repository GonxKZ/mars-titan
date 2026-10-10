// Obtención de datos en los dos modos de la página.
//
// En Pages, el índice se consulta con If-None-Match e If-Modified-Since. Una respuesta
// 304 no descarga ni vuelve a validar nada. Las páginas del historial tienen nombre por
// huella, son inmutables y se piden una sola vez.
//
// En local, el servidor empuja eventos SSE. El evento de índice solo trae su ETag y la
// página descarga el índice con la misma petición condicional que en Pages.

export const MAX_BYTES = 8 * 1024 * 1024;

export async function readLimited(response, maxBytes = MAX_BYTES) {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maxBytes) throw new Error(`La respuesta supera el límite de ${maxBytes / 1048576} MiB`);
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) {
        await reader.cancel();
        throw new Error(`La respuesta supera el límite de ${maxBytes / 1048576} MiB`);
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const output = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.byteLength; }
  return output;
}

export class ConditionalResource {
  constructor(url, { fetchImpl = globalThis.fetch.bind(globalThis), maxBytes = MAX_BYTES, timeoutMs = 15000 } = {}) {
    this.url = url;
    this.fetch = fetchImpl;
    this.maxBytes = maxBytes;
    this.timeoutMs = timeoutMs;
    this.etag = null;
    this.lastModified = null;
    this.requests = 0;
    this.notModified = 0;
  }

  // Devuelve null si el recurso no ha cambiado. El cuerpo se lee con límite de bytes.
  async get(signal) {
    const headers = { Accept: "application/json" };
    if (this.etag) headers["If-None-Match"] = this.etag;
    else if (this.lastModified) headers["If-Modified-Since"] = this.lastModified;
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(abort, this.timeoutMs);
    this.requests++;
    try {
      // no-store evita que la caché HTTP del navegador responda por su cuenta y deja
      // a esta clase el control de las cabeceras condicionales.
      const response = await this.fetch(this.url, { cache: "no-store", credentials: "omit", headers, signal: controller.signal });
      if (response.status === 304) {
        this.notModified++;
        return null;
      }
      if (!response.ok) throw new Error(`El servidor respondió con HTTP ${response.status}`);
      const body = await readLimited(response, this.maxBytes);
      const data = JSON.parse(new TextDecoder().decode(body));
      this.etag = response.headers.get("etag");
      this.lastModified = response.headers.get("last-modified");
      return { data, bytes: body.byteLength, etag: this.etag, lastModified: this.lastModified, date: response.headers.get("date") };
    } catch (error) {
      if (controller.signal.aborted && !signal?.aborted) throw new Error(`La petición superó ${this.timeoutMs / 1000} s`);
      throw error;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  }
}

// Páginas inmutables con concurrencia acotada. Una página fallida no se guarda y se
// puede volver a pedir.
export class PageStore {
  constructor(baseUrl, validate, { fetchImpl = globalThis.fetch.bind(globalThis), concurrency = 4 } = {}) {
    this.baseUrl = baseUrl;
    this.validate = validate;
    this.fetch = fetchImpl;
    this.concurrency = concurrency;
    this.pages = new Map();
    this.inflight = new Map();
    this.failed = new Set();
  }

  async one(path) {
    if (this.pages.has(path)) return this.pages.get(path);
    if (this.inflight.has(path)) return this.inflight.get(path);
    const promise = (async () => {
      // Las páginas se nombran por su huella y casi siempre salen de la caché. Un reintento
      // va directo al servidor, porque Chrome puede guardar una respuesta 503 y devolverla
      // de nuevo si se pide con force-cache.
      const response = await this.fetch(new URL(path, this.baseUrl), { cache: this.failed.has(path) ? "reload" : "default", credentials: "omit" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const document = this.validate(JSON.parse(new TextDecoder().decode(await readLimited(response))));
      this.pages.set(path, document);
      this.failed.delete(path);
      return document;
    })();
    this.inflight.set(path, promise);
    try {
      return await promise;
    } catch (error) {
      this.failed.add(path);
      throw error;
    } finally {
      this.inflight.delete(path);
    }
  }

  async all(paths, onProgress = () => {}) {
    const queue = [...paths];
    let done = 0, errors = 0;
    const worker = async () => {
      while (queue.length) {
        const path = queue.shift();
        try { await this.one(path); } catch { errors++; }
        onProgress(++done, paths.length, errors);
      }
    };
    await Promise.all(Array.from({ length: Math.min(this.concurrency, paths.length) }, worker));
    return { done, errors };
  }
}

// Conexión SSE con el servidor local. Se cierra con la pestaña oculta para no mantener
// trabajo en segundo plano y al volver se recibe de nuevo el estado completo.
export class LiveStream {
  constructor(url, handlers, { EventSourceImpl = globalThis.EventSource } = {}) {
    this.url = url;
    this.handlers = handlers;
    this.EventSource = EventSourceImpl;
    this.source = null;
    this.lastEvent = null;
    this.state = "closed";
    this.failures = 0;
  }

  open() {
    if (this.source) return;
    const source = new this.EventSource(this.url);
    this.source = source;
    this.state = "connecting";
    this.handlers.state?.(this.state);
    for (const name of ["sync", "index", "telemetry", "campaign", "traces"]) {
      source.addEventListener(name, event => {
        this.lastEvent = Date.now();
        this.failures = 0;
        if (this.state !== "open") { this.state = "open"; this.handlers.state?.(this.state); }
        let payload;
        try { payload = JSON.parse(event.data); } catch { return; }
        this.handlers[name]?.(payload);
      });
    }
    source.onopen = () => { this.state = "open"; this.handlers.state?.(this.state); };
    source.onerror = () => {
      this.failures++;
      // Un 503 por exceso de clientes cierra la conexión de forma definitiva.
      this.state = source.readyState === 2 ? "closed" : "reconnecting";
      if (this.state === "closed") this.source = null;
      this.handlers.state?.(this.state, this.failures);
    };
  }

  close() {
    this.source?.close();
    this.source = null;
    this.state = "closed";
  }
}
