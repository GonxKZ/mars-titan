// Planificador de trabajo visual. Agrupa las actualizaciones en un único fotograma de
// requestAnimationFrame y no dibuja nada mientras la pestaña está oculta. El portátil
// trabaja en modo de ahorro de energía, así que cada repintado evitado cuenta.

export class FrameScheduler {
  constructor({ raf = globalThis.requestAnimationFrame?.bind(globalThis), doc = globalThis.document } = {}) {
    this.raf = raf;
    this.doc = doc;
    this.tasks = new Map();
    this.pending = false;
    this.frames = 0;
    this.deferred = false;
    doc?.addEventListener?.("visibilitychange", () => {
      if (!doc.hidden && this.deferred) { this.deferred = false; this.request(); }
    });
  }

  // Una clave solo se ejecuta una vez por fotograma aunque se pida muchas veces.
  schedule(key, task) {
    this.tasks.set(key, task);
    this.request();
  }

  request() {
    if (this.pending) return;
    if (this.doc?.hidden) { this.deferred = true; return; }
    this.pending = true;
    this.raf(() => this.flush());
  }

  flush() {
    this.pending = false;
    if (this.doc?.hidden) { this.deferred = true; return; }
    const tasks = [...this.tasks.values()];
    this.tasks.clear();
    this.frames++;
    for (const task of tasks) {
      try { task(); } catch (error) { globalThis.reportError?.(error); }
    }
  }
}

// Limita una función a una ejecución por intervalo y conserva la última llamada, de modo
// que el estado final nunca se pierde.
export function throttle(fn, interval, { now = () => performance.now(), setTimer = setTimeout } = {}) {
  let last = -Infinity, timer = null, args = null;
  return (...next) => {
    args = next;
    const wait = last + interval - now();
    if (wait <= 0 && !timer) {
      last = now();
      fn(...args);
      return;
    }
    if (!timer) timer = setTimer(() => { timer = null; last = now(); fn(...args); }, Math.max(0, wait));
  };
}
