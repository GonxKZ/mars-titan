// Reducción de puntos para dibujar series largas.
//
// Se usa la agregación M4 de Jugel et al. (2014): en cada columna de píxeles se conservan
// el primer punto, el último, el mínimo y el máximo. Una línea dibujada con esos cuatro
// puntos ocupa los mismos píxeles que la serie completa, así que no se pierden picos.
// La función devuelve índices de la serie original. Así los puntos dibujados son medidas
// reales y la lectura exacta del cursor se hace siempre sobre la serie completa.
//
// Un NaN es una ausencia. Si una columna contiene alguna, se conserva el índice de la
// primera para que la línea se corte allí y el hueco no se rellene.

export function lowerBound(x, value, start = 0, end = x.length) {
  let low = start, high = end;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (x[middle] < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

export function nearestIndex(x, value) {
  if (!x.length) return -1;
  const index = lowerBound(x, value);
  if (index === 0) return 0;
  if (index >= x.length) return x.length - 1;
  return value - x[index - 1] <= x[index] - value ? index - 1 : index;
}

export function m4Indices(x, y, x0, x1, columns) {
  const start = Math.max(0, lowerBound(x, x0) - 1);
  const end = Math.min(x.length, lowerBound(x, x1) + 1);
  if (end - start <= columns * 4 || !(x1 > x0) || columns < 1) {
    const all = new Uint32Array(end - start);
    for (let i = 0; i < all.length; i++) all[i] = start + i;
    return all;
  }
  const output = new Uint32Array(columns * 5 + 4);
  let size = 0;
  const scale = columns / (x1 - x0);
  let bucket = -1, first = -1, last = -1, low = -1, high = -1, gap = -1;
  const flush = () => {
    if (first < 0 && gap < 0) return;
    const picks = [first, low, high, last, gap].filter(index => index >= 0).sort((a, b) => a - b);
    let previous = -1;
    for (const index of picks) if (index !== previous) { output[size++] = index; previous = index; }
  };
  for (let i = start; i < end; i++) {
    const column = Math.min(columns - 1, Math.max(0, Math.floor((x[i] - x0) * scale)));
    if (column !== bucket) {
      flush();
      bucket = column; first = last = low = high = gap = -1;
    }
    const value = y[i];
    if (Number.isNaN(value)) {
      if (gap < 0) gap = i;
      continue;
    }
    if (first < 0) first = low = high = i;
    last = i;
    if (value < y[low]) low = i;
    if (value > y[high]) high = i;
  }
  flush();
  return output.subarray(0, size);
}

// Copia los puntos elegidos a columnas nuevas. NaN pasa a null porque uPlot interpreta
// null como hueco en la línea.
export function pick(x, ys, indices) {
  const outX = new Float64Array(indices.length);
  const outYs = ys.map(() => new Array(indices.length));
  for (let k = 0; k < indices.length; k++) {
    const index = indices[k];
    outX[k] = x[index];
    for (let s = 0; s < ys.length; s++) {
      const value = ys[s][index];
      outYs[s][k] = Number.isNaN(value) ? null : value;
    }
  }
  return [outX, ...outYs];
}
