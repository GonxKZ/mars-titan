// Recorrido animado del esquema de método. Un punto sigue el orden de la información
// una sola vez por pulsación: entradas, codificadores, núcleo, predicción, etiqueta
// madura y, solo después, escritura en memoria. Con movimiento reducido el punto no se
// desplaza y los pasos se iluminan uno tras otro.

// Recorrido en coordenadas del viewBox del SVG. El número indica el paso que se ilumina
// al llegar a cada punto.
const ROUTE = [
  [140, 148, 1], [236, 148, 2], [392, 148, 3], [632, 148, 3], [696, 148, 4], [804, 148, 5],
  [804, 100, 6], [650, 100, 6], [650, 156, 6], [622, 156, 6],
];
const SPEED = 140;
let running = null;

function steps(svg, upTo) {
  for (const node of svg.querySelectorAll("[data-step]")) node.classList.toggle("lit", Number(node.dataset.step) <= upTo);
}

export function playFlow() {
  const svg = document.getElementById("flow-diagram");
  const token = document.getElementById("flow-token");
  const button = document.getElementById("flow-play");
  if (!svg || running) return;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  button.disabled = true;
  const finish = () => {
    running = null;
    button.disabled = false;
    token.style.opacity = "0";
    setTimeout(() => steps(svg, 0), 1600);
  };
  if (reduced) {
    let step = 0;
    running = setInterval(() => {
      steps(svg, ++step);
      if (step >= 6) { clearInterval(running); finish(); }
    }, 700);
    return;
  }
  const lengths = ROUTE.slice(1).map(([x, y], i) => Math.hypot(x - ROUTE[i][0], y - ROUTE[i][1]));
  const total = lengths.reduce((a, b) => a + b, 0);
  const started = performance.now();
  token.style.opacity = "1";
  const frame = now => {
    // Con la pestaña oculta requestAnimationFrame se detiene, así que el recorrido no
    // consume nada mientras nadie lo ve.
    let travelled = (now - started) / 1000 * SPEED;
    if (travelled >= total) {
      const [x, y] = ROUTE.at(-1);
      token.setAttribute("cx", x);
      token.setAttribute("cy", y);
      steps(svg, 6);
      finish();
      return;
    }
    let segment = 0;
    while (travelled > lengths[segment]) travelled -= lengths[segment++];
    const [x0, y0] = ROUTE[segment], [x1, y1, step] = ROUTE[segment + 1];
    const t = travelled / lengths[segment];
    token.setAttribute("cx", x0 + (x1 - x0) * t);
    token.setAttribute("cy", y0 + (y1 - y0) * t);
    steps(svg, t > .9 ? step : ROUTE[segment][2]);
    running = requestAnimationFrame(frame);
  };
  running = requestAnimationFrame(frame);
}
