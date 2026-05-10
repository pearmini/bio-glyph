function expim(im) {
  return [Math.cos(im), Math.sin(im)];
}

function add([rea, ima], [reb, imb]) {
  return [rea + reb, ima + imb];
}

function mul([rea, ima], [reb, imb]) {
  return [rea * reb - ima * imb, rea * imb + ima * reb];
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

/** Matches app chrome (`index.css` / `.app-root`). */
const CANVAS_BG = "#f6f6f6";

function bboxFromPoints(pts) {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const [x, y] of pts) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  return { minX, minY, maxX, maxY };
}

function cumulativeLengths(path) {
  const n = path.length;
  const L = new Float64Array(n);
  let acc = 0;
  L[0] = 0;
  for (let i = 1; i < n; i++) {
    const dx = path[i][0] - path[i - 1][0];
    const dy = path[i][1] - path[i - 1][1];
    acc += Math.hypot(dx, dy);
    L[i] = acc;
  }
  return { L, total: acc };
}

function sampleAtLength(path, cumL, total, s) {
  if (path.length === 0) return [0, 0];
  if (path.length === 1) return [path[0][0], path[0][1]];
  const target = clamp(s, 0, 1) * total;
  const L = cumL;
  let lo = 0,
    hi = L.length - 1;
  while (lo < hi) {
    const mid = ((lo + hi) / 2) | 0;
    if (L[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  const i = Math.max(1, lo);
  const l0 = L[i - 1];
  const l1 = L[i];
  const t = l1 === l0 ? 0 : (target - l0) / (l1 - l0);
  const a = path[i - 1];
  const b = path[i];
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

function resamplePathByArcLength(path, N) {
  const clean = [];
  for (const p of path) {
    if (!p || p.length < 2) continue;
    const x = Number(p[0]);
    const y = Number(p[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (clean.length) {
      const q = clean[clean.length - 1];
      if (Math.abs(q[0] - x) < 1e-9 && Math.abs(q[1] - y) < 1e-9) continue;
    }
    clean.push([x, y]);
  }
  if (clean.length < 2) return clean;
  const { L, total } = cumulativeLengths(clean);
  if (!Number.isFinite(total) || total <= 1e-6) return clean.slice(0, N);
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    out[i] = sampleAtLength(clean, L, total, i / N);
  }
  return out;
}

function centeredComplexFromPoints(pts) {
  const b = bboxFromPoints(pts);
  const cx = (b.minX + b.maxX) * 0.5;
  const cy = (b.minY + b.maxY) * 0.5;
  return pts.map(([x, y]) => [x - cx, y - cy]);
}

function kSequence(M) {
  const K = new Int16Array(M);
  for (let i = 0; i < M; i++) {
    const v = ((1 + i) >> 1) * (i & 1 ? -1 : 1);
    K[i] = v;
  }
  return K;
}

function computeDFT(P, K) {
  const N = P.length;
  const DFT = new Array(K.length);
  for (let ki = 0; ki < K.length; ki++) {
    const k = K[ki];
    let x = [0, 0];
    for (let i = 0; i < N; i++) {
      const a = (k * i * 2 * -Math.PI) / N;
      x = add(x, mul(P[i], expim(a)));
    }
    DFT[ki] = [x[0] / N, x[1] / N];
  }
  return DFT;
}

function fitToCanvasTransform(width, height, points, margin = 0.86) {
  const W = width;
  const H = height;
  const b = bboxFromPoints(points);
  const bw = Math.max(b.maxX - b.minX, 1e-6);
  const bh = Math.max(b.maxY - b.minY, 1e-6);
  const scale = Math.min(W / bw, H / bh) * margin;
  return { scale, cx: (b.minX + b.maxX) * 0.5, cy: (b.minY + b.maxY) * 0.5 };
}

function findLargestWrapJumpIndex(pts) {
  if (pts.length < 3) return 0;
  let bestI = 0;
  let bestD = -Infinity;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (d > bestD) {
      bestD = d;
      bestI = i;
    }
  }
  return bestI;
}

/**
 * Progressive Fourier-series animation for a single closed-ish polyline.
 * Call returns a cleanup function that stops animation.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {number[][]} oneLinePath points in image pixel space
 * @param {{
 *  samples?: number,
 *  epicycles?: number,
 *  outSamples?: number,
 *  fadeAlpha?: number,
 *  strokeStyle?: string,
 *  lineWidth?: number,
 *  coeffsPerSecond?: number,
 *  loop?: boolean,
 *  closeStroke?: boolean,
 *  seamGapFraction?: number,
 *  autoSeam?: boolean,
 *  onComplete?: () => void,
 *  devicePixelRatio?: number,
 *  maxDevicePixelRatio?: number,
 * }} [opts]
 */
export function startFourierOneLineAnimation(canvas, oneLinePath, opts = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return () => {};

  let cancelled = false;

  const maxDpr = clamp(opts.maxDevicePixelRatio ?? 3, 1, 4);
  const winDpr = typeof window !== "undefined" ? window.devicePixelRatio : 1;
  const dpr = Math.max(1, Math.min(opts.devicePixelRatio ?? winDpr ?? 1, maxDpr));

  let cssW = canvas.clientWidth;
  let cssH = canvas.clientHeight;
  if (cssW < 1 || cssH < 1) {
    const parent = canvas.parentElement;
    const r = parent?.getBoundingClientRect();
    if (r && r.width > 0 && r.height > 0) {
      cssW = Math.max(1, Math.floor(r.width));
      cssH = Math.max(1, Math.floor(r.height));
    } else {
      cssW = Math.max(1, canvas.width);
      cssH = Math.max(1, canvas.height);
    }
  }

  canvas.width = Math.max(1, Math.floor(cssW * dpr));
  canvas.height = Math.max(1, Math.floor(cssH * dpr));
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const samples = clamp(opts.samples ?? 2048, 256, 8192);
  const M = clamp(opts.epicycles ?? 300, 16, 800);
  const q = clamp(opts.outSamples ?? 1400, 256, 4000);
  const fadeAlpha = clamp(opts.fadeAlpha ?? 0.04, 0, 1);
  const strokeStyle = opts.strokeStyle ?? "#141414";
  const lineWidth = opts.lineWidth ?? 2.25;
  const coeffsPerSecond = clamp(opts.coeffsPerSecond ?? 90, 10, 800);
  const loop = opts.loop ?? false;
  const closeStroke = opts.closeStroke ?? false;
  const autoSeam = opts.autoSeam ?? true;
  const seamGapFraction = clamp(opts.seamGapFraction ?? 0.07, 0, 0.5);

  const P0 = resamplePathByArcLength(oneLinePath, samples);
  if (P0.length < 4) {
    ctx.fillStyle = CANVAS_BG;
    ctx.fillRect(0, 0, cssW, cssH);
    if (typeof opts.onComplete === "function") opts.onComplete();
    return () => {};
  }

  // Center in its own coordinate system for a nicer reconstruction.
  const P = centeredComplexFromPoints(P0);
  const K = kSequence(M);
  const DFT = computeDFT(P, K);

  const { scale } = fitToCanvasTransform(cssW, cssH, P, 0.88);
  const seamIdx = autoSeam ? findLargestWrapJumpIndex(P) : 0;
  const phaseOffset = (-2 * Math.PI * seamIdx) / Math.max(1, P.length);

  let raf = 0;
  let startT = 0;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const drawFrame = (tMs) => {
    if (!startT) startT = tMs;
    const elapsed = (tMs - startT) / 1000;

    let m = Math.floor(elapsed * coeffsPerSecond);
    if (loop) m = m % (M + 1);
    else m = clamp(m, 0, M);

    // Slight trailing effect like the Observable reference.
    ctx.save();
    ctx.fillStyle = `rgba(246, 246, 246, ${fadeAlpha})`;
    ctx.fillRect(0, 0, cssW, cssH);

    ctx.translate(cssW / 2, cssH / 2);
    ctx.scale(scale, scale);

    ctx.beginPath();
    for (let ti = 0; ti < q; ti++) {
      const u = ti / q;
      const inGap = seamGapFraction > 0 && (u < seamGapFraction * 0.5 || u > 1 - seamGapFraction * 0.5);
      const a = (ti * 2 * Math.PI) / q + phaseOffset;
      let p = [0, 0];
      for (let i = 0; i < m; i++) {
        p = add(p, mul(DFT[i], expim(a * K[i])));
      }
      if (ti === 0 || inGap) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    }
    if (closeStroke) ctx.closePath();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth / scale;
    ctx.stroke();
    ctx.restore();

    if (loop || m < M) raf = requestAnimationFrame(drawFrame);
    else {
      raf = 0;
      if (!cancelled && typeof opts.onComplete === "function") opts.onComplete();
    }
  };

  // Initialize background so the first fade doesn't start from transparent.
  ctx.fillStyle = CANVAS_BG;
  ctx.fillRect(0, 0, cssW, cssH);
  raf = requestAnimationFrame(drawFrame);

  return () => {
    cancelled = true;
    if (raf) cancelAnimationFrame(raf);
  };
}

