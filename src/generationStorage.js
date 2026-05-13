import bundledGenerations from "./data/itp-spring-show-2026.json";

function bboxFromPoints(pts) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    const x = p[0];
    const y = p[1];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX)) {
    return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  }
  return { minX, minY, maxX, maxY };
}

/**
 * SVG line for bubble previews (same geometry as the old raster thumb).
 * @param {number[][]} path
 * @param {number} size viewBox / coordinate space size
 * @returns {{ viewBox: string, d: string, strokeWidth: number }}
 */
export function pathToBubbleSvg(path, size = 100) {
  const vb = `0 0 ${size} ${size}`;
  if (!path || path.length < 2) {
    return { viewBox: vb, d: "", strokeWidth: 1.2 * (size / 100) };
  }
  const bb = bboxFromPoints(path);
  const pad = 8;
  const w = bb.maxX - bb.minX || 1;
  const h = bb.maxY - bb.minY || 1;
  const scale = (size - 2 * pad) / Math.max(w, h);
  const parts = [];
  for (let i = 0; i < path.length; i++) {
    const px = pad + (path[i][0] - bb.minX) * scale;
    const py = pad + (path[i][1] - bb.minY) * scale;
    parts.push(`${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`);
  }
  return {
    viewBox: vb,
    d: parts.join(" "),
    /** Same relative weight at any `size` (1.2 was tuned for the default 100× preview). */
    strokeWidth: 1.2 * (size / 100),
  };
}

/**
 * @param {number[][][]} segments
 */
function bboxFromSegments(segments) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const seg of segments) {
    if (!seg) continue;
    for (const p of seg) {
      if (!p || p.length < 2) continue;
      const x = p[0];
      const y = p[1];
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (!Number.isFinite(minX)) {
    return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  }
  return { minX, minY, maxX, maxY };
}

/**
 * Multiple contours in one SVG `d` (`M…L…` then `M…L…`), same fit as {@link pathToBubbleSvg}.
 * @param {number[][][]} segments
 * @param {number} size
 * @returns {{ viewBox: string, d: string, strokeWidth: number }}
 */
export function pathSegmentsToBubbleSvg(segments, size = 100) {
  const vb = `0 0 ${size} ${size}`;
  const nonempty = (segments ?? []).filter((s) => s && s.length >= 2);
  if (nonempty.length === 0) {
    return { viewBox: vb, d: "", strokeWidth: 1.2 * (size / 100) };
  }
  const bb = bboxFromSegments(nonempty);
  const pad = 8;
  const w = bb.maxX - bb.minX || 1;
  const h = bb.maxY - bb.minY || 1;
  const scale = (size - 2 * pad) / Math.max(w, h);
  const parts = [];
  for (const seg of nonempty) {
    for (let i = 0; i < seg.length; i++) {
      const px = pad + (seg[i][0] - bb.minX) * scale;
      const py = pad + (seg[i][1] - bb.minY) * scale;
      parts.push(`${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`);
    }
  }
  return {
    viewBox: vb,
    d: parts.join(" "),
    strokeWidth: 1.2 * (size / 100),
  };
}

/** @param {unknown[]} data */
function normalizeGenerations(data) {
  if (!Array.isArray(data)) return [];
  return data
    .filter(
      (x) =>
        x &&
        typeof x.id === "string" &&
        Array.isArray(x.path) &&
        x.path.length >= 2,
    )
    .map((x) => ({
      id: x.id,
      createdAt: typeof x.createdAt === "number" ? x.createdAt : 0,
      path: x.path,
    }));
}

function generationsFromBundled(raw) {
  if (Array.isArray(raw)) return normalizeGenerations(raw);
  if (raw && typeof raw === "object" && Array.isArray(raw["bioglyph-generations-v1"])) {
    return normalizeGenerations(raw["bioglyph-generations-v1"]);
  }
  return [];
}

/** @returns {{ id: string, createdAt: number, path: number[][] }[]} */
export function loadGenerations() {
  return generationsFromBundled(bundledGenerations);
}

