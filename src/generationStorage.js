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
    return { viewBox: vb, d: "", strokeWidth: 1 };
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
    strokeWidth: 1.2,
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

/**
 * @param {number[][]} path
 * @returns {{ id: string, createdAt: number, path: number[][] } | null}
 */
export function addGeneration(path) {
  if (!path || path.length < 2) return null;
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  return {
    id,
    createdAt: Date.now(),
    path,
  };
}
