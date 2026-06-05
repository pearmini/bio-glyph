const MAX_POINTS = 5000;
const MAX_JSON_BYTES = 500_000;

export function validatePath(path) {
  if (!Array.isArray(path) || path.length < 2) {
    return { ok: false, error: "Face path is too short to save." };
  }
  if (path.length > MAX_POINTS) {
    return { ok: false, error: "Face path is too large to save." };
  }
  for (const p of path) {
    if (!Array.isArray(p) || p.length < 2 || typeof p[0] !== "number" || typeof p[1] !== "number") {
      return { ok: false, error: "Invalid face path data." };
    }
  }
  const serialized = JSON.stringify(path);
  if (serialized.length > MAX_JSON_BYTES) {
    return { ok: false, error: "Face path is too large to save." };
  }
  return { ok: true, path };
}
