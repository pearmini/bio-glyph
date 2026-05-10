import { FaceLandmarker, FilesetResolver, ImageSegmenter } from "@mediapipe/tasks-vision";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const SEG_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";

let loadPromise = null;

/**
 * Lazily create Face Landmarker + Image Segmenter once; reuse for every extraction.
 * Shared promise avoids duplicate work under React StrictMode or parallel calls.
 */
function ensureMediaPipeModels() {
  if (!loadPromise) {
    loadPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
      /** Only one face result: MediaPipe's strongest / dominant face when several appear in frame. */
      const faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: "CPU" },
        runningMode: "IMAGE",
        numFaces: 1,
      });
      const imageSegmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: SEG_MODEL, delegate: "CPU" },
        runningMode: "IMAGE",
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
      return { faceLandmarker, imageSegmenter, FaceLandmarker };
    })();
  }
  return loadPromise;
}

/**
 * @mediapipe/tasks-vision JS omits `FaceLandmarker.FACE_LANDMARKS_NOSE` (it is undefined).
 * Same edges as Python `FaceLandmarksConnections.FACE_LANDMARKS_NOSE` so nose + lips both draw.
 */
const FACE_LANDMARKS_NOSE = [
  { start: 168, end: 6 },
  { start: 6, end: 197 },
  { start: 197, end: 195 },
  { start: 195, end: 5 },
  { start: 5, end: 4 },
  { start: 4, end: 1 },
  { start: 1, end: 19 },
  { start: 19, end: 94 },
  { start: 94, end: 2 },
  { start: 98, end: 97 },
  { start: 97, end: 2 },
  { start: 2, end: 326 },
  { start: 326, end: 327 },
  { start: 327, end: 294 },
  { start: 294, end: 278 },
  { start: 278, end: 344 },
  { start: 344, end: 440 },
  { start: 440, end: 275 },
  { start: 275, end: 4 },
  { start: 4, end: 45 },
  { start: 45, end: 220 },
  { start: 220, end: 115 },
  { start: 115, end: 48 },
  { start: 48, end: 64 },
  { start: 64, end: 98 },
];

/** Ordered landmark indices along the face-oval cycle (for fills / inside test). */
function orderedFaceOvalIndices(ovalConns) {
  const adj = new Map();
  for (const e of ovalConns) {
    if (!adj.has(e.start)) adj.set(e.start, []);
    if (!adj.has(e.end)) adj.set(e.end, []);
    adj.get(e.start).push(e.end);
    adj.get(e.end).push(e.start);
  }
  const start = Math.min(...adj.keys());
  const out = [start];
  let prev = null;
  let cur = start;
  for (let guard = 0; guard < 128; guard++) {
    const nbrs = adj.get(cur).filter((n) => n !== prev);
    if (!nbrs.length) break;
    const nxt = nbrs[0];
    if (nxt === start && out.length > 1) break;
    out.push(nxt);
    prev = cur;
    cur = nxt;
  }
  return out;
}

/** Vertex indices used by a connection list (sorted, unique). */
function idxFromConnections(conns) {
  const s = new Set();
  for (const e of conns) {
    s.add(e.start);
    s.add(e.end);
  }
  return [...s].sort((a, b) => a - b);
}

/** Horizontal extent of an eye contour in segmenter mask x (0…mw). */
function eyeXBoundsMaskSpace(lm, eyeConns, mw) {
  const ix = idxFromConnections(eyeConns);
  let minX = Infinity;
  let maxX = -Infinity;
  for (const i of ix) {
    const p = lm[i];
    if (!p) continue;
    const x = p.x * mw;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
  }
  if (!(minX <= maxX)) return null;
  return { minX, maxX };
}

function cross2(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

/** Convex hull (monotone chain); matches Python ConvexHull outline for landmark clouds. */
function convexHull2D(points) {
  if (points.length < 3) return points.slice();
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross2(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross2(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function landmarkPointsPx(lm, indices, cw, ch) {
  const pts = [];
  const seen = new Set();
  for (const i of indices) {
    const p = lm[i];
    if (!p) continue;
    const x = p.x * cw,
      y = p.y * ch;
    const k = x.toFixed(5) + "," + y.toFixed(5);
    if (seen.has(k)) continue;
    seen.add(k);
    pts.push([x, y]);
  }
  return pts;
}

/** Convex hull ring in image pixel space (closed). */
function convexHullLandmarksImagePx(lm, indices, iw, ih) {
  const pts = landmarkPointsPx(lm, indices, iw, ih);
  const hull = convexHull2D(pts);
  return hull.length >= 3 ? hull : [];
}

/** Closed loop along face-oval landmark graph in image pixel space (not a hull). */
function faceOvalRingImagePx(lm, ovalConns, iw, ih) {
  const idx = orderedFaceOvalIndices(ovalConns);
  const ring = [];
  for (const i of idx) {
    const p = lm[i];
    if (!p) continue;
    ring.push([p.x * iw, p.y * ih]);
  }
  return ring.length >= 3 ? ring : [];
}

function maskRingToImagePixels(ring, mw, mh, iw, ih) {
  if (!ring.length) return [];
  const sx = iw / mw;
  const sy = ih / mh;
  return ring.map(([x, y]) => [x * sx, y * sy]);
}

/**
 * Intrinsic pixel size of an image, video frame, or canvas (for layout + MediaPipe).
 * @param {HTMLImageElement | HTMLVideoElement | HTMLCanvasElement} el
 */
function getIntrinsicDimensions(el) {
  if (el instanceof HTMLImageElement) {
    return { width: el.naturalWidth, height: el.naturalHeight };
  }
  if (el instanceof HTMLVideoElement) {
    return { width: el.videoWidth, height: el.videoHeight };
  }
  if (el instanceof HTMLCanvasElement) {
    return { width: el.width, height: el.height };
  }
  return { width: 0, height: 0 };
}

/**
 * @param {HTMLImageElement | HTMLVideoElement | HTMLCanvasElement} el
 */
function isFaceSourceReady(el) {
  const { width, height } = getIntrinsicDimensions(el);
  if (!width || !height) return false;
  if (el instanceof HTMLImageElement && !el.complete) return false;
  return true;
}

/** @param {HTMLImageElement | HTMLVideoElement | HTMLCanvasElement} sourceEl */
export function syncOverlaySize(sourceEl, overlay, stageOutlinesEl) {
  const { width: nw, height: nh } = getIntrinsicDimensions(sourceEl);
  if (!nw || !nh) return;
  overlay.width = nw;
  overlay.height = nh;
  if (stageOutlinesEl) stageOutlinesEl.style.aspectRatio = `${nw} / ${nh}`;
}

/** 4-connected components on 0/1 mask. */
function connectedComponents(bin, mw, mh) {
  const labels = new Int32Array(mw * mh);
  const areas = new Map();
  const stack = [];
  let nextLabel = 1;
  for (let i = 0; i < bin.length; i++) {
    if (bin[i] !== 1 || labels[i]) continue;
    const id = nextLabel++;
    let area = 0;
    stack.push(i);
    labels[i] = id;
    while (stack.length) {
      const cur = stack.pop();
      area++;
      const x = cur % mw;
      const y = (cur / mw) | 0;
      if (x > 0) {
        const nb = cur - 1;
        if (bin[nb] === 1 && !labels[nb]) {
          labels[nb] = id;
          stack.push(nb);
        }
      }
      if (x + 1 < mw) {
        const nb = cur + 1;
        if (bin[nb] === 1 && !labels[nb]) {
          labels[nb] = id;
          stack.push(nb);
        }
      }
      if (y > 0) {
        const nb = cur - mw;
        if (bin[nb] === 1 && !labels[nb]) {
          labels[nb] = id;
          stack.push(nb);
        }
      }
      if (y + 1 < mh) {
        const nb = cur + mw;
        if (bin[nb] === 1 && !labels[nb]) {
          labels[nb] = id;
          stack.push(nb);
        }
      }
    }
    areas.set(id, area);
  }
  return { labels, areas };
}

function maskToBinary(u8, mw, mh, category) {
  const bin = new Uint8Array(mw * mh);
  for (let i = 0; i < u8.length; i++) if (u8[i] === category) bin[i] = 1;
  return bin;
}

function keepLargestComponent(bin, mw, mh) {
  const { labels, areas } = connectedComponents(bin, mw, mh);
  if (areas.size === 0) return bin;
  let best = 0;
  let bestA = 0;
  for (const [id, a] of areas)
    if (a > bestA) {
      bestA = a;
      best = id;
    }
  const out = new Uint8Array(mw * mh);
  for (let i = 0; i < bin.length; i++) if (labels[i] === best) out[i] = 1;
  return out;
}

/** 3×3 binary erode (full kernel). */
function erodeBinary(bin, mw, mh) {
  const out = new Uint8Array(mw * mh);
  for (let y = 1; y < mh - 1; y++) {
    for (let x = 1; x < mw - 1; x++) {
      let ok = 1;
      for (let dy = -1; dy <= 1 && ok; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!bin[(y + dy) * mw + x + dx]) {
            ok = 0;
            break;
          }
        }
      }
      out[y * mw + x] = ok;
    }
  }
  return out;
}

/** 3×3 binary dilate (full kernel). */
function dilateBinary(bin, mw, mh) {
  const out = new Uint8Array(mw * mh);
  for (let y = 1; y < mh - 1; y++) {
    for (let x = 1; x < mw - 1; x++) {
      let ok = 0;
      for (let dy = -1; dy <= 1 && !ok; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (bin[(y + dy) * mw + x + dx]) {
            ok = 1;
            break;
          }
        }
      }
      out[y * mw + x] = ok;
    }
  }
  return out;
}

/** One morphological closing step (3×3 dilate then erode). */
function closeBinaryOnce(bin, mw, mh) {
  return erodeBinary(dilateBinary(bin, mw, mh), mw, mh);
}

/** Notebook: binary_opening(binary_closing(ears_all, disk(2)), disk(1)). */
function morphEarsAllLikePython(bin, mw, mh) {
  let b = bin;
  b = closeBinaryOnce(b, mw, mh);
  b = closeBinaryOnce(b, mw, mh);
  b = erodeBinary(b, mw, mh);
  b = dilateBinary(b, mw, mh);
  return b;
}

function boxBlurFloatSeparate(f, mw, mh, r) {
  const tmp = new Float32Array(mw * mh);
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      let s = 0,
        c = 0;
      for (let dx = -r; dx <= r; dx++) {
        const xx = x + dx;
        if (xx >= 0 && xx < mw) {
          s += f[y * mw + xx];
          c++;
        }
      }
      tmp[y * mw + x] = s / c;
    }
  }
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      let s = 0,
        c = 0;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy >= 0 && yy < mh) {
          s += tmp[yy * mw + x];
          c++;
        }
      }
      f[y * mw + x] = s / c;
    }
  }
}

/** Separable Gaussian blur on float32 mask (notebook `gaussian_filter`). */
function gaussianBlurFloatSeparable(f, mw, mh, sigma) {
  if (sigma <= 0) return;
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float32Array(2 * radius + 1);
  let sum = 0;
  const s2 = 2 * sigma * sigma;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / s2);
    kernel[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const tmp = new Float32Array(mw * mh);
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const xx = Math.min(mw - 1, Math.max(0, x + k));
        acc += f[y * mw + xx] * kernel[k + radius];
      }
      tmp[y * mw + x] = acc;
    }
  }
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = Math.min(mh - 1, Math.max(0, y + k));
        acc += tmp[yy * mw + x] * kernel[k + radius];
      }
      f[y * mw + x] = acc;
    }
  }
}

function morphCloseNTimes(bin, mw, mh, n) {
  let b = bin;
  for (let i = 0; i < n; i++) b = closeBinaryOnce(b, mw, mh);
  return b;
}

function morphOpenNTimes(bin, mw, mh, n) {
  let b = bin;
  for (let i = 0; i < n; i++) {
    b = erodeBinary(b, mw, mh);
    b = dilateBinary(b, mw, mh);
  }
  return b;
}

/**
 * Notebook `smooth_hair_mask`: Gaussian σ, rethreshold 0.5, binary_closing(disk(close_radius)),
 * binary_opening(disk(open_radius)). Large disks approximated by repeated 3×3 close/open.
 */
function smoothHairMask(bin, mw, mh, gaussSigma = 7.0, closeRadius = 12, openRadius = 4) {
  const f = new Float32Array(mw * mh);
  for (let i = 0; i < bin.length; i++) f[i] = bin[i];
  gaussianBlurFloatSeparable(f, mw, mh, gaussSigma);
  let m = new Uint8Array(mw * mh);
  for (let i = 0; i < bin.length; i++) m[i] = f[i] >= 0.5 ? 1 : 0;
  m = morphCloseNTimes(m, mw, mh, closeRadius);
  m = morphOpenNTimes(m, mw, mh, openRadius);
  return m;
}

/** Notebook smooth_ear_mask: Gaussian-ish blur + threshold + close + open. */
function smoothEarMask(bin, mw, mh) {
  let any = false;
  for (let i = 0; i < bin.length; i++) {
    if (bin[i]) {
      any = true;
      break;
    }
  }
  if (!any) return bin;
  const f = new Float32Array(mw * mh);
  for (let i = 0; i < bin.length; i++) f[i] = bin[i];
  boxBlurFloatSeparate(f, mw, mh, 2);
  boxBlurFloatSeparate(f, mw, mh, 2);
  let m = new Uint8Array(mw * mh);
  for (let i = 0; i < bin.length; i++) m[i] = f[i] >= 0.5 ? 1 : 0;
  m = closeBinaryOnce(m, mw, mh);
  m = closeBinaryOnce(m, mw, mh);
  m = closeBinaryOnce(m, mw, mh);
  m = erodeBinary(m, mw, mh);
  m = dilateBinary(m, mw, mh);
  return m;
}

/**
 * Filled oval inside **segmenter mask space** (mw×mh). Matches skimage polygon fill;
 * avoids Canvas isPointInPath vs grid mismatch that traced the jaw along the face oval.
 */
function rasterizeOvalInsideMask(lm, ovalConns, mw, mh) {
  const idx = orderedFaceOvalIndices(ovalConns);
  const c = document.createElement("canvas");
  c.width = mw;
  c.height = mh;
  const g = c.getContext("2d");
  g.beginPath();
  idx.forEach((i, k) => {
    const p = lm[i];
    if (!p) return;
    const x = p.x * mw;
    const y = p.y * mh;
    if (k === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  });
  g.closePath();
  g.fillStyle = "#fff";
  g.fill();
  const id = g.getImageData(0, 0, mw, mh).data;
  const inside = new Uint8Array(mw * mh);
  for (let i = 0; i < mw * mh; i++) {
    if (id[i * 4 + 3] > 0) inside[i] = 1;
  }
  return inside;
}

/**
 * Moore–neighbor outer contour (ordered, single closed chain). Avoids angular-sort scribbles.
 */
function mooreOuterContour(bin, mw, mh) {
  const get = (x, y) => x >= 0 && x < mw && y >= 0 && y < mh && bin[y * mw + x];
  let sx = -1,
    sy = -1;
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      if (get(x, y) && !get(x - 1, y)) {
        sx = x;
        sy = y;
        break;
      }
    }
    if (sx >= 0) break;
  }
  if (sx < 0) return [];
  const nbrs = [
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
  ];
  const contour = [];
  let x = sx,
    y = sy;
  let bx = sx - 1,
    by = sy;
  const cap = mw * mh * 8;
  for (let iter = 0; iter < cap; iter++) {
    contour.push([x + 0.5, y + 0.5]);
    let backDir = nbrs.findIndex(([dx, dy]) => x + dx === bx && y + dy === by);
    if (backDir < 0) backDir = 0;
    let found = false;
    for (let step = 1; step <= 8; step++) {
      const [dx, dy] = nbrs[(backDir + step) % 8];
      const nx = x + dx,
        ny2 = y + dy;
      if (get(nx, ny2)) {
        bx = x;
        by = y;
        x = nx;
        y = ny2;
        found = true;
        break;
      }
    }
    if (!found) break;
    if (x === sx && y === sy && contour.length > 2) break;
  }
  return dedupeConsecutive(contour);
}

function dedupeConsecutive(pts) {
  if (pts.length < 2) return pts;
  const out = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const a = out[out.length - 1],
      b = pts[i];
    if (Math.abs(a[0] - b[0]) > 1e-6 || Math.abs(a[1] - b[1]) > 1e-6) out.push(b);
  }
  return out;
}

function perpDist(p, a, b) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  if (dx === 0 && dy === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy);
  const px = a[0] + t * dx,
    py = a[1] + t * dy;
  return Math.hypot(p[0] - px, p[1] - py);
}

/** Ramer–Douglas–Peucker (open polyline). */
function rdpOpen(pts, eps) {
  if (pts.length < 3) return pts.slice();
  let idx = 0,
    dmax = 0;
  const a = pts[0],
    b = pts[pts.length - 1];
  for (let i = 1; i < pts.length - 1; i++) {
    const d = perpDist(pts[i], a, b);
    if (d > dmax) {
      dmax = d;
      idx = i;
    }
  }
  if (dmax > eps) {
    const L = rdpOpen(pts.slice(0, idx + 1), eps);
    const R = rdpOpen(pts.slice(idx), eps);
    return L.slice(0, -1).concat(R);
  }
  return [a, b];
}

function simplifyClosedContour(pts, eps) {
  if (pts.length < 4) return pts;
  const open = pts.concat([pts[0]]);
  let s = rdpOpen(open, eps);
  if (s.length > 1 && s[s.length - 1][0] === s[0][0] && s[s.length - 1][1] === s[0][1]) s = s.slice(0, -1);
  return s;
}

function contourFromMask(bin, mw, mh, rdpEps) {
  const chain = mooreOuterContour(bin, mw, mh);
  if (chain.length < 3) return [];
  return simplifyClosedContour(chain, rdpEps);
}

/**
 * Ears = face_skin & ~inside_oval on mask grid; morph; split by eye geometry:
 * - If anatomical left eye is on the image-left (mirrored webcam): left ear = mask x left of left eye;
 *   right ear = mask x right of right eye.
 * - If anatomical left eye is on the image-right (standard photo): left ear = x past temporal side of
 *   left eye; right ear = x past nasal side of right eye.
 * Fallback: nose-tip x split if eye bounds are missing.
 * @returns {{ left: number[][], right: number[][] }} rings in mask grid coordinates
 */
function earOutlineRingsMaskSpace(u8, mw, mh, lm, ovalConns, FL) {
  const inside = rasterizeOvalInsideMask(lm, ovalConns, mw, mh);
  const faceSkin = maskToBinary(u8, mw, mh, 3);
  let earsAll = new Uint8Array(mw * mh);
  for (let i = 0; i < earsAll.length; i++) {
    if (faceSkin[i] && !inside[i]) earsAll[i] = 1;
  }
  earsAll = morphEarsAllLikePython(earsAll, mw, mh);

  const leftEye = eyeXBoundsMaskSpace(lm, FL.FACE_LANDMARKS_LEFT_EYE, mw);
  const rightEye = eyeXBoundsMaskSpace(lm, FL.FACE_LANDMARKS_RIGHT_EYE, mw);
  const noseX = lm[1]?.x != null ? lm[1].x * mw : mw * 0.5;

  const earL = new Uint8Array(mw * mh);
  const earR = new Uint8Array(mw * mh);
  for (let i = 0; i < earsAll.length; i++) {
    if (!earsAll[i]) continue;
    const mx = i % mw;
    let goL;
    let goR;
    if (leftEye && rightEye) {
      const leftCx = (leftEye.minX + leftEye.maxX) * 0.5;
      const rightCx = (rightEye.minX + rightEye.maxX) * 0.5;
      const leftEyeOnImageLeft = leftCx < rightCx;
      if (leftEyeOnImageLeft) {
        goL = mx < leftEye.minX;
        goR = mx > rightEye.maxX;
      } else {
        goL = mx > leftEye.maxX;
        goR = mx < rightEye.minX;
      }
    } else {
      goL = mx < noseX;
      goR = mx >= noseX;
    }
    if (goL) earL[i] = 1;
    else if (goR) earR[i] = 1;
  }

  let eL = keepLargestComponent(earL, mw, mh);
  let eR = keepLargestComponent(earR, mw, mh);
  eL = keepLargestComponent(smoothEarMask(eL, mw, mh), mw, mh);
  eR = keepLargestComponent(smoothEarMask(eR, mw, mh), mw, mh);

  return {
    left: contourFromMask(eL, mw, mh, 0.75),
    right: contourFromMask(eR, mw, mh, 0.75),
  };
}

/** Hair contour in mask grid coordinates (Moore + RDP). */
function hairOutlineRingMaskSpace(u8, mw, mh) {
  let hair = maskToBinary(u8, mw, mh, 1);
  hair = keepLargestComponent(hair, mw, mh);
  hair = smoothHairMask(hair, mw, mh, 7.0, 12, 4);
  hair = keepLargestComponent(hair, mw, mh);
  return contourFromMask(hair, mw, mh, 1.15);
}

/**
 * Axis-aligned ROI in segmenter mask pixels around the primary face (hair above, ears sideways).
 * @returns {{ x0: number, x1: number, y0: number, y1: number } | null}
 */
function primaryFaceRoiMaskBounds(lm, ovalConns, mw, mh) {
  const idx = orderedFaceOvalIndices(ovalConns);
  let minX = mw,
    maxX = 0,
    minY = mh,
    maxY = 0;
  for (const i of idx) {
    const p = lm[i];
    if (!p) continue;
    const x = p.x * mw;
    const y = p.y * mh;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  if (!(maxX > minX && maxY > minY)) return null;
  const w = maxX - minX;
  const h = maxY - minY;
  const padX = Math.max(8, w * 0.45);
  const padTop = Math.max(8, h * 0.95);
  const padBot = Math.max(8, h * 0.35);
  return {
    x0: Math.max(0, Math.floor(minX - padX)),
    x1: Math.min(mw - 1, Math.ceil(maxX + padX)),
    y0: Math.max(0, Math.floor(minY - padTop)),
    y1: Math.min(mh - 1, Math.ceil(maxY + padBot)),
  };
}

/** Zero multiclass labels outside ROI so hair/ears follow the same face as FaceLandmarker (category 0 = background). */
function restrictSegmentationMaskToRoi(u8, mw, mh, bounds) {
  if (!bounds) return u8;
  const out = new Uint8Array(u8.length);
  out.set(u8);
  const { x0, x1, y0, y1 } = bounds;
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      if (x < x0 || x > x1 || y < y0 || y > y1) out[y * mw + x] = 0;
    }
  }
  return out;
}

/** @typedef {{ id: string, points: number[][], closed: boolean }} FaceFeature */

/**
 * Loads MediaPipe models on first call (then caches), runs face + segmenter, returns feature outlines
 * in **source pixel space** (image natural size, video frame, or canvas bitmap).
 * Single-face policy: FaceLandmarker uses `numFaces: 1`; segmentation labels outside that face’s ROI are cleared
 * so hair/ears match the same subject (ImageSegmenter has no built-in face count).
 *
 * @param {HTMLImageElement | HTMLVideoElement | HTMLCanvasElement} sourceEl
 * @returns {Promise<{ ok: true, features: FaceFeature[], imageWidth: number, imageHeight: number } | { ok: false, message: string }>}
 */
export async function extractFaceFeaturesFromImage(sourceEl) {
  if (!isFaceSourceReady(sourceEl)) return { ok: false, message: "Image not ready." };

  let faceLandmarker, imageSegmenter, FaceLandmarkerClass;
  try {
    ({ faceLandmarker, imageSegmenter, FaceLandmarker: FaceLandmarkerClass } = await ensureMediaPipeModels());
  } catch (e) {
    const msg = e && typeof e === "object" && "message" in e ? String(e.message) : String(e);
    return { ok: false, message: "Failed to load models: " + msg };
  }

  const { width: iw, height: ih } = getIntrinsicDimensions(sourceEl);

  const faceRes = faceLandmarker.detect(sourceEl);
  const segRes = imageSegmenter.segment(sourceEl);

  const landmarks = faceRes.faceLandmarks ?? [];
  if (landmarks.length === 0) {
    return { ok: false, message: "No face detected." };
  }

  const lm = landmarks[0];
  const FL = FaceLandmarkerClass;
  /** @type {FaceFeature[]} */
  const features = [];

  let segU8 = null,
    segMw = 0,
    segMh = 0;
  if (segRes.categoryMask) {
    const cm = segRes.categoryMask;
    segMw = cm.width;
    segMh = cm.height;
    segU8 = new Uint8Array(cm.getAsUint8Array());
    const roi = primaryFaceRoiMaskBounds(lm, FL.FACE_LANDMARKS_FACE_OVAL, segMw, segMh);
    segU8 = restrictSegmentationMaskToRoi(segU8, segMw, segMh, roi);

    const hairRing = hairOutlineRingMaskSpace(segU8, segMw, segMh);
    const hairPx = maskRingToImagePixels(hairRing, segMw, segMh, iw, ih);
    if (hairPx.length >= 3) {
      features.push({
        id: "hair",
        points: hairPx,
        closed: true,
      });
    }
  }

  const oval = faceOvalRingImagePx(lm, FL.FACE_LANDMARKS_FACE_OVAL, iw, ih);
  /** @type {number[][] | null} */
  let leftEarPx = null;
  /** @type {number[][] | null} */
  let rightEarPx = null;
  if (segU8) {
    const { left: ringL, right: ringR } = earOutlineRingsMaskSpace(
      segU8,
      segMw,
      segMh,
      lm,
      FL.FACE_LANDMARKS_FACE_OVAL,
      FL,
    );
    const leftPx = maskRingToImagePixels(ringL, segMw, segMh, iw, ih);
    const rightPx = maskRingToImagePixels(ringR, segMw, segMh, iw, ih);
    if (leftPx.length >= 3) leftEarPx = leftPx;
    if (rightPx.length >= 3) rightEarPx = rightPx;
  }

  const pushHull = (id, indices) => {
    const ring = convexHullLandmarksImagePx(lm, indices, iw, ih);
    if (ring.length >= 3) features.push({ id, points: ring, closed: true });
  };

  pushHull("lips", idxFromConnections(FL.FACE_LANDMARKS_LIPS));
  pushHull("nose", idxFromConnections(FACE_LANDMARKS_NOSE));
  pushHull("leftEye", idxFromConnections(FL.FACE_LANDMARKS_LEFT_EYE));
  pushHull("leftEyebrow", idxFromConnections(FL.FACE_LANDMARKS_LEFT_EYEBROW));
  pushHull("rightEyebrow", idxFromConnections(FL.FACE_LANDMARKS_RIGHT_EYEBROW));
  pushHull("rightEye", idxFromConnections(FL.FACE_LANDMARKS_RIGHT_EYE));

  const hasHair = features.some((f) => f.id === "hair" && f.points.length >= 3);

  if (oval.length >= 3) {
    let shellPts = oval.map((p) => [p[0], p[1]]);
    let shellClosed = true;
    if (hasHair) {
      const { bottom } = splitFaceShellIntoTopBottom(oval, leftEarPx, rightEarPx);
      if (bottom.length >= 2) {
        shellPts = orientBottomArcRightEarToLeftEar(bottom, rightEarPx, leftEarPx);
        shellClosed = false;
      }
    }
    features.push({
      id: "faceShell",
      points: shellPts,
      closed: shellClosed,
    });
  }

  if (hasHair && rightEarPx && rightEarPx.length >= 3) {
    features.push({
      id: "earRight",
      points: rightEarPx.map((p) => [p[0], p[1]]),
      closed: true,
    });
  }

  if (hasHair && leftEarPx && leftEarPx.length >= 3) {
    features.push({
      id: "earLeft",
      points: leftEarPx.map((p) => [p[0], p[1]]),
      closed: true,
    });
  }

  return { ok: true, features, imageWidth: iw, imageHeight: ih };
}

/** No hair: full landmark face oval only (no ears in stroke). */
const ONE_LINE_ORDER_NO_HAIR = ["lips", "nose", "leftEye", "leftEyebrow", "rightEyebrow", "rightEye", "faceShell"];

/** With hair: lips/nose → left → hair → right side → right ear → jaw arc → left ear. */
const ONE_LINE_ORDER_WITH_HAIR = [
  "lips",
  "nose",
  "leftEye",
  "leftEyebrow",
  "hair",
  "rightEyebrow",
  "rightEye",
  "earRight",
  "faceShell",
  "earLeft",
];

function oneLineOrderForFeatures(features) {
  const hasHair = features.some((f) => f.id === "hair" && f.points.length >= 2);
  return hasHair ? ONE_LINE_ORDER_WITH_HAIR : ONE_LINE_ORDER_NO_HAIR;
}

function dist2Pt(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

function findClosestPointIndex(points, p) {
  let bestI = 0;
  let best = Infinity;
  for (let i = 0; i < points.length; i++) {
    const d = dist2Pt(points[i], p);
    if (d < best) {
      best = d;
      bestI = i;
    }
  }
  return bestI;
}

function polyCentroid2D(pts) {
  if (!pts.length) return [0, 0];
  let sx = 0,
    sy = 0;
  for (const p of pts) {
    sx += p[0];
    sy += p[1];
  }
  return [sx / pts.length, sy / pts.length];
}

function walkRingForward(pts, iStart, iEnd) {
  const n = pts.length;
  const out = [[pts[iStart][0], pts[iStart][1]]];
  if (iStart === iEnd) return out;
  let i = iStart;
  for (let k = 0; k < n; k++) {
    i = (i + 1) % n;
    out.push([pts[i][0], pts[i][1]]);
    if (i === iEnd) break;
  }
  return out;
}

function meanYArc(arc) {
  if (!arc.length) return 0;
  let s = 0;
  for (const p of arc) s += p[1];
  return s / arc.length;
}

/** Split closed shell into upper (smaller mean y) and lower (jaw/cheeks) arcs. */
function splitFaceShellIntoTopBottom(shellRing, leftEarPx, rightEarPx) {
  if (shellRing.length < 4) return { top: [], bottom: [] };
  const pts = shellRing.map((p) => [p[0], p[1]]);
  const n = pts.length;

  let iL, iR;
  if (leftEarPx?.length >= 2 && rightEarPx?.length >= 2) {
    iL = findClosestPointIndex(pts, polyCentroid2D(leftEarPx));
    iR = findClosestPointIndex(pts, polyCentroid2D(rightEarPx));
  } else {
    let minX = Infinity,
      maxX = -Infinity;
    let iMin = 0,
      iMax = 0;
    for (let i = 0; i < n; i++) {
      const x = pts[i][0];
      if (x < minX) {
        minX = x;
        iMin = i;
      }
      if (x > maxX) {
        maxX = x;
        iMax = i;
      }
    }
    iL = iMin;
    iR = iMax;
  }

  const arcLR = walkRingForward(pts, iL, iR);
  const arcRL = walkRingForward(pts, iR, iL);
  if (meanYArc(arcLR) <= meanYArc(arcRL)) {
    return { top: arcLR, bottom: arcRL };
  }
  return { top: arcRL, bottom: arcLR };
}

/**
 * Order bottom jaw arc so traversal runs from the oval point near the right ear to the point near
 * the left ear (short bridges: right ear → first point, last point → left ear).
 */
function orientBottomArcRightEarToLeftEar(arc, rightEarPx, leftEarPx) {
  if (arc.length < 2) return arc.map((p) => [p[0], p[1]]);
  const copy = arc.map((p) => [p[0], p[1]]);
  const a = copy[0];
  const b = copy[copy.length - 1];
  const reverse =
    rightEarPx?.length >= 2 && leftEarPx?.length >= 2
      ? dist2Pt(a, polyCentroid2D(rightEarPx)) > dist2Pt(a, polyCentroid2D(leftEarPx))
      : a[0] < b[0];
  if (reverse) copy.reverse();
  return copy;
}

/** Closed polygon as one open traversal starting at startIdx (no repeated closing vertex). */
function rotateClosedRing(points, startIdx) {
  const n = points.length;
  if (n < 2) return points.map((p) => p.slice());
  const out = [];
  for (let k = 0; k < n; k++) {
    const p = points[(startIdx + k) % n];
    out.push([p[0], p[1]]);
  }
  return out;
}

/**
 * Merge feature rings into one polyline: each closed loop is rotated to meet the previous endpoint,
 * so the pen never lifts (straight “bridges” between regions when needed).
 * @param {Array<{ id: string, points: number[][], closed: boolean }>} features
 * @param {string[]} [orderedIds]
 * @returns {number[][]}
 */
function chainFeaturesToOnePath(features, orderedIds = oneLineOrderForFeatures(features)) {
  const byId = new Map(features.map((f) => [f.id, f]));
  const faceShellFeat = byId.get("faceShell");
  let jawRightPt = null;
  if (faceShellFeat && faceShellFeat.closed === false && faceShellFeat.points.length >= 2) {
    const jp = faceShellFeat.points;
    jawRightPt = [jp[0][0], jp[0][1]];
  }
  /** @type {number[][]} */
  const out = [];
  const eps2 = 0.15 * 0.15;

  const pushPt = (p) => {
    if (out.length && dist2Pt(out[out.length - 1], p) <= eps2) return;
    out.push([p[0], p[1]]);
  };

  for (const id of orderedIds) {
    const f = byId.get(id);
    if (!f || f.points.length < 2) continue;

    let pts = f.points.map((p) => [p[0], p[1]]);

    if (f.closed && pts.length >= 3) {
      const anchor = out.length ? out[out.length - 1] : pts[0];
      if (id === "earRight" && jawRightPt) {
        let bestS = 0;
        let bestOutD = Infinity;
        let bestInD = Infinity;
        const n = pts.length;
        for (let s = 0; s < n; s++) {
          const rot = rotateClosedRing(pts, s);
          const dOut = dist2Pt(rot[rot.length - 1], jawRightPt);
          const dIn = dist2Pt(rot[0], anchor);
          if (dOut < bestOutD - 1e-12 || (Math.abs(dOut - bestOutD) <= 1e-12 && dIn < bestInD)) {
            bestOutD = dOut;
            bestInD = dIn;
            bestS = s;
          }
        }
        pts = rotateClosedRing(pts, bestS);
      } else {
        const i0 = findClosestPointIndex(pts, anchor);
        pts = rotateClosedRing(pts, i0);
      }
    }

    for (const p of pts) pushPt(p);
  }

  return out;
}

/**
 * Exported for rendering/animation (e.g. Fourier progressive reconstruction) without forcing
 * a static canvas stroke.
 * @param {FaceFeature[]} features
 * @param {string[]} [order]
 * @returns {number[][]}
 */
export function buildOneLinePath(features, order) {
  return chainFeaturesToOnePath(features, order ?? oneLineOrderForFeatures(features));
}

function strokePathSmooth(ctx, path) {
  if (path.length < 2) return;
  if (path.length === 2) {
    ctx.moveTo(path[0][0], path[0][1]);
    ctx.lineTo(path[1][0], path[1][1]);
    return;
  }
  ctx.moveTo(path[0][0], path[0][1]);
  for (let i = 1; i < path.length - 1; i++) {
    const xc = (path[i][0] + path[i + 1][0]) * 0.5;
    const yc = (path[i][1] + path[i + 1][1]) * 0.5;
    ctx.quadraticCurveTo(path[i][0], path[i][1], xc, yc);
  }
  const last = path[path.length - 1];
  ctx.lineTo(last[0], last[1]);
}

/** Axis-aligned bbox of polyline vertices (path is [x,y][]). */
function bboxFromPathPoints(path) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of path) {
    minX = Math.min(minX, p[0]);
    maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]);
    maxY = Math.max(maxY, p[1]);
  }
  return { minX, minY, maxX, maxY };
}

/**
 * Single continuous “one line” portrait (white background, one stroke).
 * By default fits the stroke’s bounding box into the canvas and centers it (uniform scale).
 * @param {HTMLCanvasElement} canvas
 * @param {FaceFeature[]} features
 * @param {{ lineWidth?: number, strokeStyle?: string, order?: string[], smooth?: boolean, fitAndCenter?: boolean, fitMargin?: number }} [options]
 */
export function drawOneLineFaceToCanvas(canvas, features, options = {}) {
  const lineWidth = options.lineWidth ?? 2.25;
  const strokeStyle = options.strokeStyle ?? "#141414";
  const order = options.order ?? oneLineOrderForFeatures(features);
  const smooth = options.smooth !== false;
  const fitAndCenter = options.fitAndCenter !== false;
  const fitMargin = options.fitMargin ?? 0.92;

  const path = chainFeaturesToOnePath(features, order);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, H);
  if (path.length < 2) return;

  ctx.save();
  if (fitAndCenter) {
    const b = bboxFromPathPoints(path);
    let { minX, minY, maxX, maxY } = b;
    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const inflate = Math.max(lineWidth * 3, Math.max(spanX, spanY, 1) * 0.04, Math.min(W, H) * 0.01);
    minX -= inflate;
    maxX += inflate;
    minY -= inflate;
    maxY += inflate;
    const bw = Math.max(maxX - minX, 1e-6);
    const bh = Math.max(maxY - minY, 1e-6);
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const scale = Math.min(W / bw, H / bh) * fitMargin;
    ctx.translate(W * 0.5, H * 0.5);
    ctx.scale(scale, scale);
    ctx.translate(-cx, -cy);
    ctx.lineWidth = lineWidth / scale;
  } else {
    ctx.lineWidth = lineWidth;
  }

  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = strokeStyle;
  ctx.beginPath();
  if (smooth) strokePathSmooth(ctx, path);
  else {
    ctx.moveTo(path[0][0], path[0][1]);
    for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
  }
  ctx.stroke();
  ctx.restore();
}
