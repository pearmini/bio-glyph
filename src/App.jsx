import { useCallback, useEffect, useRef, useState } from "react";
import {
  drawOneLinePathToCanvas,
  extractFaceFeaturesFromImage,
  syncOverlaySize,
} from "./facePipeline.js";
import "./App.css";
import { Play } from "lucide-react";
import { getFourierReconstructionContours, startFourierOneLineAnimation } from "./fourierOneLineAnimation.js";
import { loadGenerations, pathSegmentsToBubbleSvg, pathToBubbleSvg } from "./generationStorage.js";

const VIDEO_CONSTRAINTS = {
  video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
  audio: false,
};

/** Max Fourier terms used in the result animation (must match `epicycles` passed to the animator). */
const RESULT_EPICYCLES = 320;

/** Extra time on the generating screen after capture before analysis runs. */
const GENERATING_HOLD_MS = 1000;

/** Raster result stroke in CSS px (`startFourierOneLineAnimation` default `lineWidth`). */
const RESULT_LINE_CSS_PX = 2.25;
/** Same as `fitToCanvasTransform(..., margin)` in `fourierOneLineAnimation.js`. */
const RESULT_PATH_FIT = 0.88;
/** Matches `.circle-viewport { width: min(80vmin, 480px) }` max — ties SVG stroke to on-screen PNG weight. */
const RESULT_VIEWPORT_MAX_CSS = 480;
const RESULT_EXPORT_SVG_SIZE = 1024;

/** Shared with result canvas animator — SVG export uses full detail (320 terms). */
const FOURIER_SVG_EXPORT = {
  samples: 2048,
  outSamples: 1500,
  autoSeam: true,
  seamGapFraction: 0.02,
};

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** @param {Blob} blob */
function triggerFileDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function hashString(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed) {
  return function next() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Layout: top and bottom bands with a flex spacer for the camera column. Bubble
 * positions are percentages within each band so large discs stay in the margins.
 */
function bubbleLayoutForId(id) {
  const rng = mulberry32(hashString(id));
  const topBand = rng() < 0.5;
  const cx = 8 + rng() * 84;
  const cy = 18 + rng() * 64;
  const sizePx = 100 + Math.floor(rng() * 92);

  let d1x = -20 - rng() * 30;
  let d1y = -14 - rng() * 24;
  let d2x = 18 + rng() * 28;
  let d2y = -12 - rng() * 22;
  if (!topBand) {
    d1x = -22 - rng() * 28;
    d1y = 12 + rng() * 26;
    d2x = 16 + rng() * 30;
    d2y = 10 + rng() * 24;
  }

  const duration = 19 + rng() * 16;
  const delay = -rng() * duration;
  return {
    topBand,
    anchor: {
      left: `${cx}%`,
      top: `${cy}%`,
    },
    disc: {
      width: sizePx,
      height: sizePx,
      "--bubble-d1x": `${d1x}px`,
      "--bubble-d1y": `${d1y}px`,
      "--bubble-d2x": `${d2x}px`,
      "--bubble-d2y": `${d2y}px`,
      animationDuration: `${duration}s`,
      animationDelay: `${delay}s`,
    },
  };
}

/**
 * Map band-local anchor % to full bubble-field % (same vertical split as the old
 * two flex rows: top band uses the upper half, bottom band the lower half).
 */
function bubbleFieldAnchorStyle(layout) {
  const rawTop = layout.anchor.top;
  const topPct = typeof rawTop === "string" ? parseFloat(rawTop) : Number(rawTop);
  const safe = Number.isFinite(topPct) ? topPct : 0;
  const top = layout.topBand ? `${safe * 0.5}%` : `${50 + safe * 0.5}%`;
  return { left: layout.anchor.left, top };
}

/** Newer generations get a higher z-index so they paint above older bubbles. */
function bubbleStackZById(generations) {
  if (!generations.length) return new Map();
  const sorted = [...generations].sort((a, b) => {
    if (b.createdAt !== a.createdAt) return b.createdAt - a.createdAt;
    return String(a.id).localeCompare(String(b.id));
  });
  const map = new Map();
  for (let i = 0; i < sorted.length; i++) {
    map.set(sorted[i].id, sorted.length - i);
  }
  return map;
}

/** Renders a saved glyph in a bubble from stored path only (no raster thumbnail). */
function HistoryBubbleFace({ path }) {
  const { viewBox, d, strokeWidth } = pathToBubbleSvg(path);
  return (
    <svg
      className="history-bubble__svg"
      viewBox={viewBox}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden
    >
      {d ? (
        <path
          d={d}
          fill="none"
          stroke="#141414"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ) : null}
    </svg>
  );
}

/** @typedef {"idle" | "preview" | "generating" | "result"} AppPhase */

export default function App() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayRef = useRef(null);
  const resultCanvasRef = useRef(null);
  const streamRef = useRef(null);
  /** Fourier result animation: false when finished, true while coeffs are animating. */
  const [resultAnimPlaying, setResultAnimPlaying] = useState(false);
  /** Increment to restart the result animation with the same path. */
  const [resultReplayKey, setResultReplayKey] = useState(0);
  /** If set, render a static Fourier frame using exactly this many coefficients (m). */
  const [resultFixedM, setResultFixedM] = useState(null);
  /** Coefficient count shown on the slider / label (synced with animation frames when playing). */
  const [resultDisplayM, setResultDisplayM] = useState(RESULT_EPICYCLES);

  const onAnimM = useCallback((m) => {
    setResultDisplayM(m);
  }, []);

  /** @type {[AppPhase, React.Dispatch<React.SetStateAction<AppPhase>>]} */
  const [phase, setPhase] = useState("idle");
  const [cameraError, setCameraError] = useState(null);
  const [extractError, setExtractError] = useState(null);
  /** @type {[number[][] | null, React.Dispatch<React.SetStateAction<number[][] | null>>]} */
  const [resultPath, setResultPath] = useState(null);
  /** Last camera frame shown under the generating mask (data URL). */
  const [generatingFrameUrl, setGeneratingFrameUrl] = useState(null);
  /** Bumps when a new MediaStream is attached so the preview effect re-runs after async getUserMedia. */
  const [previewSession, setPreviewSession] = useState(0);
  const [savedGenerations] = useState(() => loadGenerations());

  const stopStream = useCallback(() => {
    const s = streamRef.current;
    if (s) {
      s.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const v = videoRef.current;
    if (v) v.srcObject = null;
  }, []);

  const clearOverlay = useCallback(() => {
    const overlay = overlayRef.current;
    if (!overlay || overlay.width === 0) return;
    const ctx = overlay.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, overlay.width, overlay.height);
    }
  }, []);

  const runOnSource = useCallback(
    async (sourceEl) => {
      const overlay = overlayRef.current;
      if (!sourceEl || !overlay) return null;
      syncOverlaySize(sourceEl, overlay, null);
      const extracted = await extractFaceFeaturesFromImage(sourceEl);
      if (extracted.ok) {
        // Static overlay + Fourier use the same merged path (single or multi-face, left → right).
        drawOneLinePathToCanvas(overlay, extracted.mergedPath);
        setResultPath(extracted.mergedPath);
        setExtractError(null);
        return extracted.mergedPath;
      }
      clearOverlay();
      setExtractError(extracted.message);
      return null;
    },
    [clearOverlay],
  );

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
      streamRef.current = stream;
      setPreviewSession((n) => n + 1);
      setPhase("preview");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setCameraError(msg);
      setPhase("idle");
    }
  }, []);

  const retake = useCallback(async () => {
    stopStream();
    clearOverlay();
    setGeneratingFrameUrl(null);
    setResultPath(null);
    setResultReplayKey(0);
    setResultFixedM(null);
    setResultDisplayM(RESULT_EPICYCLES);
    setExtractError(null);
    setCameraError(null);
    setPhase("preview");
    try {
      const stream = await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
      streamRef.current = stream;
      setPreviewSession((n) => n + 1);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setCameraError(msg);
      setPhase("idle");
    }
  }, [stopStream, clearOverlay]);

  const generate = useCallback(async () => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, vw, vh);

    let freezeUrl = null;
    try {
      freezeUrl = canvas.toDataURL("image/jpeg", 0.88);
    } catch {
      /* ignore */
    }
    setGeneratingFrameUrl(freezeUrl);
    stopStream();
    setPhase("generating");
    setCameraError(null);

    await delay(GENERATING_HOLD_MS);

    const path = await runOnSource(canvas);
    if (path) {
      setGeneratingFrameUrl(null);
      setResultAnimPlaying(true);
      setPhase("result");
    } else {
      setGeneratingFrameUrl(null);
      void startCamera();
    }
  }, [runOnSource, stopStream, startCamera]);

  useEffect(() => {
    if (phase !== "preview") return;
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!video || !stream) return;

    video.srcObject = stream;
    void video.play();

    const sync = () => {
      syncOverlaySize(video, overlayRef.current, null);
    };

    video.addEventListener("loadedmetadata", sync);
    if (video.readyState >= 1) sync();

    return () => {
      video.removeEventListener("loadedmetadata", sync);
      if (video.srcObject === stream) video.srcObject = null;
    };
  }, [phase, previewSession]);

  useEffect(() => {
    if (phase !== "result") {
      queueMicrotask(() => setResultAnimPlaying(false));
    }
  }, [phase]);

  const replayResultAnimation = useCallback(() => {
    const mAtClick = resultFixedM ?? resultDisplayM;
    setResultDisplayM(mAtClick);
    setResultFixedM(null);
    setResultAnimPlaying(true);
    setResultReplayKey((n) => n + 1);
  }, [resultFixedM, resultDisplayM]);

  const openSavedGeneration = useCallback(
    (record) => {
      if (!record?.path || record.path.length < 2) return;
      stopStream();
      setGeneratingFrameUrl(null);
      setExtractError(null);
      setCameraError(null);
      setResultFixedM(null);
      setResultPath(record.path);
      setResultReplayKey((n) => n + 1);
      setResultAnimPlaying(true);
      setPhase("result");
    },
    [stopStream],
  );

  const downloadResultPng = useCallback(() => {
    const canvas = resultCanvasRef.current;
    if (!canvas || canvas.width < 1 || canvas.height < 1) return;
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        triggerFileDownload(blob, `bioglyph-${Date.now()}.png`);
      },
      "image/png",
    );
  }, []);

  const downloadResultSvg = useCallback(() => {
    if (!resultPath || resultPath.length < 2) return;
    const contours = getFourierReconstructionContours(resultPath, {
      ...FOURIER_SVG_EXPORT,
      epicycles: RESULT_EPICYCLES,
      m: RESULT_EPICYCLES,
    });
    const { viewBox, d } = pathSegmentsToBubbleSvg(contours, RESULT_EXPORT_SVG_SIZE);
    if (!d) return;
    const pad = 8;
    const drawable = RESULT_EXPORT_SVG_SIZE - 2 * pad;
    const strokeWidth = (RESULT_LINE_CSS_PX * drawable) / (RESULT_PATH_FIT * RESULT_VIEWPORT_MAX_CSS);
    const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" width="${RESULT_EXPORT_SVG_SIZE}" height="${RESULT_EXPORT_SVG_SIZE}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <path fill="none" stroke="#141414" stroke-width="${strokeWidth}" stroke-linecap="round" stroke-linejoin="round" d="${d}"/>
</svg>`;
    const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    triggerFileDownload(blob, `bioglyph-${Date.now()}.svg`);
  }, [resultPath]);

  useEffect(() => {
    if (phase !== "result") return;
    const canvas = resultCanvasRef.current;
    if (!canvas || !resultPath || resultPath.length < 2) return;

    return startFourierOneLineAnimation(canvas, resultPath, {
      samples: 2048,
      epicycles: RESULT_EPICYCLES,
      fixedM: resultFixedM,
      outSamples: 1500,
      fadeAlpha: 0.04,
      strokeStyle: "#141414",
      lineWidth: RESULT_LINE_CSS_PX,
      coeffsPerSecond: 65,
      loop: false,
      autoSeam: true,
      seamGapFraction: 0.02,
      onM: onAnimM,
      onComplete: () => setResultAnimPlaying(false),
    });
  }, [phase, resultPath, resultReplayKey, resultFixedM, onAnimM]);

  const goToGeneratePage = useCallback(() => {
    setExtractError(null);
    if (phase === "preview") return;
    if (phase === "idle") {
      void startCamera();
      return;
    }
    void retake();
  }, [phase, retake, startCamera]);

  return (
    <div className="app-root">
      <div className="app-top-bar">
        <button type="button" className="app-brand app-brand--button" onClick={goToGeneratePage}>
          BioGlyph
        </button>
      </div>
      <main className={`stage stage--${phase}`}>
        {phase === "idle" && (
          <button
            type="button"
            className="btn btn--dark"
            onClick={() => {
              setExtractError(null);
              void startCamera();
            }}
          >
            Start camera
          </button>
        )}

        {phase === "preview" && (
          <div className="preview-stage">
            {savedGenerations.length > 0 ? (
              <div className="bubble-field" aria-hidden>
                {(() => {
                  const stackZById = bubbleStackZById(savedGenerations);
                  const rows = savedGenerations.map((g) => ({
                    g,
                    layout: bubbleLayoutForId(g.id),
                    stackZ: stackZById.get(g.id) ?? 1,
                  }));
                  return (
                    <div className="bubble-field__anchors">
                      {rows.map(({ g, layout, stackZ }) => (
                        <button
                          key={g.id}
                          type="button"
                          className="history-bubble-anchor"
                          style={{
                            ...bubbleFieldAnchorStyle(layout),
                            zIndex: stackZ,
                          }}
                          onClick={() => openSavedGeneration(g)}
                          aria-label="Open saved generation"
                        >
                          <span className="history-bubble" style={layout.disc}>
                            <HistoryBubbleFace path={g.path} />
                          </span>
                        </button>
                      ))}
                    </div>
                  );
                })()}
              </div>
            ) : null}
            <div className="preview-stage__foreground">
              <div className="stage__column">
                <div className="circle-viewport">
                  <video
                    ref={videoRef}
                    className="circle-viewport__video"
                    playsInline
                    muted
                    autoPlay
                  />
                </div>
                <p className="stage__tip">Place your face or faces in the circle</p>
                <button type="button" className="btn btn--dark" onClick={() => void generate()}>
                  Generate
                </button>
              </div>
            </div>
          </div>
        )}

        {phase === "generating" && (
          <div className="stage__column" aria-busy="true">
            <div className="circle-viewport">
              {generatingFrameUrl ? (
                <img
                  src={generatingFrameUrl}
                  alt=""
                  className="circle-viewport__freeze"
                />
              ) : null}
            </div>
            <button type="button" className="btn btn--dark btn--busy" disabled>
              Generate
            </button>
          </div>
        )}

        {phase === "result" && resultPath && (
          <div className="stage__column stage__column--result">
            <div className="circle-viewport circle-viewport--result">
              <canvas ref={resultCanvasRef} className="circle-viewport__result-canvas" aria-label="Fourier animation" />
            </div>
            <div className="stage__result-actions">
              <div className="result-actions__row">
                <div className="result-controls">
                  <button
                    type="button"
                    className="btn btn--icon btn--icon-dark"
                    disabled={resultAnimPlaying}
                    onClick={replayResultAnimation}
                    aria-label="Replay"
                    title="Replay"
                  >
                    <Play size={18} strokeWidth={2.4} aria-hidden />
                  </button>
                  <label className="result-m" aria-label="Detail slider">
                    <input
                      type="range"
                      className="result-m__slider"
                      min={0}
                      max={RESULT_EPICYCLES}
                      step={1}
                      value={Math.max(0, Math.min(RESULT_EPICYCLES, resultDisplayM))}
                      onChange={(e) => {
                        const next = Number(e.target.value);
                        setResultAnimPlaying(false);
                        setResultFixedM(next);
                        setResultDisplayM(next);
                        setResultReplayKey((n) => n + 1);
                      }}
                    />
                  <span className="result-m__value" aria-hidden>
                    Detail: {Math.max(0, Math.min(RESULT_EPICYCLES, resultDisplayM))}
                  </span>
                  </label>
                </div>
              </div>
              <div className="result-actions__row result-actions__row--wrap">
                <button type="button" className="btn" onClick={downloadResultPng}>
                  Download PNG
                </button>
                <button type="button" className="btn" onClick={downloadResultSvg}>
                  Download SVG
                </button>
                <button type="button" className="btn" onClick={() => void retake()}>
                  Back
                </button>
              </div>
            </div>
          </div>
        )}

        {(cameraError || extractError) && (
          <p className="stage__error" role="alert">
            {cameraError || extractError}
          </p>
        )}
      </main>

      <canvas ref={captureCanvasRef} className="offscreen-canvas" aria-hidden />
      <canvas ref={overlayRef} className="offscreen-canvas" aria-hidden />
    </div>
  );
}
