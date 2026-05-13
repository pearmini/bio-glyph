import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  drawOneLinePathToCanvas,
  extractFaceFeaturesFromImage,
  syncOverlaySize,
} from "./facePipeline.js";
import "./App.css";
import { Play, X } from "lucide-react";
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

/** GitHub logo (Lucide does not ship a GitHub icon). */
function GitHubMark() {
  return (
    <svg
      className="app-github-icon"
      width={22}
      height={22}
      viewBox="0 0 24 24"
      aria-hidden
    >
      <path
        fill="currentColor"
        d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"
      />
    </svg>
  );
}

/** @param {number[][]} path */
function ArchiveFaceThumb({ path }) {
  const { viewBox, d, strokeWidth } = pathToBubbleSvg(path, 128);
  return (
    <svg
      className="archive-grid__thumb-svg"
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

/** @typedef {"idle" | "preview" | "generating" | "result"} AppPhase */

export default function App() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayRef = useRef(null);
  const resultCanvasRef = useRef(null);
  const streamRef = useRef(null);
  /** Bumped when navigating home via BioGlyph so in-flight `generate` cannot commit. */
  const generateEpochRef = useRef(0);
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
  const [archiveOpen, setArchiveOpen] = useState(false);
  const idleDemoCanvasRef = useRef(null);

  const archiveGridItems = useMemo(
    () => savedGenerations.filter((g) => g.path && g.path.length >= 2),
    [savedGenerations],
  );

  const idleDemoPath = useMemo(() => {
    const last = archiveGridItems[archiveGridItems.length - 1];
    return last?.path ?? null;
  }, [archiveGridItems]);

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
    const generateRunId = generateEpochRef.current;

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

    if (generateRunId !== generateEpochRef.current) return;

    const path = await runOnSource(canvas);
    if (generateRunId !== generateEpochRef.current) {
      setResultPath(null);
      setExtractError(null);
      clearOverlay();
      return;
    }
    if (path) {
      setGeneratingFrameUrl(null);
      setResultAnimPlaying(true);
      setPhase("result");
    } else {
      setGeneratingFrameUrl(null);
      if (generateRunId !== generateEpochRef.current) return;
      void startCamera();
    }
  }, [runOnSource, stopStream, startCamera, clearOverlay]);

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

  useEffect(() => {
    if (phase !== "idle" || !idleDemoPath) return;
    const canvas = idleDemoCanvasRef.current;
    if (!canvas) return;

    return startFourierOneLineAnimation(canvas, idleDemoPath, {
      samples: 2048,
      epicycles: RESULT_EPICYCLES,
      outSamples: 1500,
      fadeAlpha: 0.04,
      strokeStyle: "#141414",
      lineWidth: RESULT_LINE_CSS_PX,
      coeffsPerSecond: 65,
      loop: true,
      autoSeam: true,
      seamGapFraction: 0.02,
    });
  }, [phase, idleDemoPath]);

  const goToStart = useCallback(() => {
    setExtractError(null);
    if (phase === "idle") {
      setCameraError(null);
      setArchiveOpen(false);
      return;
    }
    generateEpochRef.current += 1;
    stopStream();
    clearOverlay();
    setGeneratingFrameUrl(null);
    setResultPath(null);
    setResultReplayKey(0);
    setResultFixedM(null);
    setResultDisplayM(RESULT_EPICYCLES);
    setCameraError(null);
    setArchiveOpen(false);
    setPhase("idle");
  }, [phase, stopStream, clearOverlay]);

  const openArchiveGeneration = useCallback(
    (record) => {
      if (!record?.path || record.path.length < 2) return;
      generateEpochRef.current += 1;
      stopStream();
      clearOverlay();
      setGeneratingFrameUrl(null);
      setExtractError(null);
      setCameraError(null);
      setResultFixedM(null);
      setResultPath(record.path);
      setResultReplayKey((n) => n + 1);
      setResultAnimPlaying(true);
      setArchiveOpen(false);
      setPhase("result");
    },
    [stopStream, clearOverlay],
  );

  useEffect(() => {
    if (!archiveOpen) return;
    const onKey = (e) => {
      if (e.key === "Escape") setArchiveOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [archiveOpen]);

  useEffect(() => {
    if (!archiveOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [archiveOpen]);

  return (
    <div className="app-root">
      <div className="app-top-bar">
        <button type="button" className="app-brand app-brand--button" onClick={goToStart}>
          BioGlyph
        </button>
        <div className="app-top-bar__end">
          <button
            type="button"
            className="app-archive-btn"
            onClick={() => setArchiveOpen(true)}
            aria-haspopup="dialog"
            aria-expanded={archiveOpen}
          >
            Archive
          </button>
          <a
            href="https://github.com/pearmini/bio-glyph"
            className="app-github-link"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Bio Glyph on GitHub"
          >
            <GitHubMark />
          </a>
        </div>
      </div>
      {archiveOpen ? (
        <div className="archive-modal-layer" role="presentation">
          <button
            type="button"
            className="archive-modal-backdrop"
            aria-label="Close archive"
            onClick={() => setArchiveOpen(false)}
          />
          <div
            className="archive-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="archive-modal-title"
            aria-describedby="archive-modal-count"
          >
            <header className="archive-modal__header">
              <div className="archive-modal__titles">
                <h2 id="archive-modal-title" className="archive-modal__title">
                  Participants in ITP Spring Show 2026
                </h2>
                <p id="archive-modal-count" className="archive-modal__subtitle">
                  {archiveGridItems.length}{" "}
                  {archiveGridItems.length === 1 ? "face" : "faces"}
                </p>
              </div>
              <button
                type="button"
                className="archive-modal__close"
                onClick={() => setArchiveOpen(false)}
                aria-label="Close"
              >
                <X size={22} strokeWidth={2} aria-hidden />
              </button>
            </header>
            <div className="archive-modal__body">
              {archiveGridItems.length === 0 ? (
                <p className="archive-modal__empty">No saved faces in the archive yet.</p>
              ) : (
                <div className="archive-grid">
                  {archiveGridItems.map((g) => (
                    <button
                      key={g.id}
                      type="button"
                      className="archive-grid__cell"
                      onClick={() => openArchiveGeneration(g)}
                      aria-label={`Open archive entry ${g.id}`}
                    >
                      <ArchiveFaceThumb path={g.path} />
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}
      <main className={`stage stage--${phase}`}>
        {phase === "idle" && (
          <div className="stage__column">
            {idleDemoPath ? (
              <div className="circle-viewport circle-viewport--result circle-viewport--idle-demo">
                <canvas
                  ref={idleDemoCanvasRef}
                  className="circle-viewport__result-canvas"
                  aria-label="Example one-line face from the archive"
                />
              </div>
            ) : null}
            <p className="stage__tagline">
              Draw faces in one continuous line, your own bio signature.
            </p>
            <button
              type="button"
              className="btn btn--dark"
              onClick={() => {
                setExtractError(null);
                void startCamera();
              }}
            >
              Start
            </button>
          </div>
        )}

        {phase === "preview" && (
          <div className="preview-stage">
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
              Generating...
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
