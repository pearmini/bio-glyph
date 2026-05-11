import { useCallback, useEffect, useRef, useState } from "react";
import {
  drawOneLineFaceToCanvas,
  extractFaceFeaturesFromImage,
  buildOneLinePath,
  syncOverlaySize,
} from "./facePipeline.js";
import "./App.css";
import { Maximize2, Play, X } from "lucide-react";
import QRCode from "qrcode";
import { startFourierOneLineAnimation } from "./fourierOneLineAnimation.js";
import { addGeneration, loadGenerations, pathToBubbleSvg } from "./generationStorage.js";

const VIDEO_CONSTRAINTS = {
  video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
  audio: false,
};

/** Max Fourier terms used in the result animation (must match `epicycles` passed to the animator). */
const RESULT_EPICYCLES = 320;

/** Extra time on the generating screen after capture before analysis runs. */
const GENERATING_HOLD_MS = 1000;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

function getActiveFullscreenElement() {
  return document.fullscreenElement ?? document.webkitFullscreenElement ?? null;
}

const LITTERBOX_UPLOAD =
  "https://litterbox.catbox.moe/resources/internals/api.php";

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
/** @typedef {"idle" | "uploading" | "ready" | "error"} SharePhase */

export default function App() {
  const appRootRef = useRef(null);
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayRef = useRef(null);
  const resultCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
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
  const [savedGenerations, setSavedGenerations] = useState(() => loadGenerations());

  const shareAbortRef = useRef(null);
  const [shareOpen, setShareOpen] = useState(false);
  /** @type {[SharePhase, React.Dispatch<React.SetStateAction<SharePhase>>]} */
  const [sharePhase, setSharePhase] = useState("idle");
  const [shareImageUrl, setShareImageUrl] = useState(null);
  const [shareQrDataUrl, setShareQrDataUrl] = useState(null);
  const [shareError, setShareError] = useState(null);

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
        // Keep the original static draw as a fallback/offscreen render,
        // but prefer the path for Fourier animation in the result stage.
        drawOneLineFaceToCanvas(overlay, extracted.features);
        const path = buildOneLinePath(extracted.features);
        setResultPath(path);
        setExtractError(null);
        return path;
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
      try {
        addGeneration(path);
        setSavedGenerations(loadGenerations());
        setGeneratingFrameUrl(null);
        setResultAnimPlaying(true);
        setPhase("result");
      } catch {
        setGeneratingFrameUrl(null);
        void startCamera();
      }
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

  const closeShareModal = useCallback(() => {
    shareAbortRef.current?.abort();
    shareAbortRef.current = null;
    setShareOpen(false);
    setSharePhase("idle");
    setShareImageUrl(null);
    setShareQrDataUrl(null);
    setShareError(null);
  }, []);

  const openShareModal = useCallback(() => {
    const canvas = resultCanvasRef.current;
    if (!canvas || canvas.width < 1 || canvas.height < 1) return;

    shareAbortRef.current?.abort();
    const ac = new AbortController();
    shareAbortRef.current = ac;

    setShareOpen(true);
    setSharePhase("uploading");
    setShareImageUrl(null);
    setShareQrDataUrl(null);
    setShareError(null);

    try {
      canvas.toBlob(
        async (blob) => {
          if (ac.signal.aborted) return;
          if (!blob) {
            setSharePhase("error");
            setShareError("Could not read image.");
            return;
          }
          try {
            const fd = new FormData();
            fd.append("reqtype", "fileupload");
            fd.append("time", "1h");
            fd.append("fileToUpload", blob, `bioglyph-${Date.now()}.png`);
            const res = await fetch(LITTERBOX_UPLOAD, {
              method: "POST",
              body: fd,
              signal: ac.signal,
            });
            if (!res.ok) {
              throw new Error(`Upload failed (${res.status})`);
            }
            const text = (await res.text()).trim();
            if (!text.startsWith("http")) {
              throw new Error("Unexpected response from upload service.");
            }
            if (ac.signal.aborted) return;
            setShareImageUrl(text);
            const qr = await QRCode.toDataURL(text, {
              width: 220,
              margin: 2,
              color: { dark: "#141414", light: "#ffffff" },
            });
            if (ac.signal.aborted) return;
            setShareQrDataUrl(qr);
            setSharePhase("ready");
          } catch (e) {
            if (ac.signal.aborted) return;
            if (e instanceof DOMException && e.name === "AbortError") return;
            setSharePhase("error");
            setShareError(e instanceof Error ? e.message : "Download failed.");
          }
        },
        "image/png",
      );
    } catch {
      setSharePhase("error");
      setShareError("Could not read image.");
    }
  }, []);

  useEffect(() => {
    if (!shareOpen) return;
    const onKey = (e) => {
      if (e.key === "Escape") closeShareModal();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [shareOpen, closeShareModal]);

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
      lineWidth: 2.25,
      coeffsPerSecond: 65,
      loop: false,
      autoSeam: true,
      seamGapFraction: 0.02,
      onM: onAnimM,
      onComplete: () => setResultAnimPlaying(false),
    });
  }, [phase, resultPath, resultReplayKey, resultFixedM, onAnimM]);

  useEffect(() => {
    const sync = () => {
      const fs = getActiveFullscreenElement();
      setIsFullscreen(fs !== null && fs === appRootRef.current);
    };
    sync();
    document.addEventListener("fullscreenchange", sync);
    document.addEventListener("webkitfullscreenchange", sync);
    return () => {
      document.removeEventListener("fullscreenchange", sync);
      document.removeEventListener("webkitfullscreenchange", sync);
    };
  }, []);

  const enterFullscreen = useCallback(async () => {
    const el = appRootRef.current;
    if (!el) return;
    try {
      if (el.requestFullscreen) await el.requestFullscreen();
      else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
    } catch {
      /* blocked or unsupported */
    }
  }, []);

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
    <div ref={appRootRef} className="app-root">
      <div className="app-top-bar">
        <button type="button" className="app-brand app-brand--button" onClick={goToGeneratePage}>
          BioGlyph
        </button>
        {!isFullscreen ? (
          <button
            type="button"
            className="app-fullscreen-btn"
            aria-label="Enter full screen"
            onClick={() => void enterFullscreen()}
          >
            <Maximize2 size={18} strokeWidth={2} aria-hidden />
          </button>
        ) : null}
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
                  const rows = savedGenerations.map((g) => ({
                    g,
                    layout: bubbleLayoutForId(g.id),
                  }));
                  const renderBand = (list) =>
                    list.map(({ g, layout }) => (
                      <button
                        key={g.id}
                        type="button"
                        className="history-bubble-anchor"
                        style={layout.anchor}
                        onClick={() => openSavedGeneration(g)}
                        aria-label="Open saved generation"
                      >
                        <span className="history-bubble" style={layout.disc}>
                          <HistoryBubbleFace path={g.path} />
                        </span>
                      </button>
                    ));
                  return (
                    <>
                      <div className="bubble-band bubble-band--top">
                        {renderBand(rows.filter((r) => r.layout.topBand))}
                      </div>
                      <div className="bubble-band-spacer" />
                      <div className="bubble-band bubble-band--bottom">
                        {renderBand(rows.filter((r) => !r.layout.topBand))}
                      </div>
                    </>
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
                <p className="stage__tip">Place your face in the circle</p>
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
              <div className="result-actions__row">
                <button type="button" className="btn" onClick={openShareModal}>
                  Download to your phone
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

      {shareOpen ? (
        <div
          className="share-modal"
          role="dialog"
          aria-modal="true"
          aria-labelledby="share-modal-title"
        >
          <button
            type="button"
            className="share-modal__backdrop"
            aria-label="Close download dialog"
            onClick={closeShareModal}
          />
          <div className="share-modal__panel">
            <div className="share-modal__header">
              <h2 id="share-modal-title" className="share-modal__title">
                Download to your phone
              </h2>
              <button
                type="button"
                className="share-modal__close"
                aria-label="Close"
                onClick={closeShareModal}
              >
                <X size={20} strokeWidth={2} aria-hidden />
              </button>
            </div>
            {sharePhase === "uploading" ? (
              <p className="share-modal__status">Preparing link…</p>
            ) : null}
            {sharePhase === "error" ? (
              <p className="share-modal__error" role="alert">
                {shareError ?? "Something went wrong."}
              </p>
            ) : null}
            {sharePhase === "ready" && shareQrDataUrl ? (
              <div className="share-modal__body">
                <img
                  src={shareQrDataUrl}
                  alt="QR code linking to this image"
                  className="share-modal__qr"
                  width={220}
                  height={220}
                />
                <p className="share-modal__hint">
                  Scan to open the image, then save it from your browser. Link expires in about one
                  hour.
                </p>
                {shareImageUrl ? (
                  <a
                    href={shareImageUrl}
                    className="share-modal__link"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Open link
                  </a>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
