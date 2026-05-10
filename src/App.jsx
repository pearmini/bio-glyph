import { useCallback, useEffect, useRef, useState } from "react";
import {
  drawOneLineFaceToCanvas,
  extractFaceFeaturesFromImage,
  syncOverlaySize,
} from "./facePipeline.js";
import "./App.css";

const VIDEO_CONSTRAINTS = {
  video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
  audio: false,
};

/** @typedef {"idle" | "preview" | "generating" | "result"} AppPhase */

export default function App() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayRef = useRef(null);
  const streamRef = useRef(null);

  /** @type {[AppPhase, React.Dispatch<React.SetStateAction<AppPhase>>]} */
  const [phase, setPhase] = useState("idle");
  const [cameraError, setCameraError] = useState(null);
  const [extractError, setExtractError] = useState(null);
  const [resultImageUrl, setResultImageUrl] = useState(null);
  /** Bumps when a new MediaStream is attached so the preview effect re-runs after async getUserMedia. */
  const [previewSession, setPreviewSession] = useState(0);

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
      if (!sourceEl || !overlay) return false;
      syncOverlaySize(sourceEl, overlay, null);
      const extracted = await extractFaceFeaturesFromImage(sourceEl);
      if (extracted.ok) {
        drawOneLineFaceToCanvas(overlay, extracted.features);
        setExtractError(null);
        return true;
      }
      clearOverlay();
      setExtractError(extracted.message);
      return false;
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
    setResultImageUrl(null);
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
    ctx.save();
    ctx.translate(vw, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, vw, vh);
    ctx.restore();

    stopStream();
    setPhase("generating");
    setCameraError(null);

    const ok = await runOnSource(canvas);
    if (ok) {
      const overlay = overlayRef.current;
      try {
        const url = overlay?.toDataURL("image/png");
        if (url) {
          setResultImageUrl(url);
          setPhase("result");
        } else {
          void startCamera();
        }
      } catch {
        void startCamera();
      }
    } else {
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

  return (
    <div className="app-root">
      <main className="stage">
        {phase === "idle" && (
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => {
              setExtractError(null);
              void startCamera();
            }}
          >
            Start camera
          </button>
        )}

        {phase === "preview" && (
          <div className="stage__column">
            <div className="circle-viewport">
              <video
                ref={videoRef}
                className="circle-viewport__video circle-viewport__video--mirror"
                playsInline
                muted
                autoPlay
              />
            </div>
            <p className="stage__tip">Place your face in the circle</p>
            <button type="button" className="btn btn--primary" onClick={() => void generate()}>
              Generate
            </button>
          </div>
        )}

        {phase === "generating" && <p className="stage__generating">Generating…</p>}

        {phase === "result" && resultImageUrl && (
          <div className="stage__column stage__column--result">
            <img src={resultImageUrl} alt="One-line face outline" className="stage__result-img" />
            <button type="button" className="btn" onClick={() => void retake()}>
              Retake
            </button>
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
