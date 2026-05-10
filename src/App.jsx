import { useCallback, useEffect, useRef, useState } from "react";
import {
  drawOneLineFaceToCanvas,
  extractFaceFeaturesFromImage,
  syncOverlaySize,
} from "./facePipeline.js";
import "./App.css";

/** @typedef {"idle" | "preview" | "captured"} CapturePhase */

export default function App() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayRef = useRef(null);
  const stageOutlinesRef = useRef(null);
  const streamRef = useRef(null);

  /** @type {[CapturePhase, React.Dispatch<React.SetStateAction<CapturePhase>>]} */
  const [phase, setPhase] = useState("idle");
  const [cameraError, setCameraError] = useState(null);
  const [extractError, setExtractError] = useState(null);
  /** Full-screen centered outline only; hides capture/compare UI (not a modal). */
  const [outlineResultView, setOutlineResultView] = useState(false);
  /** PNG data URL of the outline when user confirms. */
  const [confirmedOutlineUrl, setConfirmedOutlineUrl] = useState(null);

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

  const runOnSource = useCallback(async (sourceEl) => {
    const overlay = overlayRef.current;
    if (!sourceEl || !overlay) return;
    syncOverlaySize(sourceEl, overlay, stageOutlinesRef.current);
    const extracted = await extractFaceFeaturesFromImage(sourceEl);
    if (extracted.ok) {
      drawOneLineFaceToCanvas(overlay, extracted.features);
      setExtractError(null);
    } else {
      clearOverlay();
      setExtractError(extracted.message);
    }
  }, [clearOverlay]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setExtractError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      setPhase("preview");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setCameraError(msg);
    }
  }, []);

  const capture = useCallback(async () => {
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
    setPhase("captured");

    await runOnSource(canvas);
  }, [runOnSource, stopStream]);

  const cancelPreview = useCallback(() => {
    stopStream();
    clearOverlay();
    setPhase("idle");
    setExtractError(null);
  }, [stopStream, clearOverlay]);

  const retake = useCallback(() => {
    stopStream();
    clearOverlay();
    setPhase("idle");
    setExtractError(null);
    setOutlineResultView(false);
    setConfirmedOutlineUrl(null);
  }, [stopStream, clearOverlay]);

  const openConfirmedOutline = useCallback(() => {
    if (extractError) return;
    const overlay = overlayRef.current;
    if (!overlay?.width) return;
    try {
      setConfirmedOutlineUrl(overlay.toDataURL("image/png"));
      setOutlineResultView(true);
    } catch {
      /* canvas may be tainted in edge cases */
    }
  }, [extractError]);

  const leaveOutlineResultView = useCallback(() => {
    setOutlineResultView(false);
  }, []);

  /** After paint, `phase === "preview"` shows the video and we attach the MediaStream. */
  useEffect(() => {
    if (phase !== "preview") return;
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!video || !stream) return;

    video.srcObject = stream;
    void video.play();

    const sync = () => {
      syncOverlaySize(video, overlayRef.current, stageOutlinesRef.current);
    };

    video.addEventListener("loadedmetadata", sync);
    if (video.readyState >= 1) sync();

    return () => {
      video.removeEventListener("loadedmetadata", sync);
      if (video.srcObject === stream) video.srcObject = null;
    };
  }, [phase]);

  useEffect(() => {
    if (!outlineResultView) return;
    const onKey = (e) => {
      if (e.key === "Escape") leaveOutlineResultView();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [outlineResultView, leaveOutlineResultView]);

  const canConfirmOutline = phase === "captured" && !extractError;

  return (
    <div className={`app-root${outlineResultView ? " app-root--outline-result" : ""}`}>
      {!outlineResultView && (
        <>
      <header className="toolbar">
        {phase === "idle" && (
          <button type="button" className="toolbar__btn" onClick={startCamera}>
            Start camera
          </button>
        )}
        {phase === "preview" && (
          <>
            <span className="toolbar__hint">Live preview — frame your face, then capture.</span>
            <button type="button" className="toolbar__btn toolbar__btn--primary" onClick={capture}>
              Capture
            </button>
            <button type="button" className="toolbar__btn" onClick={cancelPreview}>
              Cancel
            </button>
          </>
        )}
        {phase === "captured" && (
          <>
            <button
              type="button"
              className="toolbar__btn toolbar__btn--primary"
              onClick={openConfirmedOutline}
              disabled={!canConfirmOutline}
            >
              Confirm image
            </button>
            <button type="button" className="toolbar__btn" onClick={retake}>
              Retake
            </button>
          </>
        )}
        {cameraError && <span className="toolbar__msg toolbar__msg--error">{cameraError}</span>}
        {extractError && <span className="toolbar__msg toolbar__msg--error">{extractError}</span>}
      </header>

      <div className="compare">
        <div className={`source-slot ${phase === "preview" ? "source-slot--live" : ""}`}>
          {phase === "idle" && (
            <div className="source source--placeholder">
              <p className="source--placeholder__hint">Start the camera, then capture a frame to outline your face.</p>
            </div>
          )}
          <video
            ref={videoRef}
            className="source source--mirror source--live"
            playsInline
            muted
            autoPlay
            style={{ display: phase === "preview" ? "block" : "none" }}
          />
          <canvas
            ref={captureCanvasRef}
            className="source"
            style={{ display: phase === "captured" ? "block" : "none" }}
            aria-hidden={phase !== "captured"}
          />
        </div>

        <div className="output" ref={stageOutlinesRef}>
          <canvas ref={overlayRef} />
        </div>
      </div>
        </>
      )}

      {outlineResultView && confirmedOutlineUrl && (
        <div className="outline-result">
          <button type="button" className="outline-result__back toolbar__btn" onClick={leaveOutlineResultView}>
            Back
          </button>
          <div className="outline-result__stage">
            <img src={confirmedOutlineUrl} alt="" className="outline-result__img" />
          </div>
        </div>
      )}
    </div>
  );
}
