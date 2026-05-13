import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import "./App.css";
import { Play } from "lucide-react";
import { getFourierReconstructionContours, startFourierOneLineAnimation } from "./fourierOneLineAnimation.js";
import { loadGenerations, pathSegmentsToBubbleSvg } from "./generationStorage.js";
import { triggerFileDownload } from "./fileDownload.js";
import { GitHubMark } from "./GitHubMark.jsx";
import { ArchiveModal } from "./ArchiveModal.jsx";

const RESULT_EPICYCLES = 320;
const RESULT_LINE_CSS_PX = 2.25;
const RESULT_PATH_FIT = 0.88;
const RESULT_VIEWPORT_MAX_CSS = 480;
const RESULT_EXPORT_SVG_SIZE = 1024;
const FOURIER_SVG_EXPORT = {
  samples: 2048,
  outSamples: 1500,
  autoSeam: true,
  seamGapFraction: 0.02,
};

export default function FacePage() {
  const { id: idParam } = useParams();
  const id = typeof idParam === "string" ? decodeURIComponent(idParam) : "";

  const [savedGenerations] = useState(() => loadGenerations());
  const archiveGridItems = useMemo(
    () => savedGenerations.filter((g) => g.path && g.path.length >= 2),
    [savedGenerations],
  );

  const record = useMemo(
    () => archiveGridItems.find((g) => g.id === id) ?? null,
    [archiveGridItems, id],
  );

  const pathForCanvas = useMemo(() => {
    if (record?.path && record.path.length >= 2) return record.path;
    return null;
  }, [record]);

  const [archiveOpen, setArchiveOpen] = useState(false);
  const resultCanvasRef = useRef(null);
  const [resultAnimPlaying, setResultAnimPlaying] = useState(false);
  const [resultReplayKey, setResultReplayKey] = useState(0);
  const [resultFixedM, setResultFixedM] = useState(null);
  const [resultDisplayM, setResultDisplayM] = useState(RESULT_EPICYCLES);

  const onAnimM = useCallback((m) => {
    setResultDisplayM(m);
  }, []);

  useEffect(() => {
    if (!pathForCanvas || pathForCanvas.length < 2) return;
    const canvas = resultCanvasRef.current;
    if (!canvas) return;

    return startFourierOneLineAnimation(canvas, pathForCanvas, {
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
  }, [pathForCanvas, resultReplayKey, resultFixedM, onAnimM]);

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
    if (!pathForCanvas || pathForCanvas.length < 2) return;
    const contours = getFourierReconstructionContours(pathForCanvas, {
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
  }, [pathForCanvas]);

  const hasFace = Boolean(record?.path && record.path.length >= 2);

  return (
    <div className="app-root">
      <div className="app-top-bar">
        <Link to="/" className="app-brand app-brand--button">
          BioGlyph
        </Link>
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
      <ArchiveModal open={archiveOpen} onClose={() => setArchiveOpen(false)} items={archiveGridItems} />

      {hasFace ? (
        <main className="stage stage--result">
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
                <Link to="/" className="btn">
                  Back
                </Link>
              </div>
            </div>
          </div>
        </main>
      ) : (
        <main className="stage stage--idle">
          <div className="stage__column">
            <p className="stage__error" role="alert">
              {id ? "This face is not in the archive." : "Missing face id."}
            </p>
            <Link to="/" className="btn btn--dark">
              Back to home
            </Link>
          </div>
        </main>
      )}
    </div>
  );
}

export function FacePageRoute() {
  const { id } = useParams();
  return <FacePage key={id} />;
}
