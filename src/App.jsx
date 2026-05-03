import { useCallback, useEffect, useRef } from "react";
import {
  drawFaceFeaturesToCanvas,
  extractFaceFeaturesFromImage,
  syncOverlaySize,
} from "./facePipeline.js";
import "./App.css";

const FACE_IMAGE = "/face.jpg";

export default function App() {
  const imgRef = useRef(null);
  const overlayRef = useRef(null);
  const stageOutlinesRef = useRef(null);

  const runOnCurrentImage = useCallback(async () => {
    const imgEl = imgRef.current;
    const overlay = overlayRef.current;
    if (!imgEl || !overlay) return;
    syncOverlaySize(imgEl, overlay, stageOutlinesRef.current);
    const extracted = await extractFaceFeaturesFromImage(imgEl);
    if (extracted.ok) drawFaceFeaturesToCanvas(overlay, extracted.features);
  }, []);

  useEffect(() => {
    const imgEl = imgRef.current;
    if (!imgEl) return;

    const onLoad = async () => {
      syncOverlaySize(imgEl, overlayRef.current, stageOutlinesRef.current);
      await runOnCurrentImage();
    };

    imgEl.addEventListener("load", onLoad);
    imgEl.src = FACE_IMAGE;
    if (imgEl.complete && imgEl.naturalWidth > 0) {
      void onLoad();
    }

    return () => {
      imgEl.removeEventListener("load", onLoad);
    };
  }, [runOnCurrentImage]);

  return (
    <div className="compare">
      <img ref={imgRef} className="source" alt="" crossOrigin="anonymous" />
      <div className="output" ref={stageOutlinesRef}>
        <canvas ref={overlayRef} />
      </div>
    </div>
  );
}
