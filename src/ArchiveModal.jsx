import { useEffect } from "react";
import { Link } from "react-router-dom";
import { X } from "lucide-react";
import { pathToBubbleSvg } from "./generationStorage.js";

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

/**
 * @param {{ open: boolean, onClose: () => void, items: { id: string, path: number[][] }[] }} props
 */
export function ArchiveModal({ open, onClose, items }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  if (!open) return null;

  return (
    <div className="archive-modal-layer" role="presentation">
      <button type="button" className="archive-modal-backdrop" aria-label="Close archive" onClick={onClose} />
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
              {items.length} {items.length === 1 ? "face" : "faces"}
            </p>
          </div>
          <button type="button" className="archive-modal__close" onClick={onClose} aria-label="Close">
            <X size={22} strokeWidth={2} aria-hidden />
          </button>
        </header>
        <div className="archive-modal__body">
          {items.length === 0 ? (
            <p className="archive-modal__empty">No saved faces in the archive yet.</p>
          ) : (
            <div className="archive-grid">
              {items.map((g) => (
                <Link
                  key={g.id}
                  to={`/face/${encodeURIComponent(g.id)}`}
                  className="archive-grid__cell"
                  onClick={onClose}
                  aria-label={`View archive entry ${g.id}`}
                >
                  <ArchiveFaceThumb path={g.path} />
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
