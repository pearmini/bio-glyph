import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./App.css";
import { loadGenerations, pathToBubbleSvg } from "./generationStorage.js";

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

export default function ArchivePage() {
  const navigate = useNavigate();
  const [savedGenerations] = useState(() => loadGenerations());
  const items = useMemo(
    () => savedGenerations.filter((g) => g.path && g.path.length >= 2),
    [savedGenerations],
  );

  return (
    <div className="app-root app-root--archive-full">
      <main className="archive-page">
        <header className="archive-page__head">
          <button
            type="button"
            className="app-brand app-brand--button archive-page__brand"
            onClick={() => navigate("/")}
          >
            BioGlyph
          </button>
          <h1 className="archive-page__title-line">
            <span className="archive-page__event">Participants in ITP Spring Show 2026</span>
            <span className="archive-page__tail">
              <span className="archive-page__meta" aria-hidden="true">
                {"\u2009"}
                ·{"\u2009"}
              </span>
              <span className="archive-page__count">
                {items.length} {items.length === 1 ? "face" : "faces"}
              </span>
            </span>
          </h1>
          <button
            type="button"
            className="app-archive-btn archive-page__back"
            onClick={() => navigate("/")}
          >
            Back
          </button>
        </header>
        {items.length === 0 ? (
          <div className="archive-page__empty-wrap">
            <p className="archive-page__empty">No saved faces in the archive yet.</p>
          </div>
        ) : (
          <div className="archive-grid">
            {items.map((g) => (
              <Link
                key={g.id}
                to={`/face/${encodeURIComponent(g.id)}`}
                className="archive-grid__cell"
                aria-label={`View archive entry ${g.id}`}
              >
                <ArchiveFaceThumb path={g.path} />
              </Link>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
