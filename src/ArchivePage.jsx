import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./App.css";
import { getBrowserId } from "./lib/browserId.js";
import { deleteCommunityFace, fetchCommunityFaces, isSupabaseConfigured } from "./lib/facesApi.js";
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

function DeleteIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"
      />
    </svg>
  );
}

function ArchiveSection({ title, description, count, children }) {
  return (
    <section className="archive-section">
      <header className="archive-section__header">
        <h2 className="archive-section__title">{title}</h2>
        <p className="archive-section__description">{description}</p>
        <p className="archive-section__count">
          {count} {count === 1 ? "face" : "faces"}
        </p>
      </header>
      <div className="archive-grid">{children}</div>
    </section>
  );
}

/** @param {{ id: string, path: number[][], source?: string, browserId?: string }} entry */
function ArchiveGridCell({ entry, onDelete }) {
  const isOwn = entry.source === "community" && entry.browserId === getBrowserId();

  return (
    <div className="archive-cell">
      <Link
        to={`/face/${encodeURIComponent(entry.id)}`}
        className="archive-grid__cell"
        aria-label={`View archive entry ${entry.id}`}
      >
        <ArchiveFaceThumb path={entry.path} />
      </Link>
      {isOwn && onDelete && (
        <button
          type="button"
          className="archive-own-control"
          title="Delete your face"
          aria-label="Delete your face"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onDelete();
          }}
        >
          <span className="archive-own-mark" aria-hidden="true">
            *
          </span>
          <span className="archive-own-delete" aria-hidden="true">
            <DeleteIcon />
          </span>
        </button>
      )}
    </div>
  );
}

export default function ArchivePage() {
  const navigate = useNavigate();
  const [communityFaces, setCommunityFaces] = useState([]);
  const [loadState, setLoadState] = useState("idle");
  const [loadError, setLoadError] = useState("");

  const bundledItems = useMemo(
    () => loadGenerations().filter((g) => g.path && g.path.length >= 2),
    [],
  );

  useEffect(() => {
    if (!isSupabaseConfigured()) return;
    let cancelled = false;
    setLoadState("loading");
    setLoadError("");
    fetchCommunityFaces()
      .then((faces) => {
        if (!cancelled) {
          setCommunityFaces(faces);
          setLoadState("done");
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setLoadError(err.message ?? "Could not load community archive.");
          setLoadState("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function onDeleteCommunityFace(id) {
    if (!confirm("Remove your face from the archive?")) return;
    try {
      await deleteCommunityFace(id);
      setCommunityFaces((prev) => prev.filter((f) => f.id !== id));
    } catch (err) {
      alert(err.message ?? "Could not delete your face.");
    }
  }

  return (
    <div className="app-root app-root--archive-full">
      <main className="archive-page">
        <header className="archive-page__head">
          <button
            type="button"
            className="archive-page__site-title"
            onClick={() => navigate("/")}
          >
            BioGlyph Archive
          </button>
          <button
            type="button"
            className="app-archive-btn archive-page__back"
            onClick={() => navigate("/")}
          >
            Back
          </button>
        </header>

        <div className="archive-page__body">
          {loadState === "loading" && (
            <p className="archive-status">Loading community faces…</p>
          )}
          {loadState === "error" && (
            <p className="archive-status archive-status--error">{loadError}</p>
          )}

          {communityFaces.length > 0 && (
            <ArchiveSection
              title="Community"
              description="Faces added online after the show. Create one on the home page and press Add to Archive. Hover your face (marked with *) to delete it."
              count={communityFaces.length}
            >
            {communityFaces.map((entry) => (
              <ArchiveGridCell
                key={entry.id}
                entry={entry}
                onDelete={() => onDeleteCommunityFace(entry.id)}
              />
            ))}
          </ArchiveSection>
        )}

          <ArchiveSection
            title="ITP Spring Show 2026"
            description="One-line portraits created with BioGlyph at NYU ITP — captured at the installation."
            count={bundledItems.length}
          >
            {bundledItems.length === 0 ? (
              <p className="archive-page__empty">No show portraits in the bundled archive.</p>
            ) : (
              bundledItems.map((entry) => (
                <ArchiveGridCell key={entry.id} entry={entry} />
              ))
            )}
          </ArchiveSection>
        </div>
      </main>
    </div>
  );
}
