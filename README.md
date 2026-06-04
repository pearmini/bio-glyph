# BioGlyph

**One line. Your face. Nothing else.**

[![Live demo](https://img.shields.io/badge/demo-bio.bairui.dev-141414?style=for-the-badge)](https://bio.bairui.dev/)
[![ITP Spring Show 2026](https://img.shields.io/badge/ITP-Spring%20Show%202026-f6f6f6?style=for-the-badge&color=141414)](https://itp.nyu.edu/shows/spring2026/)

![BioGlyph one-line portrait](./img/face.png)

BioGlyph is an interactive web app that turns your face into a single continuous line — a personal signature drawn from your own features, not a filter.

Built in six hours for the [ITP Spring Show 2026](https://itp.nyu.edu/shows/spring2026/), it was tried by **150+ people** who collectively archived **428 faces**. Friends gathered around the screen, watched their portraits emerge from nothing, and took home something uniquely theirs.

![Selected one-line portraits from ITP Spring Show 2026](./img/examples.jpg)

> *Simple things still matter.*

[**Try it live →**](https://bio.bairui.dev/) · [**Read the story →**](https://bairui.dev/)

## What it does

1. **Capture** — Point your webcam at the screen.
2. **Extract** — MediaPipe detects facial landmarks; image segmentation pulls out hair, ears, and contours.
3. **Connect** — Features are ordered to minimize line overlap and merged into one path.
4. **Reveal** — A Fourier-series animation progressively reconstructs the drawing, inspired by [Mike Bostock's Fourier Series — Progressive](https://observablehq.com/@mbostock/fourier-series-progressive).
5. **Keep** — Download PNG or SVG, replay the animation, or browse the [archive](https://bio.bairui.dev/archive) of 428 portraits from ITP Spring Show 2026.

Up to **four faces** can be detected at once — couples, friends, whole groups — each merged left to right into a shared one-line portrait.

## Why it exists

BioGlyph explores facial features as a form of personal signature and identity through generative art. It asks a quiet question: after abstraction and sketching, what still makes a face *yours*?

The project grew out of earlier experiments — a Progressive Self Portrait using Fourier series, abandoned ideas around name-to-face procedural generation, and a digital penPal that never quite clicked. What worked was the simplest thing: point a camera, watch a line appear, feel something.

## How it works

```
Webcam / image
      │
      ▼
┌─────────────────────────────────────┐
│  MediaPipe Face Landmarker          │  eyes, nose, lips, face oval
│  MediaPipe Image Segmenter          │  hair, ears (multiclass mask)
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Contour extraction + ordering      │  minimize overlaps, merge multi-face
│  buildOneLinePath()                 │  lips → nose → eyes → hair → jaw …
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Fourier progressive reconstruction │  epicycle animation on canvas
│  Export PNG / SVG                   │
└─────────────────────────────────────┘
```

**Face pipeline** (`src/facePipeline.js`) — landmark-based feature outlines, segmentation-backed hair and ear contours, Ramer–Douglas–Peucker simplification, and greedy path ordering so the stroke reads cleanly as one line.

**Animation** (`src/fourierOneLineAnimation.js`) — resamples the path, computes DFT coefficients, and animates progressive reconstruction with adjustable epicycle count.

**Archive** (`/archive`, `/face/:id`) — bundled show data from `src/data/itp-spring-show-2026.json`; each portrait gets a shareable URL and downloadable assets.

A companion Jupyter notebook (`python/bio_glyph_face_pipeline.ipynb`) mirrors the extraction logic for offline experimentation.

## Tech stack

| Layer | Tools |
|-------|-------|
| UI | React 19, React Router, Vite |
| Vision | [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) — Face Landmarker + Selfie Multiclass Segmenter |
| Animation | Custom Fourier-series renderer (Canvas 2D) |
| Deploy | Vercel |

All inference runs **in the browser** — no server-side face processing, no API keys.

## Getting started

**Requirements:** Node.js 18+

```bash
git clone https://github.com/pearmini/bio-glyph.git
cd bio-glyph
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Allow camera access when prompted.

```bash
npm run build    # production build
npm run preview  # preview production build locally
npm run lint     # ESLint
```

MediaPipe WASM and model files load from CDN on first use; a short delay on the first generation is normal.

## Project structure

```
bio-glyph/
├── src/
│   ├── App.jsx                 # Main capture + generation flow
│   ├── facePipeline.js         # MediaPipe extraction → one-line path
│   ├── fourierOneLineAnimation.js
│   ├── generationStorage.js    # Archive data + SVG helpers
│   ├── ArchivePage.jsx         # ITP Spring Show gallery
│   ├── FacePage.jsx            # Individual portrait view
│   └── data/
│       └── itp-spring-show-2026.json
├── python/
│   └── bio_glyph_face_pipeline.ipynb
└── img/
    └── face.png
```

## Routes

| Path | Description |
|------|-------------|
| `/` | Create a new one-line portrait |
| `/archive` | Browse ITP Spring Show 2026 portraits |
| `/face/:id` | View, replay, and download a saved portrait |

## Credits

**Created by [Bairui Su](https://bairui.dev/)** — ITP, NYU

- Interaction design & visual prototyping with **Jin**
- Inspired by [Mike Bostock](https://bost.ocks.org/mike/)'s Fourier explorations and an earlier Progressive Self Portrait experiment
- Built with help from [Cursor](https://cursor.com/) and prior face-line experiments

**Show feedback that shaped the project:**
- *"Why do all the faces look so similar?"* — landmark geometry is a known limit; contour-based extraction is the next step
- *"Can multiple people try it together?"* — added after day one; up to four faces merge into one path
- *"It's nice to see face-detection used to create something, not just apply filters."* — exactly the point

## License

[MIT](./LICENSE) © 2026 Bairui Su

<p align="center">
  <a href="https://bio.bairui.dev/"><strong>bio.bairui.dev</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/pearmini/bio-glyph">Source</a>
  &nbsp;·&nbsp;
  <a href="https://bairui.dev/">Blog</a>
</p>

<p align="center"><em>Simple things still matter.</em></p>
