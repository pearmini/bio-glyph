import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import App from "./App.jsx";
import ArchivePage from "./ArchivePage.jsx";
import { FacePageRoute } from "./FacePage.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/archive" element={<ArchivePage />} />
        <Route path="/face/:id" element={<FacePageRoute />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
