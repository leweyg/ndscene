import { ensureNdSceneElements } from "./ndscene_text_element.js";

ensureNdSceneElements();

const pathForm = document.querySelector("#path-form");
const pathInput = document.querySelector("#path-input");
const statusBanner = document.querySelector("#status-banner");
const resolvedPathChip = document.querySelector("#resolved-path-chip");
const sceneTree = document.querySelector("#scene-tree");

function setStatus(message, state = "idle") {
  statusBanner.textContent = message;
  statusBanner.dataset.state = state;
}

function currentQueryPath() {
  const params = new URLSearchParams(window.location.search);
  return params.get("path") || "";
}

function updateUrlForPath(path) {
  const nextUrl = new URL(window.location.href);
  if (path) {
    nextUrl.searchParams.set("path", path);
  } else {
    nextUrl.searchParams.delete("path");
  }
  window.location.href = nextUrl.toString();
}

function showEmptyState() {
  const emptyState = document.createElement("div");
  emptyState.className = "empty-state";
  emptyState.innerHTML =
    'Use <code>?path=../../json/freed_go/view_1_scene.json</code> or load a file with the form above.';
  sceneTree.replaceChildren(emptyState);
}

async function loadSceneFromPath(path) {
  if (!path) {
    resolvedPathChip.textContent = "No file loaded yet";
    setStatus("Waiting for a JSON file path.", "idle");
    sceneTree.clear();
    showEmptyState();
    return;
  }

  const resolvedUrl = new URL(path, window.location.href);
  pathInput.value = path;
  resolvedPathChip.textContent = `Resolved: ${resolvedUrl.pathname}`;
  setStatus(`Loading ${path} ...`, "loading");

  try {
    const response = await fetch(resolvedUrl, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while fetching ${resolvedUrl.pathname}`);
    }

    const json = await response.json();
    sceneTree.setData(json, {
      rootLabel: path.split("/").pop() || "scene.json",
    });
    setStatus(`Loaded ${path}`, "ready");
  } catch (error) {
    sceneTree.clear();
    showEmptyState();
    setStatus(`Unable to load JSON: ${error.message}`, "error");
  }
}

pathForm.addEventListener("submit", (event) => {
  event.preventDefault();
  updateUrlForPath(pathInput.value.trim());
});

loadSceneFromPath(currentQueryPath());
