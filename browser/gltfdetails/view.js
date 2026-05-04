"use strict";

const DEFAULT_SOURCE = "door.gltf.glb";
const TEXT_DECODER = new TextDecoder("utf-8");
const MAX_INLINE_ARRAY_ITEMS = 24;
const MAX_INLINE_OBJECT_KEYS = 12;
const MAX_STRING_LENGTH = 180;

const GLB_MAGIC = 0x46546c67;
const GLB_JSON_CHUNK = 0x4e4f534a;
const GLB_BIN_CHUNK = 0x004e4942;

const form = document.querySelector("#sourceForm");
const input = document.querySelector("#sourceInput");
const statusEl = document.querySelector("#status");
const treeEl = document.querySelector("#tree");

const impliedIndexRules = [
  { parent: "gltf", key: "scene", type: "scenes" },
  { parent: "scenes", key: "nodes", type: "nodes" },
  { parent: "nodes", key: "children", type: "nodes" },
  { parent: "nodes", key: "mesh", type: "meshes" },
  { parent: "nodes", key: "skin", type: "skins" },
  { parent: "nodes", key: "camera", type: "cameras" },
  { parent: "primitives", key: "attributes", type: "accessors", appliesToObjectValues: true },
  { parent: "primitives", key: "indices", type: "accessors" },
  { parent: "primitives", key: "material", type: "materials" },
  { parent: "accessors", key: "bufferView", type: "bufferViews" },
  { parent: "bufferViews", key: "buffer", type: "buffers" },
  { parent: "images", key: "bufferView", type: "bufferViews" },
  { parent: "textures", key: "sampler", type: "samplers" },
  { parent: "textures", key: "source", type: "images" },
  { parent: "skins", key: "joints", type: "nodes" },
  { parent: "skins", key: "skeleton", type: "nodes" },
  { parent: "skins", key: "inverseBindMatrices", type: "accessors" },
  { parent: "channels", key: "sampler", type: "animation.samplers" },
  { parent: "target", key: "node", type: "nodes" },
  { parent: "samplers", key: "input", type: "accessors" },
  { parent: "samplers", key: "output", type: "accessors" },
  { parent: "materials", key: "normalTexture", type: "textures", nestedIndexKey: "index" },
  { parent: "materials", key: "occlusionTexture", type: "textures", nestedIndexKey: "index" },
  { parent: "materials", key: "emissiveTexture", type: "textures", nestedIndexKey: "index" },
  { parent: "pbrMetallicRoughness", key: "baseColorTexture", type: "textures", nestedIndexKey: "index" },
  { parent: "pbrMetallicRoughness", key: "metallicRoughnessTexture", type: "textures", nestedIndexKey: "index" },
  { parent: "KHR_materials_pbrSpecularGlossiness", key: "diffuseTexture", type: "textures", nestedIndexKey: "index" },
  { parent: "KHR_materials_pbrSpecularGlossiness", key: "specularGlossinessTexture", type: "textures", nestedIndexKey: "index" },
  { parent: "KHR_texture_transform", key: "texCoord", type: "texCoords" },
  { parent: "textureInfo", key: "index", type: "textures" }
];

form.addEventListener("submit", event => {
  event.preventDefault();
  const source = input.value.trim() || DEFAULT_SOURCE;
  setSourceInUrl(source);
  loadSource(source);
});

treeEl.addEventListener("click", event => {
  const button = event.target.closest(".toggle");
  if (!button) {
    return;
  }

  const node = button.closest(".tree-node");
  const children = node.querySelector(":scope > .children");
  if (!children) {
    return;
  }

  const collapsed = node.classList.toggle("collapsed");
  children.hidden = collapsed;
  button.setAttribute("aria-expanded", String(!collapsed));
});

treeEl.addEventListener("keydown", event => {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }

  const button = event.target.closest(".toggle");
  if (!button) {
    return;
  }

  event.preventDefault();
  button.click();
});

loadSource(getSourceFromUrl());

function getSourceFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("src") || params.get("url") || params.get("file");
  const fromHash = window.location.hash ? decodeURIComponent(window.location.hash.slice(1)) : "";
  return fromQuery || fromHash || DEFAULT_SOURCE;
}

function setSourceInUrl(source) {
  const url = new URL(window.location.href);
  url.searchParams.set("src", source);
  url.hash = "";
  window.history.replaceState(null, "", url);
}

async function loadSource(source) {
  input.value = source;
  setStatus("Loading");
  renderMessage("Loading");

  try {
    const details = await loadGltf(source);
    const root = makeRoot(details);
    renderTree(root);
    setStatus(statusFor(details));
  } catch (error) {
    console.error(error);
    renderError(error);
    setStatus("Load failed");
  }
}

async function loadGltf(source) {
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`${source}: ${response.status} ${response.statusText}`);
  }

  const contentType = response.headers.get("content-type") || "";
  const lowerSource = source.toLowerCase();

  if (lowerSource.endsWith(".gltf") || contentType.includes("json")) {
    return {
      source,
      format: "gltf",
      json: await response.json(),
      byteLength: Number(response.headers.get("content-length")) || null
    };
  }

  const arrayBuffer = await response.arrayBuffer();
  const parsed = parseGlb(arrayBuffer);
  return {
    source,
    format: "glb",
    json: parsed.json,
    byteLength: arrayBuffer.byteLength,
    chunks: parsed.chunks
  };
}

function parseGlb(arrayBuffer) {
  const dataView = new DataView(arrayBuffer);
  if (dataView.byteLength < 20) {
    throw new Error("GLB file is too small.");
  }

  const magic = dataView.getUint32(0, true);
  if (magic !== GLB_MAGIC) {
    throw new Error("File is not a GLB binary.");
  }

  const version = dataView.getUint32(4, true);
  if (version !== 2) {
    throw new Error(`Unsupported GLB version ${version}.`);
  }

  const declaredLength = dataView.getUint32(8, true);
  if (declaredLength > dataView.byteLength) {
    throw new Error("GLB declared length exceeds downloaded data.");
  }

  const chunks = [];
  let offset = 12;
  let json = null;

  while (offset + 8 <= declaredLength) {
    const chunkLength = dataView.getUint32(offset, true);
    const chunkType = dataView.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    const chunkEnd = chunkStart + chunkLength;

    if (chunkEnd > declaredLength) {
      throw new Error("GLB chunk exceeds declared file length.");
    }

    chunks.push({
      type: chunkTypeName(chunkType),
      byteOffset: chunkStart,
      byteLength: chunkLength
    });

    if (chunkType === GLB_JSON_CHUNK) {
      const jsonText = TEXT_DECODER.decode(arrayBuffer.slice(chunkStart, chunkEnd)).trim();
      json = JSON.parse(jsonText);
    }

    offset = chunkEnd;
  }

  if (!json) {
    throw new Error("GLB has no JSON chunk.");
  }

  return { json, chunks };
}

function chunkTypeName(type) {
  if (type === GLB_JSON_CHUNK) {
    return "JSON";
  }
  if (type === GLB_BIN_CHUNK) {
    return "BIN";
  }
  return `0x${type.toString(16).padStart(8, "0")}`;
}

function makeRoot(details) {
  const root = {
    source: details.source,
    format: details.format,
    byteLength: details.byteLength,
    gltf: details.json
  };

  if (details.chunks) {
    root.glbChunks = details.chunks;
  }

  return root;
}

function statusFor(details) {
  const assetVersion = details.json.asset && details.json.asset.version
    ? `glTF ${details.json.asset.version}`
    : "glTF";
  const size = details.byteLength ? `, ${formatBytes(details.byteLength)}` : "";
  return `${assetVersion}${size}`;
}

function renderTree(value) {
  treeEl.replaceChildren(renderNode("root", value, ["root"]));
}

function renderMessage(message) {
  const el = document.createElement("div");
  el.className = "tree-empty";
  el.textContent = message;
  treeEl.replaceChildren(el);
}

function renderError(error) {
  const el = document.createElement("div");
  el.className = "tree-error";
  el.textContent = error instanceof Error ? error.message : String(error);
  treeEl.replaceChildren(el);
}

function setStatus(message) {
  statusEl.textContent = message;
}

function renderNode(key, value, path) {
  const hasChildren = isExpandable(value);
  const node = document.createElement("div");
  node.className = "tree-node";

  const row = document.createElement("div");
  row.className = "row";

  if (hasChildren) {
    const toggle = document.createElement("button");
    toggle.className = "toggle";
    toggle.type = "button";
    toggle.setAttribute("aria-label", `Toggle ${key}`);
    toggle.setAttribute("aria-expanded", "true");
    row.append(toggle);
  } else {
    const spacer = document.createElement("span");
    spacer.className = "leaf-space";
    row.append(spacer);
  }

  row.append(renderKey(key, path));
  row.append(renderSeparator());
  row.append(renderSummary(value, path));
  node.append(row);

  if (hasChildren) {
    const children = document.createElement("div");
    children.className = "children";

    for (const [childKey, childValue] of childEntries(value)) {
      children.append(renderNode(childKey, childValue, path.concat(String(childKey))));
    }

    node.append(children);
  }

  return node;
}

function renderKey(key, path) {
  const el = document.createElement("span");
  el.className = isImpliedIndexPath(path) ? "key index-key" : "key";
  el.textContent = key;
  return el;
}

function renderSeparator() {
  const el = document.createElement("span");
  el.className = "punct";
  el.textContent = ":";
  return el;
}

function renderSummary(value, path) {
  const implied = impliedIndexForPath(path, value);
  if (implied) {
    const el = document.createElement("span");
    el.className = "implied-index";
    el.textContent = `@${implied.type}[${value}]`;
    return el;
  }

  if (Array.isArray(value)) {
    return renderArraySummary(value, path);
  }

  if (value && typeof value === "object") {
    return renderObjectSummary(value);
  }

  return renderScalar(value);
}

function renderArraySummary(array, path) {
  const el = document.createElement("span");
  const impliedType = impliedArrayElementType(path);

  if (array.length === 0) {
    el.className = "summary";
    el.textContent = "[]";
    return el;
  }

  if (isLargeNumericArray(array)) {
    el.className = "truncated";
    el.textContent = `[${array.slice(0, MAX_INLINE_ARRAY_ITEMS).join(", ")}, ...] (${array.length} items)`;
    return el;
  }

  if (array.length > MAX_INLINE_ARRAY_ITEMS || containsObjects(array)) {
    el.className = "summary";
    el.textContent = `[${array.length} items]`;
    return el;
  }

  el.className = impliedType ? "implied-index" : "summary";
  el.textContent = `[${array.map(item => formatInlineValue(item, impliedType)).join(", ")}]`;
  return el;
}

function renderObjectSummary(object) {
  const keys = Object.keys(object);
  const el = document.createElement("span");
  el.className = "summary";

  if (keys.length === 0) {
    el.textContent = "{}";
    return el;
  }

  if (keys.length > MAX_INLINE_OBJECT_KEYS) {
    el.textContent = `{${keys.length} keys}`;
    return el;
  }

  el.textContent = `{${keys.join(", ")}}`;
  return el;
}

function renderScalar(value) {
  const el = document.createElement("span");
  const type = value === null ? "null" : typeof value;
  el.className = type;
  el.textContent = formatInlineValue(value);
  return el;
}

function childEntries(value) {
  if (Array.isArray(value)) {
    return value.map((item, index) => [index, item]);
  }
  return Object.entries(value);
}

function isExpandable(value) {
  if (!value || typeof value !== "object") {
    return false;
  }
  if (Array.isArray(value)) {
    return value.length > 0 && containsObjects(value);
  }
  return Object.keys(value).length > 0;
}

function containsObjects(array) {
  return array.some(item => item && typeof item === "object");
}

function isLargeNumericArray(array) {
  return array.length > MAX_INLINE_ARRAY_ITEMS && array.every(item => typeof item === "number");
}

function formatInlineValue(value, impliedType = null) {
  if (impliedType && Number.isInteger(value)) {
    return `@${impliedType}[${value}]`;
  }

  if (typeof value === "string") {
    return JSON.stringify(truncateString(value));
  }

  if (value === null) {
    return "null";
  }

  if (value === undefined) {
    return "undefined";
  }

  return String(value);
}

function truncateString(value) {
  if (value.length <= MAX_STRING_LENGTH) {
    return value;
  }
  return `${value.slice(0, MAX_STRING_LENGTH)}... (${value.length} chars)`;
}

function impliedIndexForPath(path, value) {
  if (!Number.isInteger(value)) {
    return null;
  }

  const parentKey = path[path.length - 2];
  const key = path[path.length - 1];
  const ownerCollection = collectionForPropertyPath(path);
  const arrayOwnerCollection = collectionForArrayItemPath(path);
  const objectValueOwnerCollection = collectionForObjectValuePath(path);

  for (const rule of impliedIndexRules) {
    if (
      rule.nestedIndexKey &&
      key === rule.nestedIndexKey &&
      parentKey === rule.key &&
      collectionForNestedIndexPath(path) === rule.parent
    ) {
      return { type: rule.type };
    }

    if (
      rule.appliesToObjectValues &&
      parentKey === rule.key &&
      objectValueOwnerCollection === rule.parent
    ) {
      return { type: rule.type };
    }

    if (
      rule.key === key &&
      (ownerCollection === rule.parent || parentKey === rule.parent) &&
      !rule.nestedIndexKey &&
      !rule.appliesToObjectValues
    ) {
      return { type: rule.type };
    }

    if (
      isArrayIndex(key) &&
      parentKey === rule.key &&
      arrayOwnerCollection === rule.parent &&
      !rule.nestedIndexKey &&
      !rule.appliesToObjectValues
    ) {
      return { type: rule.type };
    }
  }

  if (key === "index" && isTextureInfoPath(path)) {
    return { type: "textures" };
  }

  return null;
}

function impliedArrayElementType(path) {
  const key = path[path.length - 1];
  const ownerCollection = collectionForPropertyPath(path);

  for (const rule of impliedIndexRules) {
    if (
      rule.key === key &&
      (rule.parent === ownerCollection || rule.parent === path[path.length - 2]) &&
      !rule.nestedIndexKey
    ) {
      return rule.type;
    }
  }

  return null;
}

function isImpliedIndexPath(path) {
  return Boolean(impliedIndexForPath(path, 0)) || Boolean(impliedArrayElementType(path));
}

function isTextureInfoPath(path) {
  const textureInfoKeys = new Set([
    "baseColorTexture",
    "metallicRoughnessTexture",
    "normalTexture",
    "occlusionTexture",
    "emissiveTexture",
    "diffuseTexture",
    "specularGlossinessTexture"
  ]);
  return path.length >= 3 && textureInfoKeys.has(path[path.length - 2]);
}

function collectionForPropertyPath(path) {
  if (path.length < 3) {
    return null;
  }
  return path[path.length - 3];
}

function collectionForArrayItemPath(path) {
  if (path.length < 4) {
    return null;
  }
  return path[path.length - 4];
}

function collectionForObjectValuePath(path) {
  if (path.length < 4) {
    return null;
  }
  return path[path.length - 4];
}

function collectionForNestedIndexPath(path) {
  if (path.length < 4) {
    return null;
  }
  const candidate = path[path.length - 3];
  if (isArrayIndex(candidate)) {
    return path[path.length - 4];
  }
  return candidate;
}

function isArrayIndex(key) {
  return String(Number(key)) === String(key) && Number.isInteger(Number(key));
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) {
    return "";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  const precision = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}
