function createElement(tagName, className, textContent) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (textContent !== undefined) {
    element.textContent = textContent;
  }
  return element;
}

function describeValueKind(value) {
  if (value === null) {
    return "null";
  }
  if (Array.isArray(value)) {
    return "array";
  }
  return typeof value;
}

function summarizeShapeValue(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => summarizeShapeValue(entry)).join(", ")}]`;
  }
  if (typeof value === "number") {
    return `${value}`;
  }
  if (!value || typeof value !== "object") {
    return String(value);
  }

  const name = value.name || value.key || "";
  const size = value.size !== undefined ? `x${value.size}` : "";
  const dtype = value.dtype ? `<${value.dtype}>` : "";
  const childShape = Array.isArray(value.shape) && value.shape.length > 0
    ? summarizeShapeValue(value.shape)
    : "";
  const dataTag = value.data !== undefined ? " =data" : "";

  return `${name}${size}${dtype}${childShape}${dataTag}`.trim() || "{}";
}

function summarizeStructuredValue(label, value) {
  if (label === "shape") {
    return summarizeShapeValue(value);
  }
  if (Array.isArray(value)) {
    return `${value.length} item${value.length === 1 ? "" : "s"}`;
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    let base = "";
    if ("name" in value) {
      base += ` "${value.name}" `;
    } else if ("key" in value) {
      base += ` "${value.key}" `;
    }
    var hasShape = false;
    if ("shape" in value) {
      const shapeValue = value['shape'];
      const shapeText = summarizeShapeValue(shapeValue);
      base += ` ${shapeText}`
      hasShape = true;
    }
    if ("dtype" in value) {
      const dtypeText = value['dtype'].toString();
      base += ` <${dtypeText}>`
    }
    if ("data" in value) {
      base += ` (data)`;
    }
    if ("children" in value) {
      const childrenValue = value['children'];
      base += ` {${childrenValue.length}c}`;
      hasShape = true;
    }
    if ("pose" in value) {
      base += ` *pose`;
      hasShape = true;
    }
    if ("unpose" in value) {
      base += ` *unpose`;
      hasShape = true;
    }
    if (!hasShape) {
      if (keys.length == 0) {
        base += " { }"
      } else {
        base += ` {.${keys.length}}`;
      }
    }
    return base;
  }
  return String(value);
}

function formatLeafValue(value) {
  if (typeof value === "string") {
    return `"${value}"`;
  }
  if (value === null) {
    return "null";
  }
  return String(value);
}

class NdSceneTypeRegistry {
  constructor() {
    this.entries = [];
  }

  register(name, test, render) {
    this.entries.push({ name, test, render });
  }

  renderNode(context) {
    for (const entry of this.entries) {
      if (entry.test(context.value)) {
        return entry.render(context, this);
      }
    }
    throw new Error("No ndscene renderer registered for value.");
  }
}

class NdSceneNodeElement extends HTMLElement {
  constructor() {
    super();
    this._initialized = false;
    this.expandable = false;
    this.expanded = true;
  }

  connectedCallback() {
    this.ensureInitialized();
  }

  ensureInitialized() {
    if (this._initialized) {
      return;
    }

    this.classList.add("ndscene-node");

    this.rowElement = createElement("div", "ndscene-row");
    this.toggleButton = createElement("button", "ndscene-toggle", "−");
    this.keyElement = createElement("div", "ndscene-key");
    this.kindElement = createElement("div", "ndscene-kind");
    this.summaryElement = createElement("div", "ndscene-summary");
    this.valueElement = createElement("div", "ndscene-value");
    this.childrenElement = createElement("div", "ndscene-children");

    this.toggleButton.type = "button";
    this.toggleButton.addEventListener("click", () => {
      this.expanded = !this.expanded;
      this.updateExpandedState();
    });

    this.rowElement.append(
      this.toggleButton,
      this.keyElement,
      this.kindElement,
      this.summaryElement,
      this.valueElement
    );

    this.append(this.rowElement, this.childrenElement);
    this._initialized = true;
  }

  setMeta({ label, kind, summary, valueText, valueType, expandable, expanded }) {
    this.ensureInitialized();
    this.keyElement.textContent = label;
    this.kindElement.textContent = kind;
    this.summaryElement.textContent = summary || "";
    this.valueElement.textContent = valueText || "";
    this.valueElement.dataset.valueType = valueType || "";
    this.expandable = Boolean(expandable);
    this.expanded = expanded !== false;
    this.toggleButton.dataset.empty = this.expandable ? "false" : "true";
    this.updateExpandedState();
  }

  setChildren(children) {
    this.ensureInitialized();
    this.childrenElement.replaceChildren(...children);
  }

  updateExpandedState() {
    this.ensureInitialized();
    this.toggleButton.textContent = this.expandable ? (this.expanded ? "−" : "+") : "·";
    this.childrenElement.hidden = !this.expandable || !this.expanded;
  }
}

class NdSceneTreeElement extends HTMLElement {
  constructor() {
    super();
    this.registry = NdSceneTreeElement.defaultRegistry;
    this.rootLabel = "scene";
    this.currentValue = undefined;
  }

  setData(value, options = {}) {
    this.currentValue = value;
    this.rootLabel = options.rootLabel || "scene";
    this.render();
  }

  clear() {
    this.currentValue = undefined;
    this.replaceChildren();
  }

  render() {
    if (this.currentValue === undefined) {
      this.replaceChildren();
      return;
    }

    const rootNode = this.registry.renderNode({
      label: this.rootLabel,
      path: this.rootLabel,
      depth: 0,
      value: this.currentValue,
    });
    this.replaceChildren(rootNode);
  }
}

function renderLeafNode(context) {
  const node = document.createElement("ndscene-node");
  node.setMeta({
    label: context.label,
    kind: describeValueKind(context.value),
    valueText: formatLeafValue(context.value),
    valueType: describeValueKind(context.value),
    expandable: false,
  });
  return node;
}

function renderArrayNode(context, registry) {
  const node = document.createElement("ndscene-node");
  const children = context.value.map((childValue, index) =>
    registry.renderNode({
      label: `[${index}]`,
      path: `${context.path}[${index}]`,
      depth: context.depth + 1,
      value: childValue,
    })
  );

  node.setMeta({
    label: context.label,
    kind: "array",
    summary: summarizeStructuredValue(context.label, context.value),
    expandable: true,
    expanded: context.depth < 2,
  });
  node.setChildren(children);
  return node;
}

function renderObjectNode(context, registry) {
  const node = document.createElement("ndscene-node");
  const keys = Object.keys(context.value);
  const children = keys.map((key) =>
    registry.renderNode({
      label: key,
      path: `${context.path}.${key}`,
      depth: context.depth + 1,
      value: context.value[key],
    })
  );

  node.setMeta({
    label: context.label,
    kind: "object",
    summary: summarizeStructuredValue(context.label, context.value),
    expandable: true,
    expanded: context.depth < 1,
  });
  node.setChildren(children);
  return node;
}

const defaultRegistry = new NdSceneTypeRegistry();
defaultRegistry.register("array", Array.isArray, renderArrayNode);
defaultRegistry.register(
  "object",
  (value) => value !== null && typeof value === "object" && !Array.isArray(value),
  renderObjectNode
);
defaultRegistry.register("leaf", () => true, renderLeafNode);

NdSceneTreeElement.defaultRegistry = defaultRegistry;

export function ensureNdSceneElements() {
  if (!customElements.get("ndscene-node")) {
    customElements.define("ndscene-node", NdSceneNodeElement);
  }
  if (!customElements.get("ndscene-tree")) {
    customElements.define("ndscene-tree", NdSceneTreeElement);
  }
}

export { NdSceneTreeElement, NdSceneNodeElement, NdSceneTypeRegistry, defaultRegistry };
