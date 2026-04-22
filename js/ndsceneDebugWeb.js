const DEBUG_STYLE_ID = "ndscene-debug-web-style";
export class NDSceneDebugInspector {
    element;
    summaryLabelElement;
    hintElement;
    toolbarElement;
    metaElement;
    contentElement;
    treeButton;
    listButton;
    runtime = null;
    viewMode;
    constructor(options = {}) {
        ensureDebugWebStyles();
        this.viewMode = options.initialViewMode ?? "tree";
        this.element = document.createElement("details");
        this.element.className = "ndscene-debug";
        this.element.open = options.initiallyOpen ?? false;
        const summaryElement = document.createElement("summary");
        summaryElement.className = "ndscene-debug__summary";
        this.summaryLabelElement = document.createElement("span");
        this.summaryLabelElement.className = "ndscene-debug__title";
        this.summaryLabelElement.textContent = options.title ?? "Debug Inspector";
        summaryElement.appendChild(this.summaryLabelElement);
        this.hintElement = document.createElement("span");
        this.hintElement.className = "ndscene-debug__hint";
        this.hintElement.textContent = "tree view";
        summaryElement.appendChild(this.hintElement);
        this.element.appendChild(summaryElement);
        const bodyElement = document.createElement("div");
        bodyElement.className = "ndscene-debug__body";
        this.element.appendChild(bodyElement);
        this.toolbarElement = document.createElement("div");
        this.toolbarElement.className = "ndscene-debug__toolbar";
        bodyElement.appendChild(this.toolbarElement);
        this.treeButton = this.createModeButton("Tree", "tree");
        this.listButton = this.createModeButton("List", "list");
        this.toolbarElement.append(this.treeButton, this.listButton);
        this.metaElement = document.createElement("div");
        this.metaElement.className = "ndscene-debug__meta";
        bodyElement.appendChild(this.metaElement);
        this.contentElement = document.createElement("div");
        this.contentElement.className = "ndscene-debug__content";
        bodyElement.appendChild(this.contentElement);
        this.render();
    }
    setRuntime(runtime) {
        this.runtime = runtime;
        this.render();
    }
    setViewMode(viewMode) {
        if (this.viewMode === viewMode) {
            return;
        }
        this.viewMode = viewMode;
        this.render();
    }
    createModeButton(label, viewMode) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ndscene-debug__mode";
        button.textContent = label;
        button.addEventListener("click", () => this.setViewMode(viewMode));
        return button;
    }
    render() {
        this.treeButton.setAttribute("aria-pressed", String(this.viewMode === "tree"));
        this.listButton.setAttribute("aria-pressed", String(this.viewMode === "list"));
        this.metaElement.replaceChildren();
        this.contentElement.replaceChildren();
        if (!this.runtime) {
            this.summaryLabelElement.textContent = "Debug Inspector";
            this.hintElement.textContent = "tree view";
            this.metaElement.appendChild(createChip("no runtime"));
            this.contentElement.appendChild(createEmptyState("No runtime loaded yet."));
            return;
        }
        const activeCommit = this.runtime.activeCommit;
        const activeScene = activeCommit.scene;
        this.summaryLabelElement.textContent = `Debug Inspector: ${activeCommit.commitId}`;
        this.hintElement.textContent = this.viewMode === "tree" ? "tree view" : "list view";
        this.metaElement.append(createChip(`${this.runtime.getOrderedCommits().length} commit${this.runtime.getOrderedCommits().length === 1 ? "" : "s"}`), createChip(`${activeScene.nodesByName.size} node${activeScene.nodesByName.size === 1 ? "" : "s"}`), createChip(`${activeScene.buffersByPath.size} buffer${activeScene.buffersByPath.size === 1 ? "" : "s"}`), createChip(`mode: ${this.viewMode}`));
        this.contentElement.appendChild(this.renderRuntimeSections(this.runtime));
    }
    renderRuntimeSections(runtime) {
        const container = document.createElement("div");
        container.className = "ndscene-debug__sections";
        const runtimeSection = createSection("Active Commit");
        runtimeSection.appendChild(createKvList([
            ["commitId", runtime.activeCommit.commitId],
            ["previousId", runtime.activeCommit.commitPreviousId || undefined],
            ["inputId", runtime.activeCommit.commitInputId || undefined],
            ["createdAt", runtime.activeCommit.createdAt || "unknown"],
            ["createdBy", runtime.activeCommit.createdByModelId || "unknown"],
            ["commitOrder", runtime.commitOrder.join(" -> ") || undefined],
        ]));
        container.appendChild(runtimeSection);
        if (this.viewMode === "tree") {
            container.appendChild(this.renderTreeView(runtime.activeScene));
        }
        else {
            container.appendChild(this.renderListView(runtime.activeScene));
        }
        container.appendChild(this.renderBufferSection(runtime.activeScene));
        return container;
    }
    renderTreeView(scene) {
        const section = createSection("Scene Tree");
        const rootNames = scene.rootNodeNames.length > 0
            ? [...scene.rootNodeNames]
            : [...scene.nodesByName.keys()].sort(compareText);
        if (rootNames.length === 0) {
            section.appendChild(createEmptyState("Scene has no nodes."));
            return section;
        }
        const treeContainer = document.createElement("div");
        treeContainer.className = "ndscene-debug__tree";
        for (const rootName of rootNames) {
            const rootNode = scene.getNode(rootName);
            if (!rootNode) {
                continue;
            }
            treeContainer.appendChild(this.renderNodeTree(rootNode, scene, new Set()));
        }
        const orphans = [...scene.nodesByName.values()]
            .filter((node) => !rootNames.includes(node.name))
            .filter((node) => node.parentName === "" || !scene.getNode(node.parentName))
            .sort((left, right) => compareText(left.name, right.name));
        if (orphans.length > 0) {
            const orphanSection = document.createElement("details");
            orphanSection.className = "ndscene-debug__entry";
            const orphanSummary = document.createElement("summary");
            orphanSummary.textContent = `Unrooted nodes (${orphans.length})`;
            orphanSection.appendChild(orphanSummary);
            const orphanContainer = document.createElement("div");
            orphanContainer.className = "ndscene-debug__children";
            for (const orphan of orphans) {
                orphanContainer.appendChild(this.renderNodeTree(orphan, scene, new Set(rootNames)));
            }
            orphanSection.appendChild(orphanContainer);
            treeContainer.appendChild(orphanSection);
        }
        section.appendChild(treeContainer);
        return section;
    }
    renderNodeTree(node, scene, ancestry) {
        const entry = document.createElement("details");
        entry.className = "ndscene-debug__entry";
        entry.open = ancestry.size < 2;
        const summary = document.createElement("summary");
        summary.append(createInlineLabel(node.name, "node"), createInlineMeta(`children ${node.edge.childNodeNames.length}`), ...optionalInlineMetas([
            node.edge.content ? describeTensor(node.edge.content) : undefined,
        ]));
        entry.appendChild(summary);
        const body = document.createElement("div");
        body.className = "ndscene-debug__children";
        body.appendChild(createKvList([
            ["name", node.name],
            ["commitId", node.commitId],
            ["parentName", node.parentName || undefined],
            ["pose", describeOptionalTensor(node.edge.pose)],
            ["unpose", describeOptionalTensor(node.edge.unpose)],
            ["content", describeOptionalTensor(node.edge.content)],
            ["edgePacket", node.edgePacket ? `${node.edgePacket.length} bytes` : undefined],
        ]));
        const nextAncestry = new Set(ancestry);
        nextAncestry.add(node.name);
        if (node.edge.childNodeNames.length > 0) {
            const childrenLabel = document.createElement("div");
            childrenLabel.className = "ndscene-debug__subhead";
            childrenLabel.textContent = "Children";
            body.appendChild(childrenLabel);
            for (const childName of node.edge.childNodeNames) {
                if (nextAncestry.has(childName)) {
                    body.appendChild(createWarning(`Cycle to ${childName}`));
                    continue;
                }
                const childNode = scene.getNode(childName);
                if (!childNode) {
                    body.appendChild(createWarning(`Missing child node ${childName}`));
                    continue;
                }
                body.appendChild(this.renderNodeTree(childNode, scene, nextAncestry));
            }
        }
        entry.appendChild(body);
        return entry;
    }
    renderListView(scene) {
        const section = createSection("Scene Nodes");
        const nodes = [...scene.nodesByName.values()].sort((left, right) => compareText(left.name, right.name));
        if (nodes.length === 0) {
            section.appendChild(createEmptyState("Scene has no nodes."));
            return section;
        }
        const listContainer = document.createElement("div");
        listContainer.className = "ndscene-debug__list";
        for (const node of nodes) {
            const entry = document.createElement("details");
            entry.className = "ndscene-debug__entry";
            const summary = document.createElement("summary");
            summary.append(createInlineLabel(node.name, "node"), createInlineMeta(node.parentName ? `parent ${node.parentName}` : "root"), createInlineMeta(`children ${node.edge.childNodeNames.length}`));
            entry.appendChild(summary);
            const body = document.createElement("div");
            body.className = "ndscene-debug__children";
            body.appendChild(createKvList([
                ["name", node.name],
                ["commitId", node.commitId],
                ["parentName", node.parentName || undefined],
                ["childNames", node.edge.childNodeNames.join(", ") || undefined],
                ["pose", describeOptionalTensor(node.edge.pose)],
                ["unpose", describeOptionalTensor(node.edge.unpose)],
                ["content", describeOptionalTensor(node.edge.content)],
            ]));
            entry.appendChild(body);
            listContainer.appendChild(entry);
        }
        section.appendChild(listContainer);
        return section;
    }
    renderBufferSection(scene) {
        const section = createSection("Scene Buffers");
        const buffers = [...scene.buffersByPath.values()].sort((left, right) => compareText(left.path, right.path));
        if (buffers.length === 0) {
            section.appendChild(createEmptyState("Scene has no buffers."));
            return section;
        }
        const listContainer = document.createElement("div");
        listContainer.className = "ndscene-debug__list";
        for (const buffer of buffers) {
            const entry = document.createElement("details");
            entry.className = "ndscene-debug__entry";
            const summary = document.createElement("summary");
            summary.append(createInlineLabel(buffer.path, "buffer"), createInlineMeta(buffer.format || "format unknown"), createInlineMeta(buffer.dataEncoded ? `${buffer.dataEncoded.length} bytes` : "external"));
            entry.appendChild(summary);
            const body = document.createElement("div");
            body.className = "ndscene-debug__children";
            body.appendChild(createKvList([
                ["path", buffer.path],
                ["commitId", buffer.commitId],
                ["format", buffer.format || undefined],
                ["dataEncoded", buffer.dataEncoded ? `${buffer.dataEncoded.length} bytes` : undefined],
                ["dataDecoded", describeOptionalTensor(buffer.dataDecoded)],
            ]));
            entry.appendChild(body);
            listContainer.appendChild(entry);
        }
        section.appendChild(listContainer);
        return section;
    }
}
export function createNDSceneDebugInspector(options = {}) {
    return new NDSceneDebugInspector(options);
}
function ensureDebugWebStyles() {
    if (document.getElementById(DEBUG_STYLE_ID)) {
        return;
    }
    const styleElement = document.createElement("style");
    styleElement.id = DEBUG_STYLE_ID;
    styleElement.textContent = `
    .ndscene-debug {
      width: 100%;
      max-width: 100%;
      max-height: 100%;
      color: #2f271c;
    }

    .ndscene-debug[open] {
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .ndscene-debug__summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      cursor: pointer;
      list-style: none;
      font-weight: 700;
    }

    .ndscene-debug__summary::-webkit-details-marker {
      display: none;
    }

    .ndscene-debug__title {
      min-width: 0;
      font-size: 1rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .ndscene-debug__hint {
      color: #665a49;
      font-size: 0.84rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .ndscene-debug__body {
      margin-top: 0.8rem;
      display: grid;
      min-height: 0;
      flex: 1 1 auto;
      gap: 0.75rem;
    }

    .ndscene-debug__toolbar {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .ndscene-debug__mode {
      padding: 0.45rem 0.7rem;
      border-radius: 999px;
      border: 1px solid rgba(78, 64, 45, 0.15);
      background: rgba(255, 255, 255, 0.8);
      color: #4f412e;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    .ndscene-debug__mode[aria-pressed="true"] {
      background: #8f5f24;
      color: white;
      border-color: #8f5f24;
    }

    .ndscene-debug__meta {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }

    .ndscene-debug__chip {
      padding: 0.25rem 0.55rem;
      border-radius: 999px;
      background: rgba(143, 95, 36, 0.12);
      color: #4f412e;
      font-size: 0.82rem;
      font-weight: 600;
    }

    .ndscene-debug__content {
      min-height: 0;
      max-height: none;
      flex: 1 1 auto;
      overflow: auto;
      padding-right: 0.2rem;
    }

    .ndscene-debug__sections {
      display: grid;
      gap: 0.75rem;
    }

    .ndscene-debug__section {
      padding: 0.7rem 0.75rem;
      border-radius: 0.9rem;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(78, 64, 45, 0.08);
    }

    .ndscene-debug__section-title {
      margin: 0 0 0.55rem;
      font-size: 0.88rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #665a49;
    }

    .ndscene-debug__tree,
    .ndscene-debug__list {
      display: grid;
      gap: 0.45rem;
    }

    .ndscene-debug__entry {
      border: 1px solid rgba(78, 64, 45, 0.09);
      border-radius: 0.8rem;
      background: rgba(252, 249, 244, 0.88);
      overflow: hidden;
    }

    .ndscene-debug__entry > summary {
      padding: 0.55rem 0.7rem;
      display: flex;
      gap: 0.45rem;
      flex-wrap: wrap;
      align-items: center;
      cursor: pointer;
      list-style: none;
    }

    .ndscene-debug__entry > summary::-webkit-details-marker {
      display: none;
    }

    .ndscene-debug__children {
      padding: 0 0.7rem 0.7rem;
      display: grid;
      gap: 0.55rem;
    }

    .ndscene-debug__badge {
      padding: 0.18rem 0.45rem;
      border-radius: 999px;
      background: rgba(143, 95, 36, 0.13);
      color: #70491c;
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .ndscene-debug__inline-meta {
      color: #665a49;
      font-size: 0.84rem;
    }

    .ndscene-debug__kv {
      display: grid;
      grid-template-columns: minmax(6.5rem, auto) minmax(0, 1fr);
      gap: 0.35rem 0.6rem;
      font-size: 0.86rem;
      line-height: 1.4;
    }

    .ndscene-debug__key {
      color: #665a49;
      font-weight: 700;
    }

    .ndscene-debug__value {
      color: #2f271c;
      overflow-wrap: anywhere;
    }

    .ndscene-debug__subhead {
      margin-top: 0.1rem;
      color: #665a49;
      font-size: 0.8rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .ndscene-debug__empty,
    .ndscene-debug__warning {
      padding: 0.55rem 0.65rem;
      border-radius: 0.75rem;
      font-size: 0.84rem;
    }

    .ndscene-debug__empty {
      background: rgba(255, 255, 255, 0.55);
      color: #665a49;
    }

    .ndscene-debug__warning {
      background: rgba(150, 56, 34, 0.08);
      color: #7c251d;
    }
  `;
    document.head.appendChild(styleElement);
}
function createSection(title) {
    const section = document.createElement("div");
    section.className = "ndscene-debug__section";
    const titleElement = document.createElement("h3");
    titleElement.className = "ndscene-debug__section-title";
    titleElement.textContent = title;
    section.appendChild(titleElement);
    return section;
}
function createChip(label) {
    const chip = document.createElement("span");
    chip.className = "ndscene-debug__chip";
    chip.textContent = label;
    return chip;
}
function createKvList(entries) {
    const container = document.createElement("div");
    container.className = "ndscene-debug__kv";
    for (const [key, value] of entries) {
        if (value === null || value === undefined || value === "") {
            continue;
        }
        const keyElement = document.createElement("div");
        keyElement.className = "ndscene-debug__key";
        keyElement.textContent = key;
        const valueElement = document.createElement("div");
        valueElement.className = "ndscene-debug__value";
        valueElement.textContent = value;
        container.append(keyElement, valueElement);
    }
    return container;
}
function createInlineLabel(text, badge) {
    const fragment = document.createDocumentFragment();
    const badgeElement = document.createElement("span");
    badgeElement.className = "ndscene-debug__badge";
    badgeElement.textContent = badge;
    fragment.appendChild(badgeElement);
    const textElement = document.createElement("span");
    textElement.textContent = text;
    fragment.appendChild(textElement);
    return fragment;
}
function createInlineMeta(text) {
    const metaElement = document.createElement("span");
    metaElement.className = "ndscene-debug__inline-meta";
    metaElement.textContent = text;
    return metaElement;
}
function optionalInlineMetas(values) {
    return values
        .filter((value) => value !== null && value !== undefined && value !== "")
        .map((value) => createInlineMeta(value));
}
function createEmptyState(text) {
    const element = document.createElement("div");
    element.className = "ndscene-debug__empty";
    element.textContent = text;
    return element;
}
function createWarning(text) {
    const element = document.createElement("div");
    element.className = "ndscene-debug__warning";
    element.textContent = text;
    return element;
}
function compareText(left, right) {
    return left.localeCompare(right);
}
function describeTensor(tensor) {
    if (!tensor) {
        return "none";
    }
    const shapeText = tensor.shapeSizes().length > 0
        ? `[${tensor.shapeSizes().join(", ")}]`
        : "scalar";
    if (tensor.bufferPath()) {
        return `${tensor.dtype} ${shapeText} path=${tensor.bufferPath()}`;
    }
    if (tensor.scalarString()) {
        return `${tensor.dtype} ${shapeText} "${truncateText(tensor.scalarString() ?? "", 36)}"`;
    }
    const bytes = tensor.bytes();
    if (bytes) {
        return `${tensor.dtype} ${shapeText} ${bytes.length} byte${bytes.length === 1 ? "" : "s"}`;
    }
    const numbers = tensor.denseNumbers();
    if (numbers.length > 0) {
        const preview = numbers.slice(0, 4).map((value) => formatNumber(value)).join(", ");
        const suffix = numbers.length > 4 ? ", ..." : "";
        return `${tensor.dtype} ${shapeText} [${preview}${suffix}]`;
    }
    if (tensor.shape.length > 0) {
        return `${tensor.dtype} ${shapeText} ${tensor.shape.length} child${tensor.shape.length === 1 ? "" : "ren"}`;
    }
    return `${tensor.dtype} ${shapeText}`;
}
function describeOptionalTensor(tensor) {
    return tensor ? describeTensor(tensor) : undefined;
}
function formatNumber(value) {
    if (!Number.isFinite(value)) {
        return String(value);
    }
    const rounded = Math.round(value * 1000) / 1000;
    return String(rounded);
}
function truncateText(value, maxLength) {
    if (value.length <= maxLength) {
        return value;
    }
    return `${value.slice(0, Math.max(0, maxLength - 3))}...`;
}
