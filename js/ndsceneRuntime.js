import { runtimeFromFlatbufferBuffer, runtimeFromLegacyJson, runtimeFromLegacyJsonSequence, runtimeToFlatbufferBuffer, } from "./ndsceneLegacyJson.js";
export class NDTensorShapeEntryRuntime {
    size;
    key;
    tensor;
    constructor(init = {}) {
        this.size = init.size ?? 0;
        this.key = init.key ?? "";
        this.tensor = init.tensor;
    }
}
export class NDTensorRuntime {
    shape;
    dtype;
    dataString;
    dataNumbers;
    dataUbytes;
    dataPath;
    constructor(init = {}) {
        this.shape = init.shape ? [...init.shape] : [];
        this.dtype = init.dtype ?? "null";
        this.dataString = init.dataString;
        this.dataNumbers = init.dataNumbers ? [...init.dataNumbers] : undefined;
        this.dataUbytes = init.dataUbytes ? new Uint8Array(init.dataUbytes) : undefined;
        this.dataPath = init.dataPath;
    }
    child(key) {
        return this.shape.find((entry) => entry.key === key)?.tensor;
    }
    childAt(index) {
        return this.child(String(index));
    }
    hasDenseNumbers() {
        return Boolean(this.dataNumbers && this.dataNumbers.length > 0);
    }
    denseNumbers() {
        return this.dataNumbers ? [...this.dataNumbers] : [];
    }
    scalarNumber() {
        return this.dataNumbers && this.dataNumbers.length > 0 ? this.dataNumbers[0] : undefined;
    }
    scalarBoolean() {
        if (this.dtype !== "bool") {
            return undefined;
        }
        const value = this.scalarNumber();
        return value === undefined ? undefined : value !== 0;
    }
    scalarString() {
        return this.dataString;
    }
    bytes() {
        return this.dataUbytes ? new Uint8Array(this.dataUbytes) : undefined;
    }
    bufferPath() {
        return this.dataPath;
    }
    shapeSizes() {
        return this.shape.map((entry) => entry.size);
    }
}
export class NDSceneEdgeRuntime {
    pose;
    unpose;
    content;
    childNodeNames;
    constructor(init = {}) {
        this.pose = init.pose;
        this.unpose = init.unpose;
        this.content = init.content;
        this.childNodeNames = init.childNodeNames ? [...init.childNodeNames] : [];
    }
}
export class NDSceneNodeRuntime {
    name;
    commitId;
    parentName;
    edge;
    edgePacket;
    constructor(init) {
        this.name = init.name;
        this.commitId = init.commitId;
        this.parentName = init.parentName ?? "";
        this.edge = init.edge ?? new NDSceneEdgeRuntime();
        this.edgePacket = init.edgePacket ? new Uint8Array(init.edgePacket) : undefined;
    }
}
export class NDSceneBufferRuntime {
    path;
    commitId;
    format;
    dataEncoded;
    dataDecoded;
    constructor(init) {
        this.path = init.path;
        this.commitId = init.commitId;
        this.format = init.format ?? "";
        this.dataEncoded = init.dataEncoded ? new Uint8Array(init.dataEncoded) : undefined;
        this.dataDecoded = init.dataDecoded;
    }
}
export class NDSceneGraphRuntime {
    nodesByName = new Map();
    buffersByPath = new Map();
    rootNodeNames = [];
    addNode(node) {
        if (this.nodesByName.has(node.name)) {
            throw new Error(`Duplicate scene node name "${node.name}".`);
        }
        this.nodesByName.set(node.name, node);
    }
    addBuffer(buffer) {
        this.buffersByPath.set(buffer.path, buffer);
    }
    getNode(name) {
        return this.nodesByName.get(name);
    }
    getBuffer(path) {
        return this.buffersByPath.get(path);
    }
    getRootNodes() {
        return this.rootNodeNames
            .map((name) => this.getNode(name))
            .filter((node) => Boolean(node));
    }
    getChildNodes(node) {
        return node.edge.childNodeNames
            .map((name) => this.getNode(name))
            .filter((child) => Boolean(child));
    }
    rebuildRoots() {
        const referencedNames = new Set();
        for (const node of this.nodesByName.values()) {
            for (const childName of node.edge.childNodeNames) {
                referencedNames.add(childName);
            }
        }
        this.rootNodeNames = [...this.nodesByName.values()]
            .filter((node) => node.parentName === "" && !referencedNames.has(node.name))
            .map((node) => node.name);
        if (this.rootNodeNames.length === 0) {
            this.rootNodeNames = [...this.nodesByName.values()]
                .filter((node) => !referencedNames.has(node.name))
                .map((node) => node.name);
        }
    }
}
export class NDSceneCommitRuntime {
    commitId;
    commitPreviousId;
    commitInputId;
    createdAt;
    createdByModelId;
    scene;
    packet;
    constructor(init) {
        this.commitId = init.commitId;
        this.commitPreviousId = init.commitPreviousId ?? "";
        this.commitInputId = init.commitInputId ?? init.commitId;
        this.createdAt = init.createdAt ?? new Date().toISOString();
        this.createdByModelId = init.createdByModelId ?? "ndscene-runtime";
        this.scene = init.scene ?? new NDSceneGraphRuntime();
        this.packet = init.packet ? new Uint8Array(init.packet) : undefined;
    }
}
export class NDSceneRuntime {
    commitsById = new Map();
    commitOrder = [];
    activeCommitId = null;
    addCommit(commit, setActive = false) {
        if (!this.commitsById.has(commit.commitId)) {
            this.commitOrder.push(commit.commitId);
        }
        this.commitsById.set(commit.commitId, commit);
        if (setActive || this.activeCommitId === null) {
            this.activeCommitId = commit.commitId;
        }
    }
    getCommit(commitId) {
        return this.commitsById.get(commitId);
    }
    getOrderedCommits() {
        return this.commitOrder
            .map((commitId) => this.getCommit(commitId))
            .filter((commit) => Boolean(commit));
    }
    setActiveCommit(commitId) {
        if (!this.commitsById.has(commitId)) {
            throw new Error(`Commit "${commitId}" is missing from the runtime.`);
        }
        this.activeCommitId = commitId;
    }
    get activeCommit() {
        if (!this.activeCommitId) {
            throw new Error("NDSceneRuntime has no active commit.");
        }
        const commit = this.commitsById.get(this.activeCommitId);
        if (!commit) {
            throw new Error(`Active commit "${this.activeCommitId}" is missing from the runtime.`);
        }
        return commit;
    }
    get activeScene() {
        return this.activeCommit.scene;
    }
    toFlatbufferBuffer() {
        return runtimeToFlatbufferBuffer(this);
    }
    static fromFlatbufferBuffer(buffer) {
        return runtimeFromFlatbufferBuffer(buffer);
    }
    static fromLegacyJsonSequence(commits) {
        return runtimeFromLegacyJsonSequence(commits);
    }
    static fromLegacyJson(sceneJson, options = {}) {
        return runtimeFromLegacyJson(sceneJson, options);
    }
}
