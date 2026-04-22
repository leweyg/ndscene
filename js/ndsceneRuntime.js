import { Builder, NDPacketBuffer, NDPacketRoot, NDPacketSceneCommit, NDPacketSceneEdge, NDPacketSceneNode, NDPacketSceneUpdate, NDPacketShapeEntry, NDPacketTensor, ndPacketBytes, ndPacketRootFromBuffer, } from "./ndsceneFlatbuffers.js";
const ROOT_PACKET_MAGIC = "NDSN";
const TIMELINE_MANIFEST_PATH = "__ndscene_runtime__/timeline_manifest.json";
const TIMELINE_MANIFEST_FORMAT = "application/vnd.ndscene.timeline+json";
const STRING_DECODER = new TextDecoder();
const STRING_ENCODER = new TextEncoder();
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
        return serializeRuntimeToFlatbuffer(this);
    }
    static fromFlatbufferBuffer(buffer) {
        const rootPacket = ndPacketRootFromBuffer(buffer);
        const commitPacket = rootPacket.commit();
        const scenePacket = commitPacket?.scene() ?? rootPacket.scene();
        if (!scenePacket) {
            throw new Error("FlatBuffer packet does not contain a scene or commit payload.");
        }
        const commitId = readPacketString(commitPacket?.commitId()) ?? "commit_from_buffer";
        const sceneNodes = [];
        const sceneBuffers = [];
        for (let nodeIndex = 0; nodeIndex < scenePacket.nodesLength(); nodeIndex += 1) {
            const packetNode = scenePacket.nodes(nodeIndex);
            if (!packetNode) {
                continue;
            }
            sceneNodes.push(packetSceneNodeToRuntime(packetNode, commitId, nodeIndex));
        }
        for (let bufferIndex = 0; bufferIndex < scenePacket.buffersLength(); bufferIndex += 1) {
            const packetBuffer = scenePacket.buffers(bufferIndex);
            if (!packetBuffer) {
                continue;
            }
            sceneBuffers.push(packetSceneBufferToRuntime(packetBuffer, commitId, bufferIndex));
        }
        const manifestBuffer = sceneBuffers.find((sceneBuffer) => sceneBuffer.path === TIMELINE_MANIFEST_PATH);
        if (manifestBuffer) {
            return runtimeFromTimelinePacket(commitPacket, sceneNodes, sceneBuffers, manifestBuffer);
        }
        const scene = new NDSceneGraphRuntime();
        for (const sceneNode of sceneNodes) {
            scene.addNode(sceneNode);
        }
        for (const sceneBuffer of sceneBuffers) {
            scene.addBuffer(sceneBuffer);
        }
        scene.rebuildRoots();
        const runtime = new NDSceneRuntime();
        runtime.addCommit(new NDSceneCommitRuntime({
            commitId,
            commitPreviousId: readPacketString(commitPacket?.commitPreviousId()) ?? "",
            commitInputId: readPacketString(commitPacket?.commitInputId()) ?? commitId,
            createdAt: readPacketString(commitPacket?.createdAt()) ?? "",
            createdByModelId: readPacketString(commitPacket?.createdByModelId()) ?? "ndscene-runtime",
            scene,
            packet: clonePacketBytes(commitPacket?.packetArray()),
        }), true);
        return runtime;
    }
    static fromLegacyJsonSequence(commits) {
        if (commits.length === 0) {
            throw new Error("Legacy JSON commit sequence is empty.");
        }
        const runtime = new NDSceneRuntime();
        let previousScene = null;
        let previousCommitId = "";
        for (let commitIndex = 0; commitIndex < commits.length; commitIndex += 1) {
            const commitInput = commits[commitIndex];
            const importedRuntime = NDSceneRuntime.fromLegacyJson(commitInput.sceneJson, {
                commitId: commitInput.commitId,
                commitPreviousId: commitInput.commitPreviousId ?? previousCommitId,
                commitInputId: commitInput.commitInputId ?? (previousCommitId || commitInput.commitId),
                createdAt: commitInput.createdAt,
                createdByModelId: commitInput.createdByModelId,
            });
            const importedCommit = importedRuntime.activeCommit;
            const sharedScene = shareSceneGraphSnapshot(previousScene, importedCommit.scene);
            runtime.addCommit(new NDSceneCommitRuntime({
                commitId: importedCommit.commitId,
                commitPreviousId: importedCommit.commitPreviousId,
                commitInputId: importedCommit.commitInputId,
                createdAt: importedCommit.createdAt,
                createdByModelId: importedCommit.createdByModelId,
                scene: sharedScene,
                packet: importedCommit.packet,
            }), commitIndex === commits.length - 1);
            previousScene = sharedScene;
            previousCommitId = importedCommit.commitId;
        }
        return runtime;
    }
    static fromLegacyJson(sceneJson, options = {}) {
        if (!sceneJson.root || typeof sceneJson.root !== "object") {
            throw new Error("Legacy JSON scene is missing a valid root object.");
        }
        const commitId = options.commitId ?? "commit_from_json";
        const scene = new NDSceneGraphRuntime();
        const usedNames = new Set();
        const sharedNameByDefinitionKey = new Map();
        const definitions = sceneJson.objects ?? {};
        const ensureBuffer = (path, format) => {
            const normalizedPath = path.trim();
            if (!normalizedPath) {
                throw new Error("Legacy buffer reference is missing a valid path.");
            }
            const existingBuffer = scene.getBuffer(normalizedPath);
            if (existingBuffer) {
                if (!existingBuffer.format && format) {
                    existingBuffer.format = format;
                }
                return existingBuffer;
            }
            const buffer = new NDSceneBufferRuntime({
                path: normalizedPath,
                commitId,
                format: format ?? inferBufferFormatFromPath(normalizedPath),
            });
            scene.addBuffer(buffer);
            return buffer;
        };
        const importContext = {
            registerBuffer: ensureBuffer,
        };
        const claimUniqueName = (baseName) => {
            const normalizedBase = baseName.trim() || "node";
            if (!usedNames.has(normalizedBase)) {
                usedNames.add(normalizedBase);
                return normalizedBase;
            }
            let suffix = 2;
            while (usedNames.has(`${normalizedBase}_${suffix}`)) {
                suffix += 1;
            }
            const claimedName = `${normalizedBase}_${suffix}`;
            usedNames.add(claimedName);
            return claimedName;
        };
        const importResolvedNode = (nodeJson, resolvedName, parentName) => {
            const runtimeNode = new NDSceneNodeRuntime({
                name: resolvedName,
                commitId,
                parentName,
                edge: new NDSceneEdgeRuntime({
                    pose: tensorRuntimeFromLegacyValue(nodeJson.pose, importContext),
                    unpose: tensorRuntimeFromLegacyValue(nodeJson.unpose, importContext),
                    content: tensorRuntimeFromLegacyValue(nodeJson.content, importContext),
                    childNodeNames: [],
                }),
            });
            scene.addNode(runtimeNode);
            const children = Array.isArray(nodeJson.children) ? nodeJson.children : [];
            const childNodeNames = [];
            for (let childIndex = 0; childIndex < children.length; childIndex += 1) {
                const child = children[childIndex];
                if (typeof child === "string") {
                    const definition = definitions[child];
                    if (!definition) {
                        throw new Error(`Legacy scene references missing object definition "${child}".`);
                    }
                    let sharedNodeName = sharedNameByDefinitionKey.get(child);
                    if (!sharedNodeName) {
                        const preferredName = typeof definition.name === "string" && definition.name ? definition.name : child;
                        sharedNodeName = claimUniqueName(preferredName);
                        sharedNameByDefinitionKey.set(child, sharedNodeName);
                        importResolvedNode(definition, sharedNodeName, "");
                    }
                    childNodeNames.push(sharedNodeName);
                    continue;
                }
                const inlineName = typeof child.name === "string" && child.name
                    ? claimUniqueName(child.name)
                    : claimUniqueName(`${resolvedName}_child_${childIndex}`);
                importResolvedNode(child, inlineName, resolvedName);
                childNodeNames.push(inlineName);
            }
            runtimeNode.edge.childNodeNames = childNodeNames;
            return resolvedName;
        };
        const rootPreferredName = typeof sceneJson.root.name === "string" && sceneJson.root.name
            ? sceneJson.root.name
            : "root";
        importResolvedNode(sceneJson.root, claimUniqueName(rootPreferredName), "");
        scene.rebuildRoots();
        const runtime = new NDSceneRuntime();
        runtime.addCommit(new NDSceneCommitRuntime({
            commitId,
            commitPreviousId: options.commitPreviousId ?? "",
            commitInputId: options.commitInputId ?? commitId,
            createdAt: options.createdAt ?? new Date().toISOString(),
            createdByModelId: options.createdByModelId ?? "legacy-json",
            scene,
        }), true);
        return runtime;
    }
}
function serializeCommitToFlatbuffer(commit) {
    return serializeCommitPacketToFlatbuffer(commit, [...commit.scene.nodesByName.values()], [...commit.scene.buffersByPath.values()]);
}
function serializeRuntimeToFlatbuffer(runtime) {
    const orderedCommits = runtime.getOrderedCommits();
    if (orderedCommits.length <= 1) {
        return serializeCommitToFlatbuffer(runtime.activeCommit);
    }
    const timelineNodes = [];
    const timelineBuffers = [];
    const manifestCommits = [];
    let previousScene = null;
    for (const commit of orderedCommits) {
        const diff = diffSceneGraphs(previousScene, commit.scene);
        manifestCommits.push({
            commitId: commit.commitId,
            commitPreviousId: commit.commitPreviousId,
            commitInputId: commit.commitInputId,
            createdAt: commit.createdAt,
            createdByModelId: commit.createdByModelId,
            changedNodeNames: diff.changedNodeNames,
            deletedNodeNames: diff.deletedNodeNames,
            changedBufferPaths: diff.changedBufferPaths,
            deletedBufferPaths: diff.deletedBufferPaths,
        });
        for (const nodeName of diff.changedNodeNames) {
            const changedNode = commit.scene.getNode(nodeName);
            if (!changedNode) {
                throw new Error(`Commit "${commit.commitId}" is missing changed node "${nodeName}".`);
            }
            timelineNodes.push(changedNode);
        }
        for (const bufferPath of diff.changedBufferPaths) {
            const changedBuffer = commit.scene.getBuffer(bufferPath);
            if (!changedBuffer) {
                throw new Error(`Commit "${commit.commitId}" is missing changed buffer "${bufferPath}".`);
            }
            timelineBuffers.push(changedBuffer);
        }
        previousScene = commit.scene;
    }
    const manifest = {
        version: 1,
        activeCommitId: runtime.activeCommit.commitId,
        commitOrder: orderedCommits.map((commit) => commit.commitId),
        commits: manifestCommits,
    };
    timelineBuffers.push(createTimelineManifestBuffer(runtime.activeCommit.commitId, manifest));
    return serializeCommitPacketToFlatbuffer(runtime.activeCommit, timelineNodes, timelineBuffers);
}
function serializeCommitPacketToFlatbuffer(commit, sceneNodes, sceneBuffers) {
    const builder = new Builder(1024);
    const sceneOffset = serializeSceneEntries(builder, sceneNodes, sceneBuffers);
    const commitIdOffset = createOptionalString(builder, commit.commitId);
    const commitPreviousIdOffset = createOptionalString(builder, commit.commitPreviousId);
    const commitInputIdOffset = createOptionalString(builder, commit.commitInputId);
    const createdAtOffset = createOptionalString(builder, commit.createdAt);
    const createdByModelIdOffset = createOptionalString(builder, commit.createdByModelId);
    const packetOffset = commit.packet ? NDPacketSceneCommit.createPacketVector(builder, commit.packet) : 0;
    NDPacketSceneCommit.startNDPacketSceneCommit(builder);
    if (commitIdOffset !== 0) {
        NDPacketSceneCommit.addCommitId(builder, commitIdOffset);
    }
    if (commitPreviousIdOffset !== 0) {
        NDPacketSceneCommit.addCommitPreviousId(builder, commitPreviousIdOffset);
    }
    if (commitInputIdOffset !== 0) {
        NDPacketSceneCommit.addCommitInputId(builder, commitInputIdOffset);
    }
    if (createdAtOffset !== 0) {
        NDPacketSceneCommit.addCreatedAt(builder, createdAtOffset);
    }
    if (createdByModelIdOffset !== 0) {
        NDPacketSceneCommit.addCreatedByModelId(builder, createdByModelIdOffset);
    }
    NDPacketSceneCommit.addScene(builder, sceneOffset);
    if (packetOffset !== 0) {
        NDPacketSceneCommit.addPacket(builder, packetOffset);
    }
    const commitOffset = NDPacketSceneCommit.endNDPacketSceneCommit(builder);
    const packetMagicOffset = builder.createString(ROOT_PACKET_MAGIC);
    NDPacketRoot.startNDPacketRoot(builder);
    NDPacketRoot.addPacketMagic(builder, packetMagicOffset);
    NDPacketRoot.addCommit(builder, commitOffset);
    const rootOffset = NDPacketRoot.endNDPacketRoot(builder);
    NDPacketRoot.finishNDPacketRootBuffer(builder, rootOffset);
    return new Uint8Array(builder.asUint8Array());
}
function serializeSceneGraph(builder, scene) {
    return serializeSceneEntries(builder, [...scene.nodesByName.values()], [...scene.buffersByPath.values()]);
}
function serializeSceneEntries(builder, nodes, buffers) {
    const nodeOffsets = nodes.map((node) => serializeSceneNode(builder, node));
    const bufferOffsets = buffers.map((buffer) => serializeSceneBuffer(builder, buffer));
    const nodesVectorOffset = nodeOffsets.length > 0
        ? NDPacketSceneUpdate.createNodesVector(builder, nodeOffsets)
        : 0;
    const buffersVectorOffset = bufferOffsets.length > 0
        ? NDPacketSceneUpdate.createBuffersVector(builder, bufferOffsets)
        : 0;
    NDPacketSceneUpdate.startNDPacketSceneUpdate(builder);
    if (nodesVectorOffset !== 0) {
        NDPacketSceneUpdate.addNodes(builder, nodesVectorOffset);
    }
    if (buffersVectorOffset !== 0) {
        NDPacketSceneUpdate.addBuffers(builder, buffersVectorOffset);
    }
    return NDPacketSceneUpdate.endNDPacketSceneUpdate(builder);
}
function serializeSceneNode(builder, node) {
    const edgeOffset = serializeSceneEdge(builder, node.edge);
    const nameOffset = createOptionalString(builder, node.name);
    const commitIdOffset = createOptionalString(builder, node.commitId);
    const parentNameOffset = createOptionalString(builder, node.parentName);
    const edgePacketOffset = node.edgePacket ? NDPacketSceneNode.createEdgePacketVector(builder, node.edgePacket) : 0;
    NDPacketSceneNode.startNDPacketSceneNode(builder);
    if (nameOffset !== 0) {
        NDPacketSceneNode.addName(builder, nameOffset);
    }
    if (commitIdOffset !== 0) {
        NDPacketSceneNode.addCommitId(builder, commitIdOffset);
    }
    if (parentNameOffset !== 0) {
        NDPacketSceneNode.addParentName(builder, parentNameOffset);
    }
    NDPacketSceneNode.addEdgeScene(builder, edgeOffset);
    if (edgePacketOffset !== 0) {
        NDPacketSceneNode.addEdgePacket(builder, edgePacketOffset);
    }
    return NDPacketSceneNode.endNDPacketSceneNode(builder);
}
function serializeSceneEdge(builder, edge) {
    const poseOffset = edge.pose !== undefined ? serializeTensorRuntime(builder, edge.pose) : 0;
    const unposeOffset = edge.unpose !== undefined ? serializeTensorRuntime(builder, edge.unpose) : 0;
    const contentOffset = edge.content !== undefined ? serializeTensorRuntime(builder, edge.content) : 0;
    const childOffsets = edge.childNodeNames.map((name) => builder.createString(name));
    const childVectorOffset = childOffsets.length > 0
        ? NDPacketSceneEdge.createChildNodesVector(builder, childOffsets)
        : 0;
    NDPacketSceneEdge.startNDPacketSceneEdge(builder);
    if (poseOffset !== 0) {
        NDPacketSceneEdge.addPose(builder, poseOffset);
    }
    if (unposeOffset !== 0) {
        NDPacketSceneEdge.addUnpose(builder, unposeOffset);
    }
    if (contentOffset !== 0) {
        NDPacketSceneEdge.addContent(builder, contentOffset);
    }
    if (childVectorOffset !== 0) {
        NDPacketSceneEdge.addChildNodes(builder, childVectorOffset);
    }
    return NDPacketSceneEdge.endNDPacketSceneEdge(builder);
}
function serializeSceneBuffer(builder, buffer) {
    const pathOffset = createOptionalString(builder, buffer.path);
    const commitIdOffset = createOptionalString(builder, buffer.commitId);
    const formatOffset = createOptionalString(builder, buffer.format);
    const encodedOffset = buffer.dataEncoded ? NDPacketBuffer.createDataEncodedVector(builder, buffer.dataEncoded) : 0;
    const decodedOffset = buffer.dataDecoded !== undefined ? serializeTensorRuntime(builder, buffer.dataDecoded) : 0;
    NDPacketBuffer.startNDPacketBuffer(builder);
    if (pathOffset !== 0) {
        NDPacketBuffer.addPath(builder, pathOffset);
    }
    if (commitIdOffset !== 0) {
        NDPacketBuffer.addCommitId(builder, commitIdOffset);
    }
    if (formatOffset !== 0) {
        NDPacketBuffer.addFormat(builder, formatOffset);
    }
    if (encodedOffset !== 0) {
        NDPacketBuffer.addDataEncoded(builder, encodedOffset);
    }
    if (decodedOffset !== 0) {
        NDPacketBuffer.addDataDecoded(builder, decodedOffset);
    }
    return NDPacketBuffer.endNDPacketBuffer(builder);
}
function serializeTensorRuntime(builder, tensor) {
    const shapeOffset = tensor.shape.length > 0 ? serializeTensorShapeEntries(builder, tensor.shape) : 0;
    const dtypeOffset = builder.createString(tensor.dtype);
    const dataStringOffset = tensor.dataString ? builder.createString(tensor.dataString) : 0;
    const dataNumbersOffset = tensor.dataNumbers && tensor.dataNumbers.length > 0
        ? NDPacketTensor.createDataNumbersVector(builder, tensor.dataNumbers)
        : 0;
    const dataUbytesOffset = tensor.dataUbytes && tensor.dataUbytes.length > 0
        ? NDPacketTensor.createDataUbytesVector(builder, tensor.dataUbytes)
        : 0;
    const dataPathOffset = tensor.dataPath ? builder.createString(tensor.dataPath) : 0;
    NDPacketTensor.startNDPacketTensor(builder);
    if (shapeOffset !== 0) {
        NDPacketTensor.addShape(builder, shapeOffset);
    }
    NDPacketTensor.addDtype(builder, dtypeOffset);
    if (dataStringOffset !== 0) {
        NDPacketTensor.addDataString(builder, dataStringOffset);
    }
    if (dataNumbersOffset !== 0) {
        NDPacketTensor.addDataNumbers(builder, dataNumbersOffset);
    }
    if (dataUbytesOffset !== 0) {
        NDPacketTensor.addDataUbytes(builder, dataUbytesOffset);
    }
    if (dataPathOffset !== 0) {
        NDPacketTensor.addDataPath(builder, dataPathOffset);
    }
    return NDPacketTensor.endNDPacketTensor(builder);
}
function serializeTensorShapeEntries(builder, entries) {
    const entryOffsets = entries
        .filter((entry) => entry.tensor !== undefined || entry.key !== "" || entry.size !== 0)
        .map((entry) => {
        const tensorOffset = entry.tensor ? serializeTensorRuntime(builder, entry.tensor) : 0;
        const keyOffset = entry.key ? builder.createString(entry.key) : 0;
        NDPacketShapeEntry.startNDPacketShapeEntry(builder);
        NDPacketShapeEntry.addSize(builder, entry.size);
        if (keyOffset !== 0) {
            NDPacketShapeEntry.addKey(builder, keyOffset);
        }
        if (tensorOffset !== 0) {
            NDPacketShapeEntry.addTensor(builder, tensorOffset);
        }
        return NDPacketShapeEntry.endNDPacketShapeEntry(builder);
    });
    return entryOffsets.length > 0 ? NDPacketTensor.createShapeVector(builder, entryOffsets) : 0;
}
function tensorRuntimeFromPacket(packetTensor) {
    if (!packetTensor) {
        return undefined;
    }
    const shapeEntries = [];
    for (let shapeIndex = 0; shapeIndex < packetTensor.shapeLength(); shapeIndex += 1) {
        const shapeEntry = packetTensor.shape(shapeIndex);
        if (!shapeEntry) {
            continue;
        }
        shapeEntries.push(new NDTensorShapeEntryRuntime({
            size: shapeEntry.size(),
            key: readPacketString(shapeEntry.key()) ?? "",
            tensor: tensorRuntimeFromPacket(shapeEntry.tensor()),
        }));
    }
    return new NDTensorRuntime({
        shape: shapeEntries,
        dtype: readPacketString(packetTensor.dtype()) ?? "null",
        dataString: readPacketString(packetTensor.dataString()) ?? undefined,
        dataNumbers: packetTensor.dataNumbersArray() ? Array.from(packetTensor.dataNumbersArray()) : undefined,
        dataUbytes: clonePacketBytes(packetTensor.dataUbytesArray()),
        dataPath: readPacketString(packetTensor.dataPath()) ?? undefined,
    });
}
function tensorRuntimeFromLegacyValue(value, importContext) {
    if (value === undefined) {
        return undefined;
    }
    if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
        return tensorRuntimeFromScalar(value);
    }
    if (value instanceof Uint8Array) {
        return new NDTensorRuntime({
            dtype: "bytes",
            dataUbytes: new Uint8Array(value),
        });
    }
    if (value instanceof ArrayBuffer) {
        return new NDTensorRuntime({
            dtype: "bytes",
            dataUbytes: new Uint8Array(value),
        });
    }
    if (ArrayBuffer.isView(value)) {
        const viewBytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
        return new NDTensorRuntime({
            dtype: "bytes",
            dataUbytes: new Uint8Array(viewBytes),
        });
    }
    if (Array.isArray(value)) {
        return tensorRuntimeFromLegacyArray(value, importContext);
    }
    if (value && typeof value === "object") {
        const record = value;
        if (isLegacyBufferReferenceRecord(record)) {
            return tensorRuntimeFromLegacyBufferReference(record, importContext);
        }
        if (isLegacyTensorRecord(record)) {
            return tensorRuntimeFromLegacyTensorRecord(record, importContext);
        }
        const numericArray = numericRecordToArray(record);
        if (numericArray) {
            return tensorRuntimeFromLegacyArray(numericArray, importContext);
        }
        const entries = Object.entries(record)
            .map(([entryKey, entryValue]) => {
            const tensor = tensorRuntimeFromLegacyValue(entryValue, importContext);
            if (!tensor) {
                return null;
            }
            return new NDTensorShapeEntryRuntime({
                key: entryKey,
                size: estimateTensorSize(tensor),
                tensor,
            });
        })
            .filter((entry) => entry !== null);
        return new NDTensorRuntime({
            dtype: "dict",
            shape: entries,
        });
    }
    return undefined;
}
function numericRecordToArray(record) {
    const keys = Object.keys(record);
    if (keys.length === 0 || !keys.every((key) => /^\d+$/.test(key))) {
        return null;
    }
    const sortedKeys = [...keys].sort((left, right) => Number(left) - Number(right));
    for (let index = 0; index < sortedKeys.length; index += 1) {
        if (Number(sortedKeys[index]) !== index) {
            return null;
        }
    }
    return sortedKeys.map((key) => record[key]);
}
function tensorRuntimeFromScalar(value) {
    if (value === null) {
        return new NDTensorRuntime({ dtype: "null" });
    }
    if (typeof value === "string") {
        return new NDTensorRuntime({
            dtype: "string",
            dataString: value,
        });
    }
    if (typeof value === "number") {
        return new NDTensorRuntime({
            dtype: "number",
            dataNumbers: [value],
        });
    }
    return new NDTensorRuntime({
        dtype: "bool",
        dataNumbers: [value ? 1 : 0],
    });
}
function tensorRuntimeFromLegacyArray(array, importContext) {
    const denseNumericData = flattenDenseNumericArray(array);
    if (denseNumericData) {
        return new NDTensorRuntime({
            dtype: denseNumericData.dtype,
            shape: denseNumericData.shape.map((size) => new NDTensorShapeEntryRuntime({ size })),
            dataNumbers: denseNumericData.data,
        });
    }
    const entries = array
        .map((entry, index) => {
        const tensor = tensorRuntimeFromLegacyValue(entry, importContext);
        if (!tensor) {
            return null;
        }
        return new NDTensorShapeEntryRuntime({
            key: String(index),
            size: estimateTensorSize(tensor),
            tensor,
        });
    })
        .filter((entry) => entry !== null);
    return new NDTensorRuntime({
        dtype: "array",
        shape: entries,
    });
}
function isLegacyTensorRecord(record) {
    return ("shape" in record
        || "dtype" in record
        || "data" in record
        || "data_string" in record
        || "data_numbers" in record
        || "data_ubytes" in record
        || "data_path" in record);
}
function isLegacyBufferReferenceRecord(record) {
    return (typeof record.path === "string"
        && record.path.trim() !== ""
        && Object.keys(record).every((key) => key === "path" || key === "format"));
}
function tensorRuntimeFromLegacyBufferReference(record, importContext) {
    const bufferPath = record.path.trim();
    const bufferFormat = record.format && record.format.trim()
        ? record.format.trim()
        : inferBufferFormatFromPath(bufferPath);
    if (importContext) {
        importContext.registerBuffer(bufferPath, bufferFormat);
    }
    return new NDTensorRuntime({
        dtype: inferTensorDtypeFromBufferFormat(bufferFormat),
        dataPath: bufferPath,
    });
}
function tensorRuntimeFromLegacyTensorRecord(record, importContext) {
    const legacyShape = Array.isArray(record.shape)
        ? record.shape
            .map((entry) => (typeof entry === "number" && Number.isFinite(entry) ? entry : null))
            .filter((entry) => entry !== null)
        : [];
    const dtype = typeof record.dtype === "string" ? record.dtype : inferLegacyTensorDtype(record);
    const dataPath = typeof record.data_path === "string" ? record.data_path.trim() : undefined;
    if (dataPath && importContext) {
        importContext.registerBuffer(dataPath, inferBufferFormatFromPath(dataPath));
    }
    const tensor = new NDTensorRuntime({
        dtype,
        shape: legacyShape.map((size) => new NDTensorShapeEntryRuntime({ size })),
        dataPath,
    });
    const explicitNumbers = extractLegacyNumericData(record.data_numbers);
    const explicitBytes = extractLegacyByteData(record.data_ubytes);
    if (typeof record.data_string === "string") {
        tensor.dataString = record.data_string;
        return tensor;
    }
    if (explicitNumbers) {
        tensor.dataNumbers = explicitNumbers;
        return tensor;
    }
    if (explicitBytes) {
        tensor.dataUbytes = explicitBytes;
        return tensor;
    }
    if ("data" in record) {
        const dataValue = normalizeLegacyTensorDataValue(record.data);
        if (typeof dataValue === "string") {
            tensor.dataString = dataValue;
            return tensor;
        }
        const denseNumeric = flattenDenseNumericDataValue(dataValue);
        if (denseNumeric) {
            tensor.dataNumbers = denseNumeric.data;
            if (tensor.shape.length === 0) {
                tensor.shape = denseNumeric.shape.map((size) => new NDTensorShapeEntryRuntime({ size }));
            }
            if (record.dtype === undefined) {
                tensor.dtype = denseNumeric.dtype;
            }
            return tensor;
        }
        const bytes = extractLegacyByteData(dataValue);
        if (bytes) {
            tensor.dataUbytes = bytes;
            if (record.dtype === undefined) {
                tensor.dtype = "bytes";
            }
            return tensor;
        }
    }
    return tensor;
}
function inferLegacyTensorDtype(record) {
    if (typeof record.data_string === "string") {
        return "string";
    }
    if (record.data_numbers !== undefined) {
        return "number";
    }
    if (record.data_ubytes !== undefined) {
        return "bytes";
    }
    if ("data" in record) {
        const dataValue = normalizeLegacyTensorDataValue(record.data);
        if (typeof dataValue === "string") {
            return "string";
        }
        if (flattenDenseNumericDataValue(dataValue)) {
            return "number";
        }
        if (extractLegacyByteData(dataValue)) {
            return "bytes";
        }
    }
    return "null";
}
function extractLegacyNumericData(value) {
    const dense = flattenDenseNumericDataValue(normalizeLegacyTensorDataValue(value));
    return dense ? dense.data : undefined;
}
function extractLegacyByteData(value) {
    if (value instanceof Uint8Array) {
        return new Uint8Array(value);
    }
    if (value instanceof ArrayBuffer) {
        return new Uint8Array(value);
    }
    if (ArrayBuffer.isView(value)) {
        return new Uint8Array(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
    }
    return undefined;
}
function normalizeLegacyTensorDataValue(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return value;
    }
    const numericArray = numericRecordToArray(value);
    return numericArray ?? value;
}
function flattenDenseNumericArray(array) {
    return flattenDenseNumericDataValue(array);
}
function flattenDenseNumericDataValue(value) {
    if (!Array.isArray(value)) {
        return null;
    }
    return flattenDenseNumericRecursive(value);
}
function flattenDenseNumericRecursive(value) {
    if (!Array.isArray(value)) {
        if (typeof value === "number") {
            return { dtype: "number", shape: [], data: [value] };
        }
        if (typeof value === "boolean") {
            return { dtype: "bool", shape: [], data: [value ? 1 : 0] };
        }
        return null;
    }
    const childResults = value.map((entry) => flattenDenseNumericRecursive(entry));
    if (childResults.some((entry) => entry === null)) {
        return null;
    }
    const typedChildren = childResults;
    const childShape = typedChildren.length > 0 ? typedChildren[0].shape : [];
    const childDtype = typedChildren.length > 0 ? typedChildren[0].dtype : "number";
    for (const child of typedChildren) {
        if (child.dtype !== childDtype) {
            return null;
        }
        if (child.shape.length !== childShape.length) {
            return null;
        }
        for (let index = 0; index < child.shape.length; index += 1) {
            if (child.shape[index] !== childShape[index]) {
                return null;
            }
        }
    }
    return {
        dtype: childDtype,
        shape: [value.length, ...childShape],
        data: typedChildren.flatMap((child) => child.data),
    };
}
function estimateTensorSize(tensor) {
    if (tensor.shape.length > 0 && tensor.shape.every((entry) => entry.key === "" && entry.tensor === undefined)) {
        return tensor.shape.reduce((product, entry) => product * Math.max(entry.size, 1), 1);
    }
    if (tensor.dataNumbers && tensor.dataNumbers.length > 0) {
        return tensor.dataNumbers.length;
    }
    if (tensor.dataUbytes && tensor.dataUbytes.length > 0) {
        return tensor.dataUbytes.length;
    }
    if (tensor.dataString) {
        return tensor.dataString.length;
    }
    if (tensor.shape.length > 0) {
        return tensor.shape.length;
    }
    return 1;
}
function inferTensorDtypeFromBufferFormat(format) {
    return format.startsWith("image/") ? "image" : "buffer_ref";
}
function inferBufferFormatFromPath(path) {
    const normalizedPath = path.trim().toLowerCase();
    if (normalizedPath.endsWith(".png")) {
        return "image/png";
    }
    if (normalizedPath.endsWith(".jpg") || normalizedPath.endsWith(".jpeg")) {
        return "image/jpeg";
    }
    if (normalizedPath.endsWith(".webp")) {
        return "image/webp";
    }
    if (normalizedPath.endsWith(".gif")) {
        return "image/gif";
    }
    if (normalizedPath.endsWith(".bmp")) {
        return "image/bmp";
    }
    if (normalizedPath.endsWith(".tif") || normalizedPath.endsWith(".tiff")) {
        return "image/tiff";
    }
    return "application/octet-stream";
}
function packetSceneNodeToRuntime(packetNode, defaultCommitId, nodeIndex) {
    const packetEdge = packetNode.edgeScene();
    const childNodeNames = [];
    if (packetEdge) {
        for (let childIndex = 0; childIndex < packetEdge.childNodesLength(); childIndex += 1) {
            const childName = readPacketString(packetEdge.childNodes(childIndex));
            if (childName) {
                childNodeNames.push(childName);
            }
        }
    }
    return new NDSceneNodeRuntime({
        name: readPacketString(packetNode.name()) ?? `node_${nodeIndex}`,
        commitId: readPacketString(packetNode.commitId()) ?? defaultCommitId,
        parentName: readPacketString(packetNode.parentName()) ?? "",
        edge: new NDSceneEdgeRuntime({
            pose: tensorRuntimeFromPacket(packetEdge?.pose()),
            unpose: tensorRuntimeFromPacket(packetEdge?.unpose()),
            content: tensorRuntimeFromPacket(packetEdge?.content()),
            childNodeNames,
        }),
        edgePacket: clonePacketBytes(packetNode.edgePacketArray()),
    });
}
function packetSceneBufferToRuntime(packetBuffer, defaultCommitId, bufferIndex) {
    return new NDSceneBufferRuntime({
        path: readPacketString(packetBuffer.path()) ?? `buffer_${bufferIndex}`,
        commitId: readPacketString(packetBuffer.commitId()) ?? defaultCommitId,
        format: readPacketString(packetBuffer.format()) ?? "",
        dataEncoded: clonePacketBytes(packetBuffer.dataEncodedArray()),
        dataDecoded: tensorRuntimeFromPacket(packetBuffer.dataDecoded()),
    });
}
function createTimelineManifestBuffer(commitId, manifest) {
    const manifestJson = JSON.stringify(manifest);
    return new NDSceneBufferRuntime({
        path: TIMELINE_MANIFEST_PATH,
        commitId,
        format: TIMELINE_MANIFEST_FORMAT,
        dataEncoded: STRING_ENCODER.encode(manifestJson),
        dataDecoded: new NDTensorRuntime({
            dtype: "string",
            dataString: manifestJson,
        }),
    });
}
function parseTimelineManifestBuffer(buffer) {
    const manifestJson = buffer.dataDecoded?.scalarString() ?? (buffer.dataEncoded ? STRING_DECODER.decode(buffer.dataEncoded) : "");
    if (!manifestJson) {
        throw new Error("Timeline manifest buffer is empty.");
    }
    const manifest = JSON.parse(manifestJson);
    if (manifest.version !== 1) {
        throw new Error(`Unsupported timeline manifest version "${String(manifest.version)}".`);
    }
    return manifest;
}
function runtimeFromTimelinePacket(commitPacket, sceneNodes, sceneBuffers, manifestBuffer) {
    const manifest = parseTimelineManifestBuffer(manifestBuffer);
    const commitInfoById = new Map(manifest.commits.map((commit) => [commit.commitId, commit]));
    const nodeVersionsByCommitId = new Map();
    const bufferVersionsByCommitId = new Map();
    for (const sceneNode of sceneNodes) {
        let commitNodes = nodeVersionsByCommitId.get(sceneNode.commitId);
        if (!commitNodes) {
            commitNodes = new Map();
            nodeVersionsByCommitId.set(sceneNode.commitId, commitNodes);
        }
        commitNodes.set(sceneNode.name, sceneNode);
    }
    for (const sceneBuffer of sceneBuffers) {
        if (sceneBuffer.path === TIMELINE_MANIFEST_PATH) {
            continue;
        }
        let commitBuffers = bufferVersionsByCommitId.get(sceneBuffer.commitId);
        if (!commitBuffers) {
            commitBuffers = new Map();
            bufferVersionsByCommitId.set(sceneBuffer.commitId, commitBuffers);
        }
        commitBuffers.set(sceneBuffer.path, sceneBuffer);
    }
    const runtime = new NDSceneRuntime();
    const topLevelCommitId = readPacketString(commitPacket?.commitId()) ?? manifest.activeCommitId;
    let previousScene = null;
    for (const commitId of manifest.commitOrder) {
        const commitInfo = commitInfoById.get(commitId);
        if (!commitInfo) {
            throw new Error(`Timeline manifest is missing commit metadata for "${commitId}".`);
        }
        const scene = cloneSceneGraphReferences(previousScene);
        const commitNodes = nodeVersionsByCommitId.get(commitId) ?? new Map();
        const commitBuffers = bufferVersionsByCommitId.get(commitId) ?? new Map();
        for (const deletedNodeName of commitInfo.deletedNodeNames) {
            scene.nodesByName.delete(deletedNodeName);
        }
        for (const deletedBufferPath of commitInfo.deletedBufferPaths) {
            scene.buffersByPath.delete(deletedBufferPath);
        }
        for (const changedNodeName of commitInfo.changedNodeNames) {
            const changedNode = commitNodes.get(changedNodeName);
            if (!changedNode) {
                throw new Error(`Timeline packet is missing node "${changedNodeName}" for commit "${commitId}".`);
            }
            scene.nodesByName.set(changedNodeName, changedNode);
        }
        for (const changedBufferPath of commitInfo.changedBufferPaths) {
            const changedBuffer = commitBuffers.get(changedBufferPath);
            if (!changedBuffer) {
                throw new Error(`Timeline packet is missing buffer "${changedBufferPath}" for commit "${commitId}".`);
            }
            scene.buffersByPath.set(changedBufferPath, changedBuffer);
        }
        scene.rebuildRoots();
        runtime.addCommit(new NDSceneCommitRuntime({
            commitId: commitInfo.commitId,
            commitPreviousId: commitInfo.commitPreviousId,
            commitInputId: commitInfo.commitInputId,
            createdAt: commitInfo.createdAt,
            createdByModelId: commitInfo.createdByModelId,
            scene,
            packet: commitId === topLevelCommitId ? clonePacketBytes(commitPacket?.packetArray()) : undefined,
        }), commitId === manifest.activeCommitId);
        previousScene = scene;
    }
    if (!runtime.activeCommitId && manifest.commitOrder.length > 0) {
        runtime.setActiveCommit(manifest.commitOrder[manifest.commitOrder.length - 1]);
    }
    return runtime;
}
function cloneSceneGraphReferences(scene) {
    const clonedScene = new NDSceneGraphRuntime();
    if (!scene) {
        return clonedScene;
    }
    clonedScene.nodesByName = new Map(scene.nodesByName);
    clonedScene.buffersByPath = new Map(scene.buffersByPath);
    clonedScene.rootNodeNames = [...scene.rootNodeNames];
    return clonedScene;
}
function shareSceneGraphSnapshot(previousScene, nextScene) {
    const sharedScene = new NDSceneGraphRuntime();
    for (const [nodeName, sceneNode] of nextScene.nodesByName) {
        const previousNode = previousScene?.getNode(nodeName);
        sharedScene.nodesByName.set(nodeName, previousNode && sceneNodeEquivalent(previousNode, sceneNode) ? previousNode : sceneNode);
    }
    for (const [bufferPath, sceneBuffer] of nextScene.buffersByPath) {
        const previousBuffer = previousScene?.getBuffer(bufferPath);
        sharedScene.buffersByPath.set(bufferPath, previousBuffer && sceneBufferEquivalent(previousBuffer, sceneBuffer) ? previousBuffer : sceneBuffer);
    }
    sharedScene.rootNodeNames = [...nextScene.rootNodeNames];
    return sharedScene;
}
function diffSceneGraphs(previousScene, nextScene) {
    const changedNodeNames = [];
    const changedBufferPaths = [];
    const deletedNodeNames = previousScene
        ? [...previousScene.nodesByName.keys()].filter((nodeName) => !nextScene.nodesByName.has(nodeName))
        : [];
    const deletedBufferPaths = previousScene
        ? [...previousScene.buffersByPath.keys()].filter((bufferPath) => !nextScene.buffersByPath.has(bufferPath))
        : [];
    for (const [nodeName, sceneNode] of nextScene.nodesByName) {
        const previousNode = previousScene?.getNode(nodeName);
        if (!previousNode || !sceneNodeEquivalent(previousNode, sceneNode)) {
            changedNodeNames.push(nodeName);
        }
    }
    for (const [bufferPath, sceneBuffer] of nextScene.buffersByPath) {
        const previousBuffer = previousScene?.getBuffer(bufferPath);
        if (!previousBuffer || !sceneBufferEquivalent(previousBuffer, sceneBuffer)) {
            changedBufferPaths.push(bufferPath);
        }
    }
    return {
        changedNodeNames,
        deletedNodeNames,
        changedBufferPaths,
        deletedBufferPaths,
    };
}
function sceneNodeEquivalent(left, right) {
    if (left === right) {
        return true;
    }
    return (left.name === right.name
        && left.parentName === right.parentName
        && tensorEquivalent(left.edge.pose, right.edge.pose)
        && tensorEquivalent(left.edge.unpose, right.edge.unpose)
        && tensorEquivalent(left.edge.content, right.edge.content)
        && stringArrayEquivalent(left.edge.childNodeNames, right.edge.childNodeNames)
        && uint8ArrayEquivalent(left.edgePacket, right.edgePacket));
}
function sceneBufferEquivalent(left, right) {
    if (left === right) {
        return true;
    }
    return (left.path === right.path
        && left.format === right.format
        && uint8ArrayEquivalent(left.dataEncoded, right.dataEncoded)
        && tensorEquivalent(left.dataDecoded, right.dataDecoded));
}
function tensorEquivalent(left, right) {
    if (left === right) {
        return true;
    }
    if (!left || !right) {
        return left === right;
    }
    return (left.dtype === right.dtype
        && left.dataString === right.dataString
        && left.dataPath === right.dataPath
        && numberArrayEquivalent(left.dataNumbers, right.dataNumbers)
        && uint8ArrayEquivalent(left.dataUbytes, right.dataUbytes)
        && shapeEntriesEquivalent(left.shape, right.shape));
}
function shapeEntriesEquivalent(left, right) {
    if (left.length !== right.length) {
        return false;
    }
    for (let index = 0; index < left.length; index += 1) {
        if (left[index].size !== right[index].size
            || left[index].key !== right[index].key
            || !tensorEquivalent(left[index].tensor, right[index].tensor)) {
            return false;
        }
    }
    return true;
}
function stringArrayEquivalent(left, right) {
    if (left.length !== right.length) {
        return false;
    }
    for (let index = 0; index < left.length; index += 1) {
        if (left[index] !== right[index]) {
            return false;
        }
    }
    return true;
}
function numberArrayEquivalent(left, right) {
    if (!left || !right) {
        return left === right;
    }
    if (left.length !== right.length) {
        return false;
    }
    for (let index = 0; index < left.length; index += 1) {
        if (left[index] !== right[index]) {
            return false;
        }
    }
    return true;
}
function uint8ArrayEquivalent(left, right) {
    if (!left || !right) {
        return left === right;
    }
    if (left.length !== right.length) {
        return false;
    }
    for (let index = 0; index < left.length; index += 1) {
        if (left[index] !== right[index]) {
            return false;
        }
    }
    return true;
}
function createOptionalString(builder, value) {
    return value ? builder.createString(value) : 0;
}
function readPacketString(value) {
    if (value === null || value === undefined) {
        return null;
    }
    return typeof value === "string" ? value : STRING_DECODER.decode(value);
}
function clonePacketBytes(value) {
    return value ? new Uint8Array(ndPacketBytes(value)) : undefined;
}
