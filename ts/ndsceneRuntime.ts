import {
  Builder,
  NDPacketBuffer,
  NDPacketRoot,
  NDPacketSceneCommit,
  NDPacketSceneEdge,
  NDPacketSceneNode,
  NDPacketSceneUpdate,
  NDPacketShapeEntry,
  NDPacketTensor,
  ndPacketBytes,
  ndPacketRootFromBuffer,
} from "./ndsceneFlatbuffers.js";

const ROOT_PACKET_MAGIC = "NDSN";
const STRING_DECODER = new TextDecoder();

export type NDLegacySceneNodeJson = {
  name?: string;
  pose?: unknown;
  unpose?: unknown;
  content?: unknown;
  children?: Array<string | NDLegacySceneNodeJson>;
  [key: string]: unknown;
};

export type NDLegacySceneJson = {
  root: NDLegacySceneNodeJson;
  objects?: Record<string, NDLegacySceneNodeJson>;
  [key: string]: unknown;
};

export type NDSceneRuntimeFromJsonOptions = {
  commitId?: string;
  commitPreviousId?: string;
  commitInputId?: string;
  createdAt?: string;
  createdByModelId?: string;
};

type NDLegacyImportContext = {
  registerBuffer: (path: string, format?: string) => NDSceneBufferRuntime;
};

export class NDTensorShapeEntryRuntime {
  size: number;
  key: string;
  tensor?: NDTensorRuntime;

  constructor(init: {
    size?: number;
    key?: string;
    tensor?: NDTensorRuntime;
  } = {}) {
    this.size = init.size ?? 0;
    this.key = init.key ?? "";
    this.tensor = init.tensor;
  }
}

export class NDTensorRuntime {
  shape: NDTensorShapeEntryRuntime[];
  dtype: string;
  dataString?: string;
  dataNumbers?: number[];
  dataUbytes?: Uint8Array;
  dataPath?: string;

  constructor(init: {
    shape?: NDTensorShapeEntryRuntime[];
    dtype?: string;
    dataString?: string;
    dataNumbers?: number[];
    dataUbytes?: Uint8Array;
    dataPath?: string;
  } = {}) {
    this.shape = init.shape ? [...init.shape] : [];
    this.dtype = init.dtype ?? "null";
    this.dataString = init.dataString;
    this.dataNumbers = init.dataNumbers ? [...init.dataNumbers] : undefined;
    this.dataUbytes = init.dataUbytes ? new Uint8Array(init.dataUbytes) : undefined;
    this.dataPath = init.dataPath;
  }

  child(key: string): NDTensorRuntime | undefined {
    return this.shape.find((entry) => entry.key === key)?.tensor;
  }

  childAt(index: number): NDTensorRuntime | undefined {
    return this.child(String(index));
  }

  hasDenseNumbers(): boolean {
    return Boolean(this.dataNumbers && this.dataNumbers.length > 0);
  }

  denseNumbers(): number[] {
    return this.dataNumbers ? [...this.dataNumbers] : [];
  }

  scalarNumber(): number | undefined {
    return this.dataNumbers && this.dataNumbers.length > 0 ? this.dataNumbers[0] : undefined;
  }

  scalarBoolean(): boolean | undefined {
    if (this.dtype !== "bool") {
      return undefined;
    }
    const value = this.scalarNumber();
    return value === undefined ? undefined : value !== 0;
  }

  scalarString(): string | undefined {
    return this.dataString;
  }

  bytes(): Uint8Array | undefined {
    return this.dataUbytes ? new Uint8Array(this.dataUbytes) : undefined;
  }

  bufferPath(): string | undefined {
    return this.dataPath;
  }

  shapeSizes(): number[] {
    return this.shape.map((entry) => entry.size);
  }
}

export class NDSceneEdgeRuntime {
  pose?: NDTensorRuntime;
  unpose?: NDTensorRuntime;
  content?: NDTensorRuntime;
  childNodeNames: string[];

  constructor(init: Partial<NDSceneEdgeRuntime> = {}) {
    this.pose = init.pose;
    this.unpose = init.unpose;
    this.content = init.content;
    this.childNodeNames = init.childNodeNames ? [...init.childNodeNames] : [];
  }
}

export class NDSceneNodeRuntime {
  name: string;
  commitId: string;
  parentName: string;
  edge: NDSceneEdgeRuntime;
  edgePacket?: Uint8Array;

  constructor(init: {
    name: string;
    commitId: string;
    parentName?: string;
    edge?: NDSceneEdgeRuntime;
    edgePacket?: Uint8Array;
  }) {
    this.name = init.name;
    this.commitId = init.commitId;
    this.parentName = init.parentName ?? "";
    this.edge = init.edge ?? new NDSceneEdgeRuntime();
    this.edgePacket = init.edgePacket ? new Uint8Array(init.edgePacket) : undefined;
  }
}

export class NDSceneBufferRuntime {
  path: string;
  commitId: string;
  format: string;
  dataEncoded?: Uint8Array;
  dataDecoded?: NDTensorRuntime;

  constructor(init: {
    path: string;
    commitId: string;
    format?: string;
    dataEncoded?: Uint8Array;
    dataDecoded?: NDTensorRuntime;
  }) {
    this.path = init.path;
    this.commitId = init.commitId;
    this.format = init.format ?? "";
    this.dataEncoded = init.dataEncoded ? new Uint8Array(init.dataEncoded) : undefined;
    this.dataDecoded = init.dataDecoded;
  }
}

export class NDSceneGraphRuntime {
  nodesByName = new Map<string, NDSceneNodeRuntime>();
  buffersByPath = new Map<string, NDSceneBufferRuntime>();
  rootNodeNames: string[] = [];

  addNode(node: NDSceneNodeRuntime): void {
    if (this.nodesByName.has(node.name)) {
      throw new Error(`Duplicate scene node name "${node.name}".`);
    }
    this.nodesByName.set(node.name, node);
  }

  addBuffer(buffer: NDSceneBufferRuntime): void {
    this.buffersByPath.set(buffer.path, buffer);
  }

  getNode(name: string): NDSceneNodeRuntime | undefined {
    return this.nodesByName.get(name);
  }

  getBuffer(path: string): NDSceneBufferRuntime | undefined {
    return this.buffersByPath.get(path);
  }

  getRootNodes(): NDSceneNodeRuntime[] {
    return this.rootNodeNames
      .map((name) => this.getNode(name))
      .filter((node): node is NDSceneNodeRuntime => Boolean(node));
  }

  getChildNodes(node: NDSceneNodeRuntime): NDSceneNodeRuntime[] {
    return node.edge.childNodeNames
      .map((name) => this.getNode(name))
      .filter((child): child is NDSceneNodeRuntime => Boolean(child));
  }

  rebuildRoots(): void {
    const referencedNames = new Set<string>();
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
  commitId: string;
  commitPreviousId: string;
  commitInputId: string;
  createdAt: string;
  createdByModelId: string;
  scene: NDSceneGraphRuntime;
  packet?: Uint8Array;

  constructor(init: {
    commitId: string;
    commitPreviousId?: string;
    commitInputId?: string;
    createdAt?: string;
    createdByModelId?: string;
    scene?: NDSceneGraphRuntime;
    packet?: Uint8Array;
  }) {
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
  commitsById = new Map<string, NDSceneCommitRuntime>();
  activeCommitId: string | null = null;

  addCommit(commit: NDSceneCommitRuntime, setActive = false): void {
    this.commitsById.set(commit.commitId, commit);
    if (setActive || this.activeCommitId === null) {
      this.activeCommitId = commit.commitId;
    }
  }

  get activeCommit(): NDSceneCommitRuntime {
    if (!this.activeCommitId) {
      throw new Error("NDSceneRuntime has no active commit.");
    }

    const commit = this.commitsById.get(this.activeCommitId);
    if (!commit) {
      throw new Error(`Active commit "${this.activeCommitId}" is missing from the runtime.`);
    }
    return commit;
  }

  get activeScene(): NDSceneGraphRuntime {
    return this.activeCommit.scene;
  }

  toFlatbufferBuffer(): Uint8Array {
    return serializeCommitToFlatbuffer(this.activeCommit);
  }

  static fromFlatbufferBuffer(buffer: ArrayBuffer | ArrayBufferView): NDSceneRuntime {
    const rootPacket = ndPacketRootFromBuffer(buffer);
    const commitPacket = rootPacket.commit();
    const scenePacket = commitPacket?.scene() ?? rootPacket.scene();
    if (!scenePacket) {
      throw new Error("FlatBuffer packet does not contain a scene or commit payload.");
    }

    const scene = new NDSceneGraphRuntime();
    const commitId = readPacketString(commitPacket?.commitId()) ?? "commit_from_buffer";

    for (let nodeIndex = 0; nodeIndex < scenePacket.nodesLength(); nodeIndex += 1) {
      const packetNode = scenePacket.nodes(nodeIndex);
      if (!packetNode) {
        continue;
      }

      const packetEdge = packetNode.edgeScene();
      const childNodeNames: string[] = [];
      if (packetEdge) {
        for (let childIndex = 0; childIndex < packetEdge.childNodesLength(); childIndex += 1) {
          const childName = readPacketString(packetEdge.childNodes(childIndex));
          if (childName) {
            childNodeNames.push(childName);
          }
        }
      }

      scene.addNode(
        new NDSceneNodeRuntime({
          name: readPacketString(packetNode.name()) ?? `node_${nodeIndex}`,
          commitId: readPacketString(packetNode.commitId()) ?? commitId,
          parentName: readPacketString(packetNode.parentName()) ?? "",
          edge: new NDSceneEdgeRuntime({
            pose: tensorRuntimeFromPacket(packetEdge?.pose()),
            unpose: tensorRuntimeFromPacket(packetEdge?.unpose()),
            content: tensorRuntimeFromPacket(packetEdge?.content()),
            childNodeNames,
          }),
          edgePacket: clonePacketBytes(packetNode.edgePacketArray()),
        }),
      );
    }

    for (let bufferIndex = 0; bufferIndex < scenePacket.buffersLength(); bufferIndex += 1) {
      const packetBuffer = scenePacket.buffers(bufferIndex);
      if (!packetBuffer) {
        continue;
      }

      scene.addBuffer(
        new NDSceneBufferRuntime({
          path: readPacketString(packetBuffer.path()) ?? `buffer_${bufferIndex}`,
          commitId: readPacketString(packetBuffer.commitId()) ?? commitId,
          format: readPacketString(packetBuffer.format()) ?? "",
          dataEncoded: clonePacketBytes(packetBuffer.dataEncodedArray()),
          dataDecoded: tensorRuntimeFromPacket(packetBuffer.dataDecoded()),
        }),
      );
    }

    scene.rebuildRoots();

    const runtime = new NDSceneRuntime();
    runtime.addCommit(
      new NDSceneCommitRuntime({
        commitId,
        commitPreviousId: readPacketString(commitPacket?.commitPreviousId()) ?? "",
        commitInputId: readPacketString(commitPacket?.commitInputId()) ?? commitId,
        createdAt: readPacketString(commitPacket?.createdAt()) ?? "",
        createdByModelId: readPacketString(commitPacket?.createdByModelId()) ?? "ndscene-runtime",
        scene,
        packet: clonePacketBytes(commitPacket?.packetArray()),
      }),
      true,
    );
    return runtime;
  }

  static fromLegacyJson(sceneJson: NDLegacySceneJson, options: NDSceneRuntimeFromJsonOptions = {}): NDSceneRuntime {
    if (!sceneJson.root || typeof sceneJson.root !== "object") {
      throw new Error("Legacy JSON scene is missing a valid root object.");
    }

    const commitId = options.commitId ?? "commit_from_json";
    const scene = new NDSceneGraphRuntime();
    const usedNames = new Set<string>();
    const sharedNameByDefinitionKey = new Map<string, string>();
    const definitions = sceneJson.objects ?? {};
    const ensureBuffer = (path: string, format?: string): NDSceneBufferRuntime => {
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
    const importContext: NDLegacyImportContext = {
      registerBuffer: ensureBuffer,
    };

    const claimUniqueName = (baseName: string): string => {
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

    const importResolvedNode = (
      nodeJson: NDLegacySceneNodeJson,
      resolvedName: string,
      parentName: string,
    ): string => {
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
      const childNodeNames: string[] = [];

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

        const inlineName =
          typeof child.name === "string" && child.name
            ? claimUniqueName(child.name)
            : claimUniqueName(`${resolvedName}_child_${childIndex}`);
        importResolvedNode(child, inlineName, resolvedName);
        childNodeNames.push(inlineName);
      }

      runtimeNode.edge.childNodeNames = childNodeNames;
      return resolvedName;
    };

    const rootPreferredName =
      typeof sceneJson.root.name === "string" && sceneJson.root.name
        ? sceneJson.root.name
        : "root";

    importResolvedNode(sceneJson.root, claimUniqueName(rootPreferredName), "");
    scene.rebuildRoots();

    const runtime = new NDSceneRuntime();
    runtime.addCommit(
      new NDSceneCommitRuntime({
        commitId,
        commitPreviousId: options.commitPreviousId ?? "",
        commitInputId: options.commitInputId ?? commitId,
        createdAt: options.createdAt ?? new Date().toISOString(),
        createdByModelId: options.createdByModelId ?? "legacy-json",
        scene,
      }),
      true,
    );
    return runtime;
  }
}

function serializeCommitToFlatbuffer(commit: NDSceneCommitRuntime): Uint8Array {
  const builder = new Builder(1024);
  const sceneOffset = serializeSceneGraph(builder, commit.scene);

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

function serializeSceneGraph(builder: Builder, scene: NDSceneGraphRuntime): number {
  const nodeOffsets = [...scene.nodesByName.values()].map((node) => serializeSceneNode(builder, node));
  const bufferOffsets = [...scene.buffersByPath.values()].map((buffer) => serializeSceneBuffer(builder, buffer));

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

function serializeSceneNode(builder: Builder, node: NDSceneNodeRuntime): number {
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

function serializeSceneEdge(builder: Builder, edge: NDSceneEdgeRuntime): number {
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

function serializeSceneBuffer(builder: Builder, buffer: NDSceneBufferRuntime): number {
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

function serializeTensorRuntime(builder: Builder, tensor: NDTensorRuntime): number {
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

function serializeTensorShapeEntries(
  builder: Builder,
  entries: ReadonlyArray<NDTensorShapeEntryRuntime>,
): number {
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

function tensorRuntimeFromPacket(packetTensor: NDPacketTensor | null | undefined): NDTensorRuntime | undefined {
  if (!packetTensor) {
    return undefined;
  }

  const shapeEntries: NDTensorShapeEntryRuntime[] = [];
  for (let shapeIndex = 0; shapeIndex < packetTensor.shapeLength(); shapeIndex += 1) {
    const shapeEntry = packetTensor.shape(shapeIndex);
    if (!shapeEntry) {
      continue;
    }
    shapeEntries.push(
      new NDTensorShapeEntryRuntime({
        size: shapeEntry.size(),
        key: readPacketString(shapeEntry.key()) ?? "",
        tensor: tensorRuntimeFromPacket(shapeEntry.tensor()),
      }),
    );
  }

  return new NDTensorRuntime({
    shape: shapeEntries,
    dtype: readPacketString(packetTensor.dtype()) ?? "null",
    dataString: readPacketString(packetTensor.dataString()) ?? undefined,
    dataNumbers: packetTensor.dataNumbersArray() ? Array.from(packetTensor.dataNumbersArray() as Float32Array) : undefined,
    dataUbytes: clonePacketBytes(packetTensor.dataUbytesArray()),
    dataPath: readPacketString(packetTensor.dataPath()) ?? undefined,
  });
}

function tensorRuntimeFromLegacyValue(
  value: unknown,
  importContext?: NDLegacyImportContext,
): NDTensorRuntime | undefined {
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
    const record = value as Record<string, unknown>;
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
      .filter((entry): entry is NDTensorShapeEntryRuntime => entry !== null);

    return new NDTensorRuntime({
      dtype: "dict",
      shape: entries,
    });
  }
  return undefined;
}

function numericRecordToArray(record: Record<string, unknown>): unknown[] | null {
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

function tensorRuntimeFromScalar(value: null | string | number | boolean): NDTensorRuntime {
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

function tensorRuntimeFromLegacyArray(array: unknown[], importContext?: NDLegacyImportContext): NDTensorRuntime {
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
    .filter((entry): entry is NDTensorShapeEntryRuntime => entry !== null);

  return new NDTensorRuntime({
    dtype: "array",
    shape: entries,
  });
}

function isLegacyTensorRecord(record: Record<string, unknown>): boolean {
  return (
    "shape" in record
    || "dtype" in record
    || "data" in record
    || "data_string" in record
    || "data_numbers" in record
    || "data_ubytes" in record
    || "data_path" in record
  );
}

function isLegacyBufferReferenceRecord(record: Record<string, unknown>): record is { path: string; format?: string } {
  return (
    typeof record.path === "string"
    && record.path.trim() !== ""
    && Object.keys(record).every((key) => key === "path" || key === "format")
  );
}

function tensorRuntimeFromLegacyBufferReference(
  record: { path: string; format?: string },
  importContext?: NDLegacyImportContext,
): NDTensorRuntime {
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

function tensorRuntimeFromLegacyTensorRecord(
  record: Record<string, unknown>,
  importContext?: NDLegacyImportContext,
): NDTensorRuntime {
  const legacyShape = Array.isArray(record.shape)
    ? record.shape
        .map((entry) => (typeof entry === "number" && Number.isFinite(entry) ? entry : null))
        .filter((entry): entry is number => entry !== null)
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

function inferLegacyTensorDtype(record: Record<string, unknown>): string {
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

function extractLegacyNumericData(value: unknown): number[] | undefined {
  const dense = flattenDenseNumericDataValue(normalizeLegacyTensorDataValue(value));
  return dense ? dense.data : undefined;
}

function extractLegacyByteData(value: unknown): Uint8Array | undefined {
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

function normalizeLegacyTensorDataValue(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }

  const numericArray = numericRecordToArray(value as Record<string, unknown>);
  return numericArray ?? value;
}

function flattenDenseNumericArray(array: unknown[]): { dtype: "number" | "bool"; shape: number[]; data: number[] } | null {
  return flattenDenseNumericDataValue(array);
}

function flattenDenseNumericDataValue(value: unknown): { dtype: "number" | "bool"; shape: number[]; data: number[] } | null {
  if (!Array.isArray(value)) {
    return null;
  }
  return flattenDenseNumericRecursive(value);
}

function flattenDenseNumericRecursive(value: unknown): { dtype: "number" | "bool"; shape: number[]; data: number[] } | null {
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

  const typedChildren = childResults as Array<{ dtype: "number" | "bool"; shape: number[]; data: number[] }>;
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

function estimateTensorSize(tensor: NDTensorRuntime): number {
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

function inferTensorDtypeFromBufferFormat(format: string): string {
  return format.startsWith("image/") ? "image" : "buffer_ref";
}

function inferBufferFormatFromPath(path: string): string {
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

function createOptionalString(builder: Builder, value: string): number {
  return value ? builder.createString(value) : 0;
}

function readPacketString(value: string | Uint8Array | null | undefined): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  return typeof value === "string" ? value : STRING_DECODER.decode(value);
}

function clonePacketBytes(value: Uint8Array | null | undefined): Uint8Array | undefined {
  return value ? new Uint8Array(ndPacketBytes(value)) : undefined;
}
