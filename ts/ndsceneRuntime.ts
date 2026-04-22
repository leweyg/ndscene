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

export type NDRuntimeScalar = null | string | number | boolean | Uint8Array;
export type NDRuntimeValue = NDRuntimeScalar | NDRuntimeValue[] | { [key: string]: NDRuntimeValue };

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

export class NDSceneEdgeRuntime {
  pose?: NDRuntimeValue;
  unpose?: NDRuntimeValue;
  content?: NDRuntimeValue;
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
  dataDecoded?: NDRuntimeValue;

  constructor(init: {
    path: string;
    commitId: string;
    format?: string;
    dataEncoded?: Uint8Array;
    dataDecoded?: NDRuntimeValue;
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
            pose: tensorToRuntimeValue(packetEdge?.pose()),
            unpose: tensorToRuntimeValue(packetEdge?.unpose()),
            content: tensorToRuntimeValue(packetEdge?.content()),
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
          dataDecoded: tensorToRuntimeValue(packetBuffer.dataDecoded()),
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
          pose: normalizeLegacyValue(nodeJson.pose),
          unpose: normalizeLegacyValue(nodeJson.unpose),
          content: normalizeLegacyValue(nodeJson.content),
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
  const poseOffset = edge.pose !== undefined ? serializeRuntimeValueTensor(builder, edge.pose) : 0;
  const unposeOffset = edge.unpose !== undefined ? serializeRuntimeValueTensor(builder, edge.unpose) : 0;
  const contentOffset = edge.content !== undefined ? serializeRuntimeValueTensor(builder, edge.content) : 0;
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
  const decodedOffset = buffer.dataDecoded !== undefined ? serializeRuntimeValueTensor(builder, buffer.dataDecoded) : 0;

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

function serializeRuntimeValueTensor(builder: Builder, value: NDRuntimeValue): number {
  const normalizedValue = normalizeLegacyValue(value);
  if (normalizedValue === undefined) {
    return 0;
  }

  let dtype = "null";
  let shapeOffset = 0;
  let dataStringOffset = 0;
  let dataNumbersOffset = 0;
  let dataUbytesOffset = 0;

  if (normalizedValue === null) {
    dtype = "null";
  } else if (typeof normalizedValue === "string") {
    dtype = "string";
    dataStringOffset = builder.createString(normalizedValue);
  } else if (typeof normalizedValue === "number") {
    dtype = "number";
    dataNumbersOffset = NDPacketTensor.createDataNumbersVector(builder, [normalizedValue]);
  } else if (typeof normalizedValue === "boolean") {
    dtype = "bool";
    dataNumbersOffset = NDPacketTensor.createDataNumbersVector(builder, [normalizedValue ? 1 : 0]);
  } else if (normalizedValue instanceof Uint8Array) {
    dtype = "bytes";
    dataUbytesOffset = NDPacketTensor.createDataUbytesVector(builder, normalizedValue);
  } else if (Array.isArray(normalizedValue)) {
    if (normalizedValue.every((entry) => typeof entry === "number")) {
      dtype = "number[]";
      dataNumbersOffset = NDPacketTensor.createDataNumbersVector(builder, normalizedValue);
    } else if (normalizedValue.every((entry) => typeof entry === "boolean")) {
      dtype = "bool[]";
      dataNumbersOffset = NDPacketTensor.createDataNumbersVector(
        builder,
        normalizedValue.map((entry) => (entry ? 1 : 0)),
      );
    } else {
      dtype = "array";
      shapeOffset = serializeTensorShapeEntries(
        builder,
        normalizedValue.map((entry, index) => [String(index), entry] as const),
      );
    }
  } else {
    dtype = "dict";
    shapeOffset = serializeTensorShapeEntries(builder, Object.entries(normalizedValue));
  }

  const dtypeOffset = builder.createString(dtype);
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
  return NDPacketTensor.endNDPacketTensor(builder);
}

function serializeTensorShapeEntries(
  builder: Builder,
  entries: ReadonlyArray<readonly [string, NDRuntimeValue]>,
): number {
  const entryOffsets = entries
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => {
      const tensorOffset = serializeRuntimeValueTensor(builder, value);
      const keyOffset = builder.createString(key);
      NDPacketShapeEntry.startNDPacketShapeEntry(builder);
      NDPacketShapeEntry.addSize(builder, estimateValueSize(value));
      NDPacketShapeEntry.addKey(builder, keyOffset);
      if (tensorOffset !== 0) {
        NDPacketShapeEntry.addTensor(builder, tensorOffset);
      }
      return NDPacketShapeEntry.endNDPacketShapeEntry(builder);
    });

  return entryOffsets.length > 0 ? NDPacketTensor.createShapeVector(builder, entryOffsets) : 0;
}

function tensorToRuntimeValue(packetTensor: NDPacketTensor | null | undefined): NDRuntimeValue | undefined {
  if (!packetTensor) {
    return undefined;
  }

  const dtype = readPacketString(packetTensor.dtype()) ?? "";
  if (dtype === "null") {
    return null;
  }
  if (dtype === "string") {
    return readPacketString(packetTensor.dataString()) ?? "";
  }
  if (dtype === "number") {
    const numbers = packetTensor.dataNumbersArray();
    return numbers && numbers.length > 0 ? numbers[0] : 0;
  }
  if (dtype === "bool") {
    const numbers = packetTensor.dataNumbersArray();
    return Boolean(numbers && numbers.length > 0 ? numbers[0] : 0);
  }
  if (dtype === "number[]") {
    return packetTensor.dataNumbersArray() ? Array.from(packetTensor.dataNumbersArray() as Float32Array) : [];
  }
  if (dtype === "bool[]") {
    return packetTensor.dataNumbersArray()
      ? Array.from(packetTensor.dataNumbersArray() as Float32Array, (entry) => Boolean(entry))
      : [];
  }
  if (dtype === "bytes") {
    return clonePacketBytes(packetTensor.dataUbytesArray()) ?? new Uint8Array();
  }

  const shapeEntries: Array<[string, NDRuntimeValue]> = [];
  for (let shapeIndex = 0; shapeIndex < packetTensor.shapeLength(); shapeIndex += 1) {
    const shapeEntry = packetTensor.shape(shapeIndex);
    if (!shapeEntry) {
      continue;
    }
    const entryKey = readPacketString(shapeEntry.key()) ?? String(shapeIndex);
    shapeEntries.push([entryKey, tensorToRuntimeValue(shapeEntry.tensor()) ?? null]);
  }

  if (dtype === "array" || (shapeEntries.length > 0 && shapeEntries.every(([key]) => /^\d+$/.test(key)))) {
    return shapeEntries
      .sort((left, right) => Number(left[0]) - Number(right[0]))
      .map(([, value]) => value);
  }

  if (dtype === "dict" || shapeEntries.length > 0) {
    const result: { [key: string]: NDRuntimeValue } = {};
    for (const [key, value] of shapeEntries) {
      result[key] = value;
    }
    return result;
  }

  if (packetTensor.dataNumbersLength() > 0 && packetTensor.dataNumbersArray()) {
    return Array.from(packetTensor.dataNumbersArray() as Float32Array);
  }
  if (packetTensor.dataUbytesLength() > 0) {
    return clonePacketBytes(packetTensor.dataUbytesArray()) ?? new Uint8Array();
  }
  if (packetTensor.dataString() !== null) {
    return readPacketString(packetTensor.dataString()) ?? "";
  }
  return null;
}

function normalizeLegacyValue(value: unknown): NDRuntimeValue | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (value instanceof Uint8Array) {
    return new Uint8Array(value);
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  if (ArrayBuffer.isView(value)) {
    if (value instanceof Uint8Array) {
      return new Uint8Array(value);
    }
    const viewBytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    return Array.from(viewBytes) as NDRuntimeValue[];
  }
  if (Array.isArray(value)) {
    return value
      .map((entry) => normalizeLegacyValue(entry))
      .filter((entry): entry is NDRuntimeValue => entry !== undefined);
  }
  if (value && typeof value === "object") {
    const numericArray = numericRecordToArray(value as Record<string, unknown>);
    if (numericArray) {
      return numericArray
        .map((entry) => normalizeLegacyValue(entry))
        .filter((entry): entry is NDRuntimeValue => entry !== undefined);
    }

    const result: { [key: string]: NDRuntimeValue } = {};
    for (const [entryKey, entryValue] of Object.entries(value as Record<string, unknown>)) {
      const normalizedEntry = normalizeLegacyValue(entryValue);
      if (normalizedEntry !== undefined) {
        result[entryKey] = normalizedEntry;
      }
    }
    return result;
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

function estimateValueSize(value: NDRuntimeValue): number {
  if (value instanceof Uint8Array) {
    return value.length;
  }
  if (Array.isArray(value)) {
    return value.length;
  }
  if (value && typeof value === "object") {
    return Object.keys(value).length;
  }
  if (typeof value === "string") {
    return value.length;
  }
  return 1;
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
