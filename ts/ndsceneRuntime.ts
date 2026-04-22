import {
  runtimeFromFlatbufferBuffer,
  runtimeToFlatbufferBuffer,
} from "./ndsceneLegacyJson.js";

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
  commitOrder: string[] = [];
  activeCommitId: string | null = null;

  addCommit(commit: NDSceneCommitRuntime, setActive = false): void {
    if (!this.commitsById.has(commit.commitId)) {
      this.commitOrder.push(commit.commitId);
    }
    this.commitsById.set(commit.commitId, commit);
    if (setActive || this.activeCommitId === null) {
      this.activeCommitId = commit.commitId;
    }
  }

  getCommit(commitId: string): NDSceneCommitRuntime | undefined {
    return this.commitsById.get(commitId);
  }

  getOrderedCommits(): NDSceneCommitRuntime[] {
    return this.commitOrder
      .map((commitId) => this.getCommit(commitId))
      .filter((commit): commit is NDSceneCommitRuntime => Boolean(commit));
  }

  setActiveCommit(commitId: string): void {
    if (!this.commitsById.has(commitId)) {
      throw new Error(`Commit "${commitId}" is missing from the runtime.`);
    }
    this.activeCommitId = commitId;
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
    return runtimeToFlatbufferBuffer(this);
  }

  static fromFlatbufferBuffer(buffer: ArrayBuffer | ArrayBufferView): NDSceneRuntime {
    return runtimeFromFlatbufferBuffer(buffer);
  }
}
