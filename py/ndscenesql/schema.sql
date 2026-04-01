CREATE TABLE IF NOT EXISTS NDDataBlob(
    data_id TEXT PRIMARY KEY,
    path TEXT,
    format TEXT,
    buffer BLOB,
    tensor_cache_ref TEXT,
    content_hash TEXT NOT NULL,
    extra_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS NDTensorVersion(
    tensor_version_id TEXT PRIMARY KEY,
    tensor_id TEXT NOT NULL,
    parent_tensor_id TEXT,
    key TEXT,
    size INTEGER,
    dtype TEXT,
    data_id TEXT,
    tensor_json TEXT,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(data_id) REFERENCES NDDataBlob(data_id)
);

CREATE TABLE IF NOT EXISTS NDSceneCommit(
    scene_commit_id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    packet_data BLOB,
    packet_format TEXT,
    packet_path TEXT,
    packet_data_id TEXT,
    is_external INTEGER NOT NULL DEFAULT 0,
    parent_commit_id TEXT,
    commit_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(packet_data_id) REFERENCES NDDataBlob(data_id),
    FOREIGN KEY(parent_commit_id) REFERENCES NDSceneCommit(scene_commit_id)
);

CREATE TABLE IF NOT EXISTS NDObjectVersion(
    version_id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL,
    parent_object_id TEXT,
    scene_commit_id TEXT NOT NULL,
    state TEXT,
    version_patch_parent TEXT,
    pose_tensor_version_id TEXT,
    unpose_tensor_version_id TEXT,
    content_tensor_version_id TEXT,
    object_json TEXT,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(scene_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(version_patch_parent) REFERENCES NDObjectVersion(version_id),
    FOREIGN KEY(pose_tensor_version_id) REFERENCES NDTensorVersion(tensor_version_id),
    FOREIGN KEY(unpose_tensor_version_id) REFERENCES NDTensorVersion(tensor_version_id),
    FOREIGN KEY(content_tensor_version_id) REFERENCES NDTensorVersion(tensor_version_id)
);

CREATE TABLE IF NOT EXISTS NDSceneCommitObject(
    scene_commit_id TEXT NOT NULL,
    object_version_id TEXT NOT NULL,
    PRIMARY KEY(scene_commit_id, object_version_id),
    FOREIGN KEY(scene_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(object_version_id) REFERENCES NDObjectVersion(version_id)
);

CREATE TABLE IF NOT EXISTS NDSceneCommitTensor(
    scene_commit_id TEXT NOT NULL,
    tensor_version_id TEXT NOT NULL,
    PRIMARY KEY(scene_commit_id, tensor_version_id),
    FOREIGN KEY(scene_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(tensor_version_id) REFERENCES NDTensorVersion(tensor_version_id)
);

CREATE TABLE IF NOT EXISTS NDSceneCommitData(
    scene_commit_id TEXT NOT NULL,
    data_id TEXT NOT NULL,
    PRIMARY KEY(scene_commit_id, data_id),
    FOREIGN KEY(scene_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(data_id) REFERENCES NDDataBlob(data_id)
);

CREATE TABLE IF NOT EXISTS NDInferenceEvent(
    event_id TEXT PRIMARY KEY,
    model_id TEXT,
    target_object_id TEXT,
    scene_commit_id TEXT,
    input_commit_id TEXT,
    output_commit_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(scene_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(input_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(output_commit_id) REFERENCES NDSceneCommit(scene_commit_id)
);

CREATE TABLE IF NOT EXISTS NDInferenceTrace(
    event_id TEXT NOT NULL,
    op_index INTEGER NOT NULL,
    op_type TEXT NOT NULL,
    op_data TEXT,
    PRIMARY KEY(event_id, op_index),
    FOREIGN KEY(event_id) REFERENCES NDInferenceEvent(event_id)
);

CREATE TABLE IF NOT EXISTS NDModelLabel(
    label_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    event_id TEXT,
    context_commit_id TEXT,
    input_commit_id TEXT,
    label_commit_id TEXT NOT NULL,
    label_scene_id TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(context_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(input_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(label_commit_id) REFERENCES NDSceneCommit(scene_commit_id),
    FOREIGN KEY(event_id) REFERENCES NDInferenceEvent(event_id)
);

CREATE TABLE IF NOT EXISTS NDEditSession(
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    view_commit_id TEXT NOT NULL,
    label_range_start TEXT,
    label_range_end TEXT,
    ui_state TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(view_commit_id) REFERENCES NDSceneCommit(scene_commit_id)
);

CREATE INDEX IF NOT EXISTS idx_nddatablob_hash ON NDDataBlob(content_hash);
CREATE INDEX IF NOT EXISTS idx_ndtensorversion_tensor_id ON NDTensorVersion(tensor_id);
CREATE INDEX IF NOT EXISTS idx_ndtensorversion_parent_tensor_id ON NDTensorVersion(parent_tensor_id);
CREATE INDEX IF NOT EXISTS idx_ndobjectversion_object_scene ON NDObjectVersion(object_id, scene_commit_id);
CREATE INDEX IF NOT EXISTS idx_ndobjectversion_parent_object_id ON NDObjectVersion(parent_object_id);
CREATE INDEX IF NOT EXISTS idx_ndscenecommit_scene_created ON NDSceneCommit(scene_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ndmodellabel_input ON NDModelLabel(input_commit_id);
CREATE INDEX IF NOT EXISTS idx_ndmodellabel_label_scene ON NDModelLabel(label_scene_id);
