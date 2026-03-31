import argparse
import base64
import contextlib
import datetime as _datetime
import hashlib
from importlib.resources import files
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from ndscenepy.ndscene import NDData, NDObject, NDScene, NDTensor


def _utc_now_iso() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _json_dumps(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_native_tensor(value) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return True
    return hasattr(value, "tolist") and hasattr(value, "shape")


def _tensor_to_jsonable(value):
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    shape = None
    if hasattr(value, "shape"):
        try:
            shape = [int(v) for v in value.shape]
        except Exception:
            shape = None
    dtype = None
    if hasattr(value, "dtype"):
        dtype = str(value.dtype)
    payload = value.tolist() if hasattr(value, "tolist") else repr(value)
    return {
        "shape": shape,
        "dtype": dtype,
        "data": payload,
    }


def _jsonable_data_blob(data: NDData):
    if data is None:
        return None
    buffer_value = data.buffer
    if isinstance(buffer_value, bytes):
        buffer_json = {"@bytes_b64": base64.b64encode(buffer_value).decode("ascii")}
    else:
        buffer_json = _to_jsonable(buffer_value)
    tensor_json = _tensor_to_jsonable(data.tensor) if _is_native_tensor(data.tensor) else _to_jsonable(data.tensor)
    return {
        "path": data.path,
        "format": data.format,
        "buffer": buffer_json,
        "tensor": tensor_json,
    }


def _jsonable_tensor(tensor: NDTensor):
    if tensor is None:
        return None
    return {
        "key": tensor.key,
        "size": tensor.size,
        "dtype": tensor.dtype,
        "shape": [_jsonable_tensor(child) for child in (tensor.shape or [])],
        "data": _jsonable_data_blob(tensor.data),
    }


def _jsonable_object(obj: NDObject):
    if obj is None:
        return None
    return {
        "name": obj.name,
        "pose": _to_jsonable(obj.pose),
        "unpose": _to_jsonable(obj.unpose),
        "content": _to_jsonable(obj.content),
        "children": [child.name for child in (obj.children or [])],
        "parents": [parent.name for parent in (obj.parents or [])],
    }


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"@bytes_b64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, NDData):
        return _jsonable_data_blob(value)
    if isinstance(value, NDTensor):
        return _jsonable_tensor(value)
    if isinstance(value, NDObject):
        return _jsonable_object(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if _is_native_tensor(value):
        return _tensor_to_jsonable(value)
    return repr(value)


def _object_identifier(obj: NDObject) -> str:
    if obj.name:
        return obj.name
    return "object_" + _sha256_text(_json_dumps(_jsonable_object(obj)))[:16]


def _example_database_default_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data" / "ndscenesql_example.sqlite"


def _schema_sql() -> str:
    return files("ndscenesql").joinpath("schema.sql").read_text(encoding="utf-8")


def _build_example_scene(version_index: int) -> NDScene:
    scene = NDScene()

    root = NDObject(key="world")
    camera = NDObject(
        key="camera",
        content=NDTensor.from_data(NDData.from_text(f"camera frame {version_index}")),
        scene=scene,
    )
    board = NDObject(
        key="board",
        content=NDTensor.from_data(NDData.from_text(f"board state {version_index}")),
        scene=scene,
    )
    label_hint = NDObject(
        key="label_hint",
        content=NDTensor.from_data(NDData.from_text(f"approved={1 if version_index % 2 == 0 else 0}")),
        scene=scene,
    )

    root.children = [camera, board, label_hint]
    camera.parents = [root]
    board.parents = [root]
    label_hint.parents = [root]

    scene.root = root
    scene.objects = {
        "world": root,
        "camera": camera,
        "board": board,
        "label_hint": label_hint,
    }
    scene.tensors = {
        "camera_text": camera.content,
        "board_text": board.content,
        "label_hint_text": label_hint.content,
    }
    return scene


class NDSceneSQLClient:
    """SQLite persistence layer for packet streams, scene commits, labels, and review queries."""

    def __init__(self, db_path: str = ":memory:", create_tables: bool = True):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if create_tables:
            self.create_tables()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.close()

    @contextlib.contextmanager
    def transaction(self):
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def create_tables(self):
        self.conn.executescript(_schema_sql())
        self.conn.commit()

    def _insert_ignore(self, sql: str, params):
        self.conn.execute(sql, params)

    def _persist_data(self, data: Optional[NDData]):
        if data is None:
            return None

        payload = _jsonable_data_blob(data)
        payload_json = _json_dumps(payload)
        content_hash = _sha256_text(payload_json)
        data_id = "data_" + content_hash[:24]

        buffer_blob = data.buffer if isinstance(data.buffer, bytes) else None
        tensor_cache_ref = None
        if payload.get("tensor") is not None:
            tensor_cache_ref = "tensor_" + _sha256_text(_json_dumps(payload["tensor"]))[:24]

        self._insert_ignore(
            """
            INSERT OR IGNORE INTO NDDataBlob(
                data_id, path, format, buffer, tensor_cache_ref, content_hash, extra_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data_id,
                data.path,
                data.format,
                buffer_blob,
                tensor_cache_ref,
                content_hash,
                payload_json,
                _utc_now_iso(),
            ),
        )
        return data_id

    def _persist_tensor(self, tensor: Optional[NDTensor], scene_commit_id: Optional[str] = None):
        if tensor is None:
            return None

        child_ids = []
        for child in tensor.shape or []:
            child_ids.append(self._persist_tensor(child, scene_commit_id=scene_commit_id))

        data_id = self._persist_data(tensor.data)
        payload = _jsonable_tensor(tensor)
        payload_json = _json_dumps(payload)
        content_hash = _sha256_text(payload_json)
        tensor_id = tensor.key or ("tensor_" + content_hash[:24])
        tensor_version_id = "tensorver_" + content_hash[:24]

        self._insert_ignore(
            """
            INSERT OR IGNORE INTO NDTensorVersion(
                tensor_version_id, tensor_id, key, size, dtype, data_id, shape_patch_parent,
                tensor_json, content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tensor_version_id,
                tensor_id,
                tensor.key,
                tensor.size,
                tensor.dtype,
                data_id,
                None,
                payload_json,
                content_hash,
                _utc_now_iso(),
            ),
        )

        for ordinal, child_tensor_version_id in enumerate(child_ids):
            self._insert_ignore(
                """
                INSERT OR IGNORE INTO NDTensorShapeEdge(
                    tensor_version_id, child_tensor_version_id, ordinal
                ) VALUES (?, ?, ?)
                """,
                (tensor_version_id, child_tensor_version_id, ordinal),
            )

        if scene_commit_id is not None:
            self._insert_ignore(
                """
                INSERT OR IGNORE INTO NDSceneCommitTensor(scene_commit_id, tensor_version_id)
                VALUES (?, ?)
                """,
                (scene_commit_id, tensor_version_id),
            )
            if data_id is not None:
                self._insert_ignore(
                    """
                    INSERT OR IGNORE INTO NDSceneCommitData(scene_commit_id, data_id)
                    VALUES (?, ?)
                    """,
                    (scene_commit_id, data_id),
                )

        return tensor_version_id

    def _persist_object_tree(self, obj: NDObject, scene_commit_id: str, memo: dict):
        memo_key = id(obj)
        if memo_key in memo:
            return memo[memo_key]

        object_id = _object_identifier(obj)
        pose_tensor_version_id = self._persist_tensor(obj.pose, scene_commit_id=scene_commit_id)
        unpose_tensor_version_id = self._persist_tensor(obj.unpose, scene_commit_id=scene_commit_id)
        content_tensor_version_id = self._persist_tensor(obj.content, scene_commit_id=scene_commit_id)

        payload = _jsonable_object(obj)
        payload_json = _json_dumps(payload)
        content_hash = _sha256_text(payload_json)
        version_id = "objver_" + _sha256_text(scene_commit_id + ":" + object_id + ":" + content_hash)[:24]

        self._insert_ignore(
            """
            INSERT OR IGNORE INTO NDObjectVersion(
                version_id, object_id, scene_commit_id, state, version_patch_parent,
                pose_tensor_version_id, unpose_tensor_version_id, content_tensor_version_id,
                object_json, content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                object_id,
                scene_commit_id,
                payload_json,
                None,
                pose_tensor_version_id,
                unpose_tensor_version_id,
                content_tensor_version_id,
                payload_json,
                content_hash,
                _utc_now_iso(),
            ),
        )
        self._insert_ignore(
            """
            INSERT OR IGNORE INTO NDSceneCommitObject(scene_commit_id, object_version_id)
            VALUES (?, ?)
            """,
            (scene_commit_id, version_id),
        )

        memo[memo_key] = version_id

        for ordinal, child in enumerate(obj.children or []):
            child_version_id = self._persist_object_tree(child, scene_commit_id, memo)
            self._insert_ignore(
                """
                INSERT OR IGNORE INTO NDObjectEdge(
                    object_version_id, related_object_version_id, relation_type, ordinal
                ) VALUES (?, ?, ?, ?)
                """,
                (version_id, child_version_id, "child", ordinal),
            )

        for ordinal, parent in enumerate(obj.parents or []):
            parent_version_id = self._persist_object_tree(parent, scene_commit_id, memo)
            self._insert_ignore(
                """
                INSERT OR IGNORE INTO NDObjectEdge(
                    object_version_id, related_object_version_id, relation_type, ordinal
                ) VALUES (?, ?, ?, ?)
                """,
                (version_id, parent_version_id, "parent", ordinal),
            )

        return version_id

    def record_scene_commit(
        self,
        scene_id: str,
        scene: Optional[NDScene] = None,
        packet_data: Optional[bytes] = None,
        packet_format: Optional[str] = None,
        packet_path: Optional[str] = None,
        is_external: bool = False,
        parent_commit_id: Optional[str] = None,
        commit_metadata: Optional[dict] = None,
    ) -> str:
        scene_commit_id = "commit_" + uuid.uuid4().hex
        commit_json = _json_dumps(_to_jsonable(commit_metadata or {}))

        packet_blob = None
        packet_data_id = None
        if packet_data is not None:
            packet_blob = packet_data.encode("utf-8") if isinstance(packet_data, str) else packet_data
            packet_data_obj = NDData(buffer=packet_blob, format=packet_format, path=packet_path)
            packet_data_id = self._persist_data(packet_data_obj)

        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO NDSceneCommit(
                    scene_commit_id, scene_id, packet_data, packet_format, packet_path,
                    packet_data_id, is_external, parent_commit_id, commit_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_commit_id,
                    scene_id,
                    packet_blob,
                    packet_format,
                    packet_path,
                    packet_data_id,
                    1 if is_external else 0,
                    parent_commit_id,
                    commit_json,
                    _utc_now_iso(),
                ),
            )

            if packet_data_id is not None:
                self._insert_ignore(
                    """
                    INSERT OR IGNORE INTO NDSceneCommitData(scene_commit_id, data_id)
                    VALUES (?, ?)
                    """,
                    (scene_commit_id, packet_data_id),
                )

            if scene is not None:
                memo = {}
                if scene.root is not None:
                    self._persist_object_tree(scene.root, scene_commit_id, memo)
                for obj in (scene.objects or {}).values():
                    self._persist_object_tree(obj, scene_commit_id, memo)
                for tensor in (scene.tensors or {}).values():
                    self._persist_tensor(tensor, scene_commit_id=scene_commit_id)

        return scene_commit_id

    def list_commits(self, scene_id: Optional[str] = None, limit: int = 100):
        sql = """
            SELECT scene_commit_id, scene_id, packet_format, packet_path, packet_data_id,
                   is_external, parent_commit_id, commit_json, created_at
            FROM NDSceneCommit
        """
        params = []
        if scene_id is not None:
            sql += " WHERE scene_id = ?"
            params.append(scene_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_commit(self, scene_commit_id: str):
        commit = self.conn.execute(
            "SELECT * FROM NDSceneCommit WHERE scene_commit_id = ?",
            (scene_commit_id,),
        ).fetchone()
        if commit is None:
            return None

        objects = self.conn.execute(
            """
            SELECT ov.*
            FROM NDObjectVersion ov
            JOIN NDSceneCommitObject sco ON sco.object_version_id = ov.version_id
            WHERE sco.scene_commit_id = ?
            ORDER BY ov.object_id
            """,
            (scene_commit_id,),
        ).fetchall()

        tensors = self.conn.execute(
            """
            SELECT tv.*
            FROM NDTensorVersion tv
            JOIN NDSceneCommitTensor sct ON sct.tensor_version_id = tv.tensor_version_id
            WHERE sct.scene_commit_id = ?
            ORDER BY tv.tensor_id
            """,
            (scene_commit_id,),
        ).fetchall()

        data_rows = self.conn.execute(
            """
            SELECT db.*
            FROM NDDataBlob db
            JOIN NDSceneCommitData scd ON scd.data_id = db.data_id
            WHERE scd.scene_commit_id = ?
            ORDER BY db.data_id
            """,
            (scene_commit_id,),
        ).fetchall()

        labels = self.conn.execute(
            """
            SELECT *
            FROM NDModelLabel
            WHERE input_commit_id = ? OR context_commit_id = ? OR label_commit_id = ?
            ORDER BY created_at DESC
            """,
            (scene_commit_id, scene_commit_id, scene_commit_id),
        ).fetchall()

        return {
            "commit": dict(commit),
            "objects": [dict(row) for row in objects],
            "tensors": [dict(row) for row in tensors],
            "data": [dict(row) for row in data_rows],
            "labels": [dict(row) for row in labels],
        }

    def list_object_updates(self, scene_id: Optional[str] = None, object_id: Optional[str] = None, limit: int = 200):
        sql = """
            SELECT ov.version_id, ov.object_id, ov.scene_commit_id, ov.object_json, ov.created_at, sc.scene_id
            FROM NDObjectVersion ov
            JOIN NDSceneCommit sc ON sc.scene_commit_id = ov.scene_commit_id
        """
        where = []
        params = []
        if scene_id is not None:
            where.append("sc.scene_id = ?")
            params.append(scene_id)
        if object_id is not None:
            where.append("ov.object_id = ?")
            params.append(object_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ov.created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def add_label(
        self,
        label_commit_id: str,
        input_commit_id: Optional[str] = None,
        context_commit_id: Optional[str] = None,
        model_id: str = "review",
        event_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        label_row = self.conn.execute(
            "SELECT scene_id FROM NDSceneCommit WHERE scene_commit_id = ?",
            (label_commit_id,),
        ).fetchone()
        if label_row is None:
            raise ValueError(f"Unknown label_commit_id: {label_commit_id}")

        label_id = "label_" + uuid.uuid4().hex
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO NDModelLabel(
                    label_id, model_id, event_id, context_commit_id, input_commit_id,
                    label_commit_id, label_scene_id, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label_id,
                    model_id,
                    event_id,
                    context_commit_id,
                    input_commit_id,
                    label_commit_id,
                    label_row["scene_id"],
                    notes,
                    _utc_now_iso(),
                ),
            )
        return label_id

    def filter_commits_by_label(
        self,
        label_scene_id: Optional[str] = None,
        label_commit_id: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 100,
    ):
        sql = """
            SELECT DISTINCT sc.scene_commit_id, sc.scene_id, sc.packet_format, sc.packet_path,
                            sc.is_external, sc.parent_commit_id, sc.commit_json, sc.created_at
            FROM NDSceneCommit sc
            JOIN NDModelLabel ml
              ON ml.input_commit_id = sc.scene_commit_id
              OR ml.context_commit_id = sc.scene_commit_id
        """
        where = []
        params = []
        if label_scene_id is not None:
            where.append("ml.label_scene_id = ?")
            params.append(label_scene_id)
        if label_commit_id is not None:
            where.append("ml.label_commit_id = ?")
            params.append(label_commit_id)
        if model_id is not None:
            where.append("ml.model_id = ?")
            params.append(model_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY sc.created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def create_edit_session(
        self,
        view_commit_id: str,
        user_id: Optional[str] = None,
        label_range_start: Optional[str] = None,
        label_range_end: Optional[str] = None,
        ui_state: Optional[dict] = None,
    ) -> str:
        session_id = "edit_" + uuid.uuid4().hex
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO NDEditSession(
                    session_id, user_id, view_commit_id, label_range_start,
                    label_range_end, ui_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    view_commit_id,
                    label_range_start,
                    label_range_end,
                    _json_dumps(_to_jsonable(ui_state or {})),
                    _utc_now_iso(),
                ),
            )
        return session_id

    def list_labels_for_commit(self, commit_id: str):
        rows = self.conn.execute(
            """
            SELECT *
            FROM NDModelLabel
            WHERE input_commit_id = ? OR context_commit_id = ? OR label_commit_id = ?
            ORDER BY created_at DESC
            """,
            (commit_id, commit_id, commit_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_inference_event(
        self,
        model_id: str,
        target_object_id: Optional[str] = None,
        scene_commit_id: Optional[str] = None,
        input_commit_id: Optional[str] = None,
        output_commit_id: Optional[str] = None,
        trace: Optional[list[dict]] = None,
    ) -> str:
        event_id = "event_" + uuid.uuid4().hex
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO NDInferenceEvent(
                    event_id, model_id, target_object_id, scene_commit_id,
                    input_commit_id, output_commit_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    model_id,
                    target_object_id,
                    scene_commit_id,
                    input_commit_id,
                    output_commit_id,
                    _utc_now_iso(),
                ),
            )
            for op_index, op in enumerate(trace or []):
                op_type = op.get("op_type", "op")
                op_data = _json_dumps(_to_jsonable(op))
                self.conn.execute(
                    """
                    INSERT INTO NDInferenceTrace(event_id, op_index, op_type, op_data)
                    VALUES (?, ?, ?, ?)
                    """,
                    (event_id, op_index, op_type, op_data),
                )
        return event_id


def write_example_database(db_path: Optional[str] = None) -> str:
    output_path = Path(db_path) if db_path else _example_database_default_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar_path = Path(str(output_path) + suffix)
        if sidecar_path.exists():
            sidecar_path.unlink()

    with NDSceneSQLClient(str(output_path)) as client:
        main_scene_commit_ids = []
        for version_index in range(3):
            scene = _build_example_scene(version_index)
            commit_id = client.record_scene_commit(
                scene_id="scene/main/review_demo",
                scene=scene,
                packet_data=f"packet {version_index}",
                packet_format="text/plain",
                packet_path=f"packets/{version_index:03d}.txt",
                is_external=(version_index == 0),
                commit_metadata={"sequence_index": version_index},
            )
            main_scene_commit_ids.append(commit_id)

        approved_scene = _build_example_scene(100)
        rejected_scene = _build_example_scene(101)
        approved_commit_id = client.record_scene_commit(
            scene_id="scene/labels/approved",
            scene=approved_scene,
            packet_data="approved label packet",
            packet_format="text/plain",
            packet_path="labels/approved.txt",
            commit_metadata={"label": "approved"},
        )
        rejected_commit_id = client.record_scene_commit(
            scene_id="scene/labels/rejected",
            scene=rejected_scene,
            packet_data="rejected label packet",
            packet_format="text/plain",
            packet_path="labels/rejected.txt",
            commit_metadata={"label": "rejected"},
        )

        for commit_index, input_commit_id in enumerate(main_scene_commit_ids):
            label_commit_id = approved_commit_id if commit_index != 1 else rejected_commit_id
            label_name = "approved" if commit_index != 1 else "rejected"
            event_id = client.record_inference_event(
                model_id="demo-model",
                target_object_id="board",
                input_commit_id=input_commit_id,
                output_commit_id=label_commit_id,
                trace=[
                    {"op_type": "load_packet", "sequence_index": commit_index},
                    {"op_type": "predict_label", "label": label_name},
                ],
            )
            client.add_label(
                label_commit_id=label_commit_id,
                input_commit_id=input_commit_id,
                model_id="demo-review",
                event_id=event_id,
                notes=f"seed label {label_name} for commit {commit_index}",
            )

        client.create_edit_session(
            view_commit_id=main_scene_commit_ids[0],
            user_id="example-user",
            label_range_start=main_scene_commit_ids[0],
            label_range_end=main_scene_commit_ids[-1],
            ui_state={"selected_object": "board", "filter": "all"},
        )

    checkpoint_conn = sqlite3.connect(str(output_path))
    try:
        checkpoint_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        checkpoint_conn.execute("PRAGMA journal_mode = DELETE")
    finally:
        checkpoint_conn.close()

    for suffix in ("-wal", "-shm"):
        sidecar_path = Path(str(output_path) + suffix)
        if sidecar_path.exists():
            sidecar_path.unlink()

    return str(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create or refresh an example ndscene SQLite database.")
    parser.add_argument(
        "--write-example-db",
        dest="db_path",
        nargs="?",
        const=str(_example_database_default_path()),
        help="Write the example database. Optionally provide a custom output path.",
    )
    args = parser.parse_args(argv)

    db_path = args.db_path or str(_example_database_default_path())
    written_path = write_example_database(db_path)
    print(written_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
