import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import ndscenepy.ndscene as ndscene
from ndscenesql.ndscenesql import NDSceneSQLClient, _schema_sql


def _default_db_path() -> Path:
    return Path(os.environ.get("NDSCENE_DB_PATH", REPO_ROOT / "py" / "examples" / "freed_go.sqlite"))


def _blob_summary(value: Optional[bytes]) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    return {"byteLength": len(value)}


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_row(row: Union[sqlite3.Row, dict[str, Any]]) -> dict[str, Any]:
    source = dict(row)
    normalized: dict[str, Any] = {}
    for key, value in source.items():
        if isinstance(value, bytes):
            normalized[key] = _blob_summary(value)
        elif key.endswith("_json") and value is not None:
            normalized[key] = _parse_json_value(value)
        else:
            normalized[key] = value
    return normalized


def _timeline_rows(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT sc.scene_commit_id,
                   sc.scene_id,
                   sc.packet_format,
                   sc.packet_path,
                   sc.is_external,
                   sc.parent_commit_id,
                   sc.commit_json,
                   sc.created_at,
                   COUNT(DISTINCT sco.object_version_id) AS object_count,
                   COUNT(DISTINCT sct.tensor_version_id) AS tensor_count,
                   COUNT(DISTINCT scd.data_id) AS data_count
            FROM NDSceneCommit sc
            LEFT JOIN NDSceneCommitObject sco ON sco.scene_commit_id = sc.scene_commit_id
            LEFT JOIN NDSceneCommitTensor sct ON sct.scene_commit_id = sc.scene_commit_id
            LEFT JOIN NDSceneCommitData scd ON scd.scene_commit_id = sc.scene_commit_id
            GROUP BY sc.scene_commit_id
            ORDER BY sc.created_at DESC
            """
        ).fetchall()
        return [_normalize_row(row) for row in rows]
    finally:
        conn.close()


def _render_freed_go_view_to_database(db_path: Path) -> dict[str, Any]:
    scene_path = REPO_ROOT / "json" / "freed_go" / "view_1_scene.json"
    voxel_path = REPO_ROOT / "json" / "freed_go" / "voxels.json"

    voxel_scene = ndscene.NDJson.scene_from_path(str(voxel_path))
    voxels = voxel_scene.root.child_find("voxels")
    scene = ndscene.NDJson.scene_from_path(str(scene_path))
    ndscene.NDMethod.setup_standard_methods(scene)

    world = scene.root.child_find("world")
    world.child_add(voxels)
    camera = scene.root.child_find("camera", recursive=True)
    image_path = camera.child_find("image", recursive=True)
    image_tensor = image_path.content.native_tensor(scene)
    before_sum = float(image_tensor.sum().item())

    with NDSceneSQLClient(str(db_path), create_tables=True) as client:
        before_commit_id = client.record_scene_commit(
            scene_id="scene/freed_go/view_1",
            scene=scene,
            packet_path=str(scene_path),
            packet_format="application/json",
            commit_metadata={"stage": "before_render", "source": "server"},
        )
        renderer = ndscene.NDRender()
        renderer.update_object_from_world(image_path, scene)
        after_sum = float(image_tensor.sum().item())
        after_commit_id = client.record_scene_commit(
            scene_id="scene/freed_go/view_1",
            scene=scene,
            packet_path=str(scene_path),
            packet_format="application/json",
            parent_commit_id=before_commit_id,
            commit_metadata={"stage": "after_render", "source": "server"},
        )

    return {
        "databasePath": str(db_path),
        "scenePath": str(scene_path),
        "beforeCommitId": before_commit_id,
        "afterCommitId": after_commit_id,
        "beforeImageSum": before_sum,
        "afterImageSum": after_sum,
        "imageChanged": before_sum != after_sum,
    }


app = FastAPI(title="ndscene server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    db_path = _default_db_path()
    return {
        "ok": True,
        "databasePath": str(db_path),
        "databaseExists": db_path.exists(),
    }


@app.get("/api/schema.sql", response_class=PlainTextResponse)
def schema_sql():
    return _schema_sql()


@app.get("/api/commits")
def list_commits():
    db_path = _default_db_path()
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"Database not found: {db_path}")
    return {
        "databasePath": str(db_path),
        "commits": _timeline_rows(db_path),
    }


@app.get("/api/commits/{scene_commit_id}")
def get_commit(scene_commit_id: str):
    db_path = _default_db_path()
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"Database not found: {db_path}")

    with NDSceneSQLClient(str(db_path), create_tables=False) as client:
        detail = client.get_commit(scene_commit_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Commit not found: {scene_commit_id}")

    return {
        "databasePath": str(db_path),
        "detail": {
            "commit": _normalize_row(detail["commit"]),
            "objects": [_normalize_row(row) for row in detail["objects"]],
            "object_edges": [_normalize_row(row) for row in detail["object_edges"]],
            "object_graph": [_normalize_row(row) for row in detail["object_graph"]],
            "tensors": [_normalize_row(row) for row in detail["tensors"]],
            "data": [_normalize_row(row) for row in detail["data"]],
            "labels": [_normalize_row(row) for row in detail["labels"]],
        },
    }


@app.post("/api/render/freed-go/view-1")
def render_freed_go_view():
    db_path = _default_db_path()
    return _render_freed_go_view_to_database(db_path)
