import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import ndscenepy.ndscene as ndscene
from ndscenesql.ndscenesql import NDSceneSQLClient, write_example_database


class NDSceneSQLTests(unittest.TestCase):
    def test_freed_go_scene_render_update_commit(self):
        repo_root = Path(__file__).resolve().parents[2]
        scene_path = repo_root / "json" / "freed_go" / "view_1_scene.json"
        voxel_path = repo_root / "json" / "freed_go" / "voxels.json"

        voxel_scene = ndscene.NDJson.scene_from_path(str(voxel_path))
        voxels = voxel_scene.root.child_find("voxels")
        self.assertIsNotNone(voxels)

        scene = ndscene.NDJson.scene_from_path(str(scene_path))
        ndscene.NDMethod.setup_standard_methods(scene)
        world = scene.root.child_find("world")
        self.assertIsNotNone(world)
        world.child_add(voxels)

        camera = scene.root.child_find("camera", recursive=True)
        self.assertIsNotNone(camera)
        image_path = camera.child_find("image", recursive=True)
        self.assertIsNotNone(image_path)

        image_tensor = image_path.content.native_tensor(scene)
        before = image_tensor.clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path( repo_root / "py" / "examples" )
            db_path = os.path.join(temp_dir, "freed_go.sqlite")
            with NDSceneSQLClient(db_path) as client:
                commit_before = client.record_scene_commit(
                    scene_id="scene/freed_go/view_1",
                    scene=scene,
                    packet_path=str(scene_path),
                    packet_format="application/json",
                    commit_metadata={"stage": "before_render"},
                )

                renderer = ndscene.NDRender()
                renderer.update_object_from_world(image_path, scene)

                commit_after = client.record_scene_commit(
                    scene_id="scene/freed_go/view_1",
                    scene=scene,
                    packet_path=str(scene_path),
                    packet_format="application/json",
                    parent_commit_id=commit_before,
                    commit_metadata={"stage": "after_render"},
                )

                commit_detail = client.get_commit(commit_after)
                object_updates = client.list_object_updates(
                    scene_id="scene/freed_go/view_1",
                    object_id="image",
                    limit=16,
                )

            self.assertTrue(os.path.exists(db_path))

        self.assertNotEqual(commit_before, commit_after)
        self.assertIsNotNone(commit_detail)
        self.assertGreaterEqual(len(commit_detail["objects"]), 1)
        self.assertGreaterEqual(len(object_updates), 1)
        world_graph = next(
            entry for entry in commit_detail["object_graph"] if entry["object_id"] == "world"
        )
        self.assertIn("camera", world_graph["child_object_ids"])
        camera_graph = next(
            entry for entry in commit_detail["object_graph"] if entry["object_id"] == "camera"
        )
        self.assertEqual(camera_graph["parent_object_id"], "world")
        self.assertTrue((before != image_tensor).any().item())

    def test_example_database_population(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_ndscene.sqlite")
            written_path = write_example_database(db_path)

            self.assertEqual(written_path, db_path)
            self.assertTrue(os.path.exists(db_path))

            with NDSceneSQLClient(db_path, create_tables=False) as client:
                commits = client.list_commits(limit=16)
                self.assertGreaterEqual(len(commits), 5)

                commit_detail = client.get_commit(commits[0]["scene_commit_id"])
                self.assertIsNotNone(commit_detail)
                self.assertGreaterEqual(len(commit_detail["objects"]), 1)
                self.assertGreaterEqual(len(commit_detail["object_graph"]), 1)
                self.assertGreaterEqual(len(commit_detail["data"]), 1)

                approved = client.filter_commits_by_label(label_scene_id="scene/labels/approved")
                rejected = client.filter_commits_by_label(label_scene_id="scene/labels/rejected")
                self.assertEqual(len(approved), 2)
                self.assertEqual(len(rejected), 1)

            self.assertFalse(os.path.exists(os.path.join(temp_dir, "missing.sqlite")))

    def test_example_database_has_expected_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "schema_check.sqlite")
            write_example_database(db_path)

            conn = sqlite3.connect(db_path)
            try:
                names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            finally:
                conn.close()

            self.assertIn("NDSceneCommit", names)
            self.assertIn("NDObjectVersion", names)
            self.assertIn("NDModelLabel", names)


if __name__ == "__main__":
    unittest.main()
