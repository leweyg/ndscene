import os
import sqlite3
import tempfile
import unittest

from ndscenesql.ndscenesql import NDSceneSQLClient, write_example_database


class NDSceneSQLTests(unittest.TestCase):
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
