from pathlib import Path
import tempfile
import unittest

from src.osf_inventory import inventory_node, write_inventory


class OSFInventoryTests(unittest.TestCase):
    def test_recursive_inventory_and_pagination(self):
        root = "https://api.osf.io/v2/nodes/test/files/osfstorage/"
        second = "https://api.example/root?page=2"
        folder = "https://api.example/folder"
        payloads = {
            root: {
                "data": [
                    {
                        "id": "folder-id",
                        "attributes": {"kind": "folder", "name": "Task 2"},
                        "relationships": {
                            "files": {"links": {"related": {"href": folder}}}
                        },
                    },
                    {
                        "id": "file-a",
                        "attributes": {
                            "kind": "file",
                            "name": "README.txt",
                            "materialized_path": "/README.txt",
                            "size": 10,
                        },
                        "links": {"download": "https://download/a"},
                    },
                ],
                "links": {"next": second},
            },
            second: {
                "data": [
                    {
                        "id": "file-b",
                        "attributes": {
                            "kind": "file",
                            "name": "sentences.txt",
                            "materialized_path": "/sentences.txt",
                            "size": 20,
                        },
                        "links": {"download": "https://download/b"},
                    }
                ],
                "links": {"next": None},
            },
            folder: {
                "data": [
                    {
                        "id": "file-c",
                        "attributes": {
                            "kind": "file",
                            "name": "resultsZAB_NR.mat",
                            "materialized_path": "/Task 2/resultsZAB_NR.mat",
                            "size": 30,
                        },
                        "links": {"download": "https://download/c"},
                    }
                ],
                "links": {"next": None},
            },
        }
        files = inventory_node("test", fetch_json=payloads.__getitem__)
        self.assertEqual({item.file_id for item in files}, {"file-a", "file-b", "file-c"})
        with tempfile.TemporaryDirectory() as directory:
            summary = write_inventory(files, "test", directory)
            self.assertTrue((Path(directory) / "files.csv").exists())
            self.assertTrue((Path(directory) / "inventory.json").exists())
        self.assertEqual(summary["file_count"], 3)
        self.assertEqual(summary["total_known_size_bytes"], 60)


if __name__ == "__main__":
    unittest.main()
