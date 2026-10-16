"""Regression tests for the FLUX.3 bbox parsing in nodes.py.

nodes.py imports ComfyUI and torch at module level, so the functions under test are pulled out of the file with
``ast`` and run on their own; nothing here needs ComfyUI. Standard library only. Run from the repository root:

    python -m unittest discover -s tests -v
"""

import ast
import json
import math
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "nodes.py"
FUNCTIONS = {"parse_flux3_bbox", "_parse_row_bbox", "_flux3_clean_desc", "parse_flux3_bbox_payload"}
ASSIGNS = {"FLUX3_BBOX_GRID"}


def _load():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    ns = {"json": json, "math": math}
    for node in tree.body:
        keep = (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS) or (
            isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ASSIGNS for t in node.targets)
        )
        if keep:
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
    missing = (FUNCTIONS | ASSIGNS) - set(ns)
    if missing:
        raise AssertionError(f"not found in nodes.py: {sorted(missing)}")
    return ns


NS = _load()
parse = NS["parse_flux3_bbox"]
parse_payload = NS["parse_flux3_bbox_payload"]


class ValidBoxes(unittest.TestCase):
    def test_valid_boxes_are_returned_as_integers(self):
        cases = [
            ([100, 100, 800, 800], [100, 100, 800, 800]),
            ([0, 0, 1000, 1000], [0, 0, 1000, 1000]),
            ((150, 150, 850, 850), [150, 150, 850, 850]),
            ([100.0, 100.0, 800.0, 800.0], [100, 100, 800, 800]),  # integral floats from JSON
            ("[150, 150, 850, 850]", [150, 150, 850, 850]),  # the string the canvas writes
            ("150, 150, 850, 850", [150, 150, 850, 850]),
            ({"t": 1, "l": 2, "b": 3, "r": 4}, [1, 2, 3, 4]),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                out = parse(value)
                self.assertEqual(out, expected)
                self.assertTrue(all(type(v) is int for v in out))


class ExplicitNull(unittest.TestCase):
    def test_explicit_null_is_none(self):
        # null is meaningful in the edit table (src null = New row, tgt null = Remove row)
        for value in (None, "null", "NULL", " null ", ""):
            with self.subTest(value=value):
                self.assertIsNone(parse(value, "tgt_bbox"))


class MalformedBoxes(unittest.TestCase):
    def test_malformed_boxes_raise_instead_of_becoming_null(self):
        cases = [
            [0.1, 0.1, 0.5, 0.5],  # 0-1 fractions, used to round to a zero-area box and become null
            [0.1, 0.1, 0.9, 0.9],  # ... or to [0, 0, 1, 1]
            "[0.1, 0.1, 0.9, 0.9]",
            "[100, 100, 800,",  # truncated
            "[100, 100, 800, 8OO]",  # typo
            "not a box",
            "5",
            [100, 100, 800],  # 3 values
            [100, 100, 800, 800, 900],  # 5 values
            [],
            [float("nan"), 100, 800, 800],
            [100, 100, float("inf"), 800],
            [100, 100, float("-inf"), 800],
            "[NaN, 100, 800, 800]",
            [-1, 100, 800, 800],
            [100, 100, 1001, 800],
            [100, 100, 1500, 1200],  # used to be clamped
            [100.5, 100, 800, 800],
            [100, 100, 800, "abc"],
            [100, 100, 800, None],
            [True, 100, 800, 800],
            [100, 100, 100, 800],  # top == bottom
            [100, 100, 800, 100],  # left == right
            [800, 100, 100, 800],  # top > bottom
            [100, 800, 800, 100],  # left > right
            {"t": 1, "l": 2},
            {"top": 1, "left": 2, "bottom": 3, "right": 4},
            42,
            object(),
        ]
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse(value, "tgt_bbox")

    def test_error_names_the_field(self):
        with self.assertRaisesRegex(ValueError, "tgt_bbox"):
            parse([0.1, 0.1, 0.5, 0.5], "tgt_bbox")


def _editing(**row):
    return {"version": 1, "mode": "editing", "rows": [{"id": "cup_1", "desc": "a cup", "from": "ref_image_0", **row}]}


class RowLevel(unittest.TestCase):
    def test_malformed_target_does_not_become_a_remove_row(self):
        for tgt in ([0.1, 0.1, 0.5, 0.5], "[100, 100, 800,", [100, 100, 100, 100]):
            with self.subTest(tgt=tgt):
                with self.assertRaisesRegex(ValueError, "cup_1"):
                    parse_payload(_editing(src_bbox=[100, 100, 400, 400], tgt_bbox=tgt))

    def test_malformed_source_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "src_bbox"):
            parse_payload(_editing(src_bbox=[100, 100, 400], tgt_bbox=[200, 200, 500, 500]))

    def test_malformed_generation_bbox_is_an_error(self):
        payload = {"mode": "generation", "rows": [{"id": "sky_1", "desc": "sky", "bbox": [100, 100, 99, 800]}]}
        with self.assertRaisesRegex(ValueError, "sky_1"):
            parse_payload(payload)


class CanvasPayloadsKeepTheirMeaning(unittest.TestCase):
    def test_editing_payload_is_unchanged(self):
        payload = {
            "version": 1,
            "mode": "editing",
            "rows": [
                {"id": "keep_1", "desc": "stays", "from": "ref_image_0", "src_bbox": [0, 0, 1000, 1000], "tgt_bbox": [0, 0, 1000, 1000], "bbox": [0, 0, 1000, 1000]},
                {"id": "move_1", "desc": "moves", "from": "ref_image_0", "src_bbox": [500, 150, 850, 350], "tgt_bbox": [194, 55, 544, 255], "bbox": [194, 55, 544, 255]},
                {"id": "new_1", "desc": "new", "from": None, "src_bbox": None, "tgt_bbox": [550, 620, 850, 850], "bbox": [550, 620, 850, 850]},
                {"id": "gone_1", "desc": "removed", "from": "ref_image_0", "src_bbox": [200, 200, 400, 400], "tgt_bbox": None, "bbox": None},
            ],
        }
        rows, mode = parse_payload(payload)
        self.assertEqual(mode, "editing")
        self.assertEqual(
            [(r["id"], r["from"], r["src_bbox"], r["tgt_bbox"]) for r in rows],
            [
                ("keep_1", "ref_image_0", [0, 0, 1000, 1000], [0, 0, 1000, 1000]),
                ("move_1", "ref_image_0", [500, 150, 850, 350], [194, 55, 544, 255]),
                ("new_1", None, None, [550, 620, 850, 850]),
                ("gone_1", "ref_image_0", [200, 200, 400, 400], None),  # an explicit null is still a removal
            ],
        )
        self.assertEqual([r["_kind"] for r in rows], ["Keep", "Move", "New", "Remove"])

    def test_generation_payload_is_unchanged(self):
        payload = {"mode": "generation", "rows": [{"id": "bg_1", "desc": "backdrop", "bbox": [0, 0, 1000, 1000]}]}
        rows, mode = parse_payload(payload)
        self.assertEqual(mode, "generation")
        self.assertEqual(rows, [{"id": "bg_1", "bbox": [0, 0, 1000, 1000], "desc": "backdrop"}])


if __name__ == "__main__":
    unittest.main()
