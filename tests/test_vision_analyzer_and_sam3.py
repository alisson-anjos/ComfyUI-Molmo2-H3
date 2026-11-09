import json
import os
import sys
from pathlib import Path
import unittest
import torch

if os.environ.get("COMFYUI_PATH"):
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
sys.path.insert(0, str(Path(__file__).parents[1]))

from comfyui_molmo_h3.nodes import (
    Molmo2VisionAnalyzer,
    MolmoSAM3Loader,
    MolmoSAM3Propagate,
    detect_cuts,
    shots_of,
    points_data,
    draw_points,
)
from comfyui_molmo_h3 import MolmoH3EncoderLoader


class VisionAnalyzerSchemaTest(unittest.TestCase):
    def test_analyzer_schema(self):
        schema = Molmo2VisionAnalyzer.define_schema()
        inputs = {x.id: x for x in schema.inputs}
        outputs = {x.id: x for x in schema.outputs}

        # Check required inputs
        self.assertIn("clip", inputs)
        self.assertIn("task", inputs)
        self.assertIn("prompt", inputs)
        self.assertIn("video", inputs)
        self.assertIn("image_1", inputs)
        self.assertIn("draw_points_preview", inputs)

        # Check outputs
        self.assertIn("response", outputs)
        self.assertIn("points", outputs)
        self.assertIn("subjects_json", outputs)
        self.assertIn("marked_video", outputs)
        self.assertIn("binding_vector", outputs)

    def test_points_formatting_matches_draw_points(self):
        # Verify points JSON output can be read by points_data
        sample_packet = {
            "points": [
                {"time_seconds": 0.5, "object_id": 1, "x": 0.3, "y": 0.4, "video": 1},
                {"time_seconds": 1.2, "object_id": 2, "x": 0.7, "y": 0.8, "video": 1},
            ],
            "description": "boxer",
            "frame_count": 48,
            "fps": 24.0
        }
        json_str = json.dumps(sample_packet)
        parsed = points_data(json_str, video_index=1, object_id=0)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["object_id"], 1)
        self.assertEqual(parsed[1]["object_id"], 2)


class SAM3NodesTest(unittest.TestCase):
    def test_sam3_loader_schema(self):
        schema = MolmoSAM3Loader.define_schema()
        inputs = {x.id: x for x in schema.inputs}
        outputs = {x.id: x for x in schema.outputs}
        self.assertIn("checkpoint", inputs)
        self.assertIn("sam_model", outputs)

    def test_sam3_propagate_schema(self):
        schema = MolmoSAM3Propagate.define_schema()
        inputs = {x.id: x for x in schema.inputs}
        outputs = {x.id: x for x in schema.outputs}
        self.assertIn("images", inputs)
        self.assertIn("sam_model", inputs)
        self.assertIn("points", inputs)
        self.assertIn("detect_camera_cuts", inputs)
        self.assertIn("mask", outputs)
        self.assertIn("preview", outputs)
        self.assertIn("report", outputs)

    def test_detect_cuts_and_shots(self):
        # 10 identical frames -> no cuts
        frames = torch.ones(10, 32, 32, 3)
        cuts = detect_cuts(frames)
        self.assertEqual(cuts, [])
        shots = shots_of(10, cuts)
        self.assertEqual(shots, [(0, 10)])

        # Sharp cut at frame 5
        frames[5:] = 0.0
        cuts = detect_cuts(frames)
        self.assertIn(5, cuts)
        shots = shots_of(10, cuts)
        self.assertEqual(shots, [(0, 5), (5, 10)])


class LoaderAutoDownloadDescriptionTest(unittest.TestCase):
    def test_loader_has_auto_download_description(self):
        self.assertIn("Auto Download", MolmoH3EncoderLoader.DESCRIPTION)
        self.assertIn("HuggingFace", MolmoH3EncoderLoader.DESCRIPTION)
        self.assertIn("downloaded automatically", MolmoH3EncoderLoader.OUTPUT_TOOLTIPS[0])


if __name__ == "__main__":
    unittest.main()
