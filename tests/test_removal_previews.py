"""Real preview encoding, incremental publication, and workflow isolation."""
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import torch

spec = importlib.util.spec_from_file_location(
    "removal_previews", Path(__file__).parents[1] / "h3_relay/removal_previews.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PreviewTests(unittest.TestCase):
    def test_incremental_videos_and_isolated_run_history(self):
        with tempfile.TemporaryDirectory() as temp:
            events = []
            modules = {
                "folder_paths": SimpleNamespace(get_temp_directory=lambda: temp),
                "server": SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(
                    send_sync=lambda event, data: events.append((event, data))))),
                "comfy_execution.utils": SimpleNamespace(
                    get_executing_context=lambda: SimpleNamespace(prompt_id="job-a")),
            }
            with patch.dict("sys.modules", modules):
                run = m.begin_run("70", {"workflow": {"id": "workflow-a"}}, [0, 21], 22, 43, 904234)
                frames = torch.zeros(22, 40, 80, 3)
                frames[..., 1] = .8
                m.publish_window(run, 0, frames, 0, 0)
                first = m.latest_run("workflow-a", "70")
                self.assertEqual(first["status"], "rendering")
                self.assertEqual(len(first["segments"]), 1)
                self.assertEqual(first["prompt_id"], "job-a")
                self.assertIsNone(m.latest_run("workflow-b", "70"))
                self.assertIsNone(m.latest_run("workflow-b", "70", run))
                self.assertIsNone(m.latest_run("workflow-a", "70", "../../outside"))
                item = first["segments"][0]["video"]
                video = Path(temp) / item["subfolder"] / item["filename"]
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    self.assertEqual(float(stream.average_rate), 24.)
                    decoded = list(container.decode(video=0))
                    self.assertEqual(len(decoded), 22)
                    self.assertEqual((decoded[0].width, decoded[0].height), (80, 40))
                    self.assertGreater(decoded[0].to_ndarray(format="rgb24")[..., 1].mean(), 195)
                m.publish_window(run, 1, frames, 21, 1)
                final = m.latest_run("workflow-a", "70")
                self.assertEqual(final["status"], "complete")
                self.assertEqual([s["index"] for s in final["segments"]], [0, 1])
                self.assertEqual(len(events[1][1]["segments"]), 1)  # Event snapshots do not mutate later.
                next_run = m.begin_run("70", {"workflow": {"id": "workflow-a"}}, [0], 39, 39, 904235)
                self.assertNotEqual(next_run, run)
                self.assertEqual(m.latest_run("workflow-a", "70")["segments"], [])
                self.assertTrue(video.exists())  # Rerunning doesn't delete prior completed previews.
                self.assertEqual(len(m.latest_run("workflow-a", "70", run)["segments"]), 2)
                with self.assertLogs(m._LOG, level="WARNING"), patch.object(m, "encode_preview", side_effect=RuntimeError("encoder unavailable")):
                    m.publish_window(next_run, 0, frames, 0, 0)
                self.assertIn("error", m.latest_run("workflow-a", "70")["segments"][0])


if __name__ == "__main__":
    unittest.main()
