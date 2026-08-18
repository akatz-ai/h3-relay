import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3_relay_staged_standalone", ROOT / "h3_relay" / "staged.py"
)
STAGED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGED)


def node(class_type, **inputs):
    return {
        "class_type": class_type,
        "inputs": inputs,
        "_meta": {"title": class_type},
    }


class StagedPlanTest(unittest.TestCase):
    def test_orders_all_h3_before_per_shot_finishing(self):
        prompt = {
            "1": node("H3RelaySequenceStart"),
            "10": node("H3RelayGenerateShot", sequence=["1", 0]),
            "11": node("H3RelayGenerateShot", sequence=["10", 0]),
            "20": node("H3RelayEnhanceShot", sequence=["10", 0]),
            "30": node("H3RelayInterpolateShot", enhanced=["20", 0]),
            "21": node(
                "H3RelayEnhanceShot",
                sequence=["11", 0],
                previous_enhanced=["30", 0],
            ),
            "31": node("H3RelayInterpolateShot", enhanced=["21", 0]),
            "40": node("H3RelayAssemble", enhanced=["31", 0]),
        }
        plan = STAGED.build_stage_plan(prompt, "40")
        self.assertEqual(
            [item["node_id"] for item in plan],
            ["10", "11", "20", "30", "21", "31", "40"],
        )
        self.assertEqual(
            [item["kind"] for item in plan],
            ["h3", "h3", "ltx", "rife", "ltx", "rife", "assemble"],
        )
        self.assertEqual(
            [item["shot_index"] for item in plan],
            ["1", "2", "1", "1", "2", "2", "2"],
        )
        rewritten = STAGED.rewrite_stage_with_disk_restores(
            prompt, plan[4], "restore_contract"
        )
        self.assertEqual(
            rewritten["21"]["inputs"]["sequence"],
            ["h3_relay_restore_raw_0002", 0],
        )
        self.assertEqual(
            rewritten["21"]["inputs"]["previous_enhanced"],
            ["h3_relay_restore_interpolated_0001", 0],
        )
        self.assertEqual(
            rewritten["h3_relay_restore_interpolated_0001"]["inputs"]["delivery_count"],
            1,
        )
        self.assertEqual(
            rewritten["h3_relay_restore_raw_0002"]["class_type"],
            "H3RelayInternalRestoreRawSequence",
        )

    def test_requires_assemble_target(self):
        with self.assertRaisesRegex(ValueError, "Assemble"):
            STAGED.build_stage_plan({"1": node("H3RelayGenerateShot")}, "1")

    def test_ltx_only_chain_restores_zero_deliveries(self):
        prompt = {
            "1": node("H3RelaySequenceStart", run_name="ltx_only"),
            "10": node("H3RelayGenerateShot", sequence=["1", 0]),
            "11": node("H3RelayGenerateShot", sequence=["10", 0]),
            "20": node("H3RelayEnhanceShot", sequence=["10", 0]),
            "21": node(
                "H3RelayEnhanceShot",
                sequence=["11", 0],
                previous_enhanced=["20", 0],
            ),
            "30": node("H3RelayAssemble", enhanced=["21", 0]),
        }
        plan = STAGED.build_stage_plan(prompt, "30")
        self.assertEqual([item["kind"] for item in plan], [
            "h3", "h3", "ltx", "ltx", "assemble",
        ])
        self.assertEqual([item["delivery_count"] for item in plan], [
            "0", "0", "0", "0", "0",
        ])
        rewritten = STAGED.rewrite_stage_with_disk_restores(
            prompt, plan[3], "ltx_only"
        )
        restore = rewritten["h3_relay_restore_ltx_0001"]
        self.assertEqual(restore["inputs"]["delivery_count"], 0)


if __name__ == "__main__":
    unittest.main()
