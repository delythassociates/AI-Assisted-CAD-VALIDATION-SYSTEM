import unittest
from backend.core.models import PartMetadata, FaceGeometry
from backend.rules.die_casting import dc_draft_001, dc_draft_002
from backend.rules.injection import inj_undercut_001

class TestDraftRules(unittest.TestCase):
    def setUp(self):
        # Helper function to emulate CAD EstimateDraftAngle math
        import math
        self.estimate_draft = lambda nz: math.asin(min(abs(nz), 1.0)) * 180.0 / math.pi

    def test_draft_angles(self):
        # 1. Assert vertical wall (nz = 0) -> 0 degrees
        self.assertAlmostEqual(self.estimate_draft(0.0), 0.0, places=4)

        # 2. Assert horizontal face (nz = 1) -> 90 degrees
        self.assertAlmostEqual(self.estimate_draft(1.0), 90.0, places=4)

        # 3. Assert 5-degree draft face (nz ≈ 0.0871557) -> ~5 degrees
        self.assertAlmostEqual(self.estimate_draft(0.0871557), 5.0, places=1)

    def test_rules_guard_undercut_vs_draft(self):
        # Create a mock part metadata for aluminium die casting validation
        part = PartMetadata(
            filename="test_diecast.SLDPRT",
            process="die_cast_al",
            material="A380",
            bounding_box_mm="50x50x50",
            face_count=3,
            nominal_wall_mm=2.0,
            pull_direction="+Z",
            faces=[
                FaceGeometry(
                    face_id="vertical_wall",
                    face_type="Plane",
                    area_mm2=100.0,
                    thickness_mm=2.0,
                    normal_x=1.0,
                    normal_y=0.0,
                    normal_z=0.0,
                    draft_angle_deg=0.0, # vertical
                    sw_feature_name="Wall1",
                ),
                FaceGeometry(
                    face_id="drafted_5deg_wall",
                    face_type="Plane",
                    area_mm2=100.0,
                    thickness_mm=2.0,
                    normal_x=0.99619,
                    normal_y=0.0,
                    normal_z=0.08716,
                    draft_angle_deg=5.0, # 5 deg draft
                    sw_feature_name="Wall2",
                ),
                FaceGeometry(
                    face_id="undercut_wall",
                    face_type="Plane",
                    area_mm2=100.0,
                    thickness_mm=2.0,
                    normal_x=0.86603,
                    normal_y=0.0,
                    normal_z=-0.5, # opposing pull direction (Nz < 0)
                    draft_angle_deg=30.0, # Math.Abs(nz) magnitude is 0.5 (30 deg)
                    sw_feature_name="UndercutWall",
                )
            ]
        )

        # Run die-casting external draft check (DC-DRAFT-001)
        violations = dc_draft_001(part)
        v_ids = [v.face_ids[0] for v in violations if v.rule_id == "DC-DRAFT-001"]

        # 1. Vertical wall has 0 deg draft (which is < 1.0 deg ext_draft) -> must trigger violation
        self.assertIn("vertical_wall", v_ids)

        # 2. Drafted 5 deg wall has 5 deg draft (which is >= 1.0 deg ext_draft) -> must NOT trigger violation
        self.assertNotIn("drafted_5deg_wall", v_ids)

        # 3. Undercut wall has nz = -0.5 (opposing pull) -> must NOT trigger draft violation due to sign guard
        self.assertNotIn("undercut_wall", v_ids)

        # 4. Undercut wall must trigger undercut violation instead
        undercuts = inj_undercut_001(part)
        u_ids = [v.face_ids[0] for v in undercuts if v.rule_id == "INJ-UNDERCUT-001"]
        self.assertIn("undercut_wall", u_ids)

if __name__ == '__main__':
    unittest.main()
