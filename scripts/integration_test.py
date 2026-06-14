"""End-to-end integration test for Eureka 3.0 DFM API."""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("EUREKA_BACKEND_URL", "http://localhost:8001")
API_KEY = os.environ.get("EUREKA_API_KEY", "eureka-dev-key-change-me")


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())


def post(path, body, api_key=API_KEY):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def post_expect_status(path, body, expected_status, api_key=API_KEY):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            raise AssertionError(f"Expected HTTP {expected_status}, got {r.status}")
    except urllib.error.HTTPError as e:
        if e.code != expected_status:
            raise AssertionError(
                f"Expected HTTP {expected_status}, got {e.code}: {e.read().decode()[:500]}"
            )
        return e.code


# Gate 3: Health check
print("=== Gate 3: Health ===")
health = get("/health")
print(json.dumps(health, indent=2))
assert health.get("status") == "ok", "Health check failed"

# Gate 7: Auth middleware — POST without API key → 401
print("\n=== Gate 7: Auth (no API key) ===")
minimal_part = {
    "filename": "test.SLDPRT",
    "process": "injection_moulding",
    "material": "ABS",
    "bounding_box_mm": "10x10x10",
    "face_count": 1,
    "nominal_wall_mm": 2.0,
    "faces": [
        {
            "face_id": "1",
            "face_type": "Plane",
            "area_mm2": 100,
            "thickness_mm": 2.0,
            "draft_angle_deg": 1.0,
            "radius_mm": 0.2,
        }
    ],
}
status = post_expect_status("/validate", minimal_part, 401, api_key=None)
print(f"POST /validate without API key -> {status} (expected 401)")

# Gate 8: Face limit — POST with 501 faces → 422
print("\n=== Gate 8: Face limit (501 faces) ===")
large_part = {
    "filename": "large.SLDPRT",
    "process": "injection_moulding",
    "material": "ABS",
    "bounding_box_mm": "10x10x10",
    "face_count": 501,
    "nominal_wall_mm": 2.0,
    "faces": [
        {
            "face_id": f"F{i}",
            "face_type": "Plane",
            "area_mm2": 100,
            "thickness_mm": 2.0,
            "draft_angle_deg": 1.0,
            "radius_mm": 0.2,
        }
        for i in range(501)
    ],
}
status = post_expect_status("/validate", large_part, 422)
print(f"POST /validate with 501 faces -> {status} (expected 422)")

# Gate 4: Sample part validation
sample_part = {
    "filename": "demo_bracket.SLDPRT",
    "solidworks_part_number": "BRK-001",
    "process": "injection_molding",
    "material": "ABS",
    "bounding_box_mm": "120 x 80 x 15",
    "face_count": 12,
    "volume_mm3": 45000.0,
    "surface_area_mm2": 32000.0,
    "nominal_wall_mm": 2.0,
    "faces": [
        {"face_id": "F1", "face_type": "Plane", "area_mm2": 9600.0, "thickness_mm": 0.7, "draft_angle_deg": 0.5, "radius_mm": 0.3, "depth_mm": 0, "width_mm": 120},
        {"face_id": "F2", "face_type": "Plane", "area_mm2": 9600.0, "thickness_mm": 0.7, "draft_angle_deg": 0.5, "radius_mm": 0.3, "depth_mm": 0, "width_mm": 120},
        {"face_id": "F3", "face_type": "Plane", "area_mm2": 1200.0, "thickness_mm": 0.7, "draft_angle_deg": 0.5, "radius_mm": 0.2, "depth_mm": 0, "width_mm": 80},
        {"face_id": "F4", "face_type": "Plane", "area_mm2": 1200.0, "thickness_mm": 0.7, "draft_angle_deg": 0.5, "radius_mm": 0.2, "depth_mm": 0, "width_mm": 80},
        {"face_id": "F5", "face_type": "Cylinder", "area_mm2": 282.7, "thickness_mm": 3.0, "draft_angle_deg": 1.0, "radius_mm": 3.0, "depth_mm": 15, "width_mm": 15},
        {"face_id": "F6", "face_type": "Cylinder", "area_mm2": 282.7, "thickness_mm": 3.0, "draft_angle_deg": 1.0, "radius_mm": 3.0, "depth_mm": 15, "width_mm": 15},
        {"face_id": "F7", "face_type": "Cylinder", "area_mm2": 282.7, "thickness_mm": 3.0, "draft_angle_deg": 1.0, "radius_mm": 3.0, "depth_mm": 15, "width_mm": 15},
        {"face_id": "F8", "face_type": "Cylinder", "area_mm2": 282.7, "thickness_mm": 3.0, "draft_angle_deg": 1.0, "radius_mm": 3.0, "depth_mm": 15, "width_mm": 15},
        {"face_id": "F9", "face_type": "Plane", "area_mm2": 4800.0, "thickness_mm": 2.5, "draft_angle_deg": 2.0, "radius_mm": 1.5, "depth_mm": 8, "width_mm": 60},
        {"face_id": "F10", "face_type": "Plane", "area_mm2": 4800.0, "thickness_mm": 2.5, "draft_angle_deg": 2.0, "radius_mm": 1.5, "depth_mm": 8, "width_mm": 60},
        {"face_id": "F11", "face_type": "Cylinder", "area_mm2": 94.2, "thickness_mm": 3.0, "draft_angle_deg": 0.0, "radius_mm": 2.0, "depth_mm": 15, "width_mm": 15},
        {"face_id": "F12", "face_type": "Cylinder", "area_mm2": 94.2, "thickness_mm": 3.0, "draft_angle_deg": 0.0, "radius_mm": 2.0, "depth_mm": 15, "width_mm": 15},
    ],
}

print("\n=== Gate 4: Validate ===")
try:
    result = post("/validate", sample_part)
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:2000]}")
    sys.exit(1)
print(f"Score: {result['overall_manufacturability_score']}/100")
print(f"Violations: {len(result['violations'])}")
print(f"GNN risk: {result['gnn_risk_score']}")
print(f"Review needed: {result['engineer_review_required']}")
for v in result["violations"][:3]:
    print(f"  {v['rule_id']}: {v['description'][:80]}")
print(f"Passed checks: {result['passed_checks']}")
print(f"Confidence: {result['confidence']}")

# Fix suggestion
if result["violations"]:
    v = result["violations"][0]
    print("\n=== Fix Suggestion ===")
    fix = post("/fix-suggestion", {"violation": v, "part": sample_part})
    print(f"Rule: {fix['rule_id']}")
    print(f"Root cause: {fix['root_cause'][:80]}")
    print(f"Fix: {fix['immediate_fix']['description']}")
    print(f"Est time: {fix['estimated_time_minutes']} min")
    print(f"SW steps: {fix['immediate_fix']['solidworks_steps'][0]}")

# Report
print("\n=== Report ===")
report = post("/report", result)
print(report["report"][:500])

print("\nALL TESTS PASSED")
