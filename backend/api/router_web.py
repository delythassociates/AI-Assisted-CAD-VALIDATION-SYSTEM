from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from ..core.models import PartMetadata, FaceGeometry, FaceType, ValidationResult
from .router_validate import validate_part

router = APIRouter()

# Schema mapping for user friendly UI labels to internal FaceType enums
TYPE_LABEL_MAP = {
    "Plane (Flat)": "Plane",
    "Cylinder (Round)": "Cylinder",
    "Cone (Tapered)": "Cone",
    "Sphere (Curved)": "Sphere",
    "Torus (Donut)": "Torus",
    "BSpline (Complex)": "BSpline",
    "Other": "Other"
}

class WebFaceInput(BaseModel):
    face_id: str
    face_type_label: str
    thickness_mm: Optional[float] = None
    radius_mm: Optional[float] = None
    area_mm2: Optional[float] = 0.0
    depth_mm: Optional[float] = None
    width_mm: Optional[float] = None
    curvature_min: Optional[float] = 0.0
    curvature_max: Optional[float] = 0.0
    draft_angle_deg: Optional[float] = None
    parent_wall_mm: Optional[float] = None
    centroid_dist_mm: Optional[float] = None

class WebValidateRequest(BaseModel):
    part_name: str
    process: str
    material: str
    nominal_wall_mm: Optional[float] = 1.5
    faces: List[WebFaceInput]
    adjacency: List[List[int]] = []

def map_face_type(label: str) -> str:
    if label in TYPE_LABEL_MAP:
        return TYPE_LABEL_MAP[label]
    for v in TYPE_LABEL_MAP.values():
        if v.lower() == label.lower():
            return v
    return "Other"

@router.post("/validate-graph", response_model=ValidationResult)
async def validate_graph(request: WebValidateRequest):
    if not request.faces:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one face is required for validation."
        )

    # 1. Bounding Box estimation from face areas
    total_area = sum(f.area_mm2 for f in request.faces if f.area_mm2)
    # Estimate a cubic volume shape based on total area
    x = (total_area / 13.0) ** 0.5 if total_area > 0 else 10.0
    l = 2 * x
    w = 1.5 * x
    h = x
    bounding_box_mm = f"{l:.1f} x {w:.1f} x {h:.1f}"

    # 2. Build FaceGeometry list
    faces_geom = []
    for f in request.faces:
        f_type_str = map_face_type(f.face_type_label)
        
        # Curvature default fallbacks
        c_min = f.curvature_min if f.curvature_min is not None else 0.0
        c_max = f.curvature_max if f.curvature_max is not None else 0.0
        
        # Centroid distance mapping (maps centroid_x to centroid_dist_mm to keep normalization exact)
        c_dist = f.centroid_dist_mm if f.centroid_dist_mm is not None else 0.0

        faces_geom.append(FaceGeometry(
            face_id=f.face_id,
            face_type=FaceType(f_type_str),
            area_mm2=f.area_mm2 if f.area_mm2 is not None else 0.0,
            thickness_mm=f.thickness_mm,
            draft_angle_deg=f.draft_angle_deg,
            radius_mm=f.radius_mm,
            depth_mm=f.depth_mm,
            width_mm=f.width_mm,
            parent_wall_thickness_mm=f.parent_wall_mm,
            sw_feature_name=f.face_id,
            sw_feature_type=f_type_str,
            curvature_min=c_min,
            curvature_max=c_max,
            centroid_x=c_dist,
            centroid_y=0.0,
            centroid_z=0.0,
            normal_x=0.0,
            normal_y=0.0,
            normal_z=1.0
        ))

    # 3. Create PartMetadata
    part = PartMetadata(
        filename=f"{request.part_name}.SLDPRT",
        solidworks_part_number=request.part_name,
        process=request.process,
        material=request.material,
        bounding_box_mm=bounding_box_mm,
        face_count=len(faces_geom),
        faces=faces_geom,
        nominal_wall_mm=request.nominal_wall_mm,
        classification="structural",
        pull_direction="auto"
    )

    # Inject adjacency directly into instance dict so backend/ml/inference.py uses it
    part.__dict__["adjacency"] = request.adjacency

    # 4. Invoke validation pipeline
    result = await validate_part(part)
    return result

@router.get("/presets")
async def get_presets():
    return [
        {
            "name": "Thin Wall Bracket (CRITICAL)",
            "description": "Injection moulded PC/ABS bracket with wall violation",
            "process": "injection_molding",
            "material": "PC/ABS",
            "nominal_wall_mm": 1.5,
            "faces": [
                {
                    "face_id": "F1",
                    "face_type_label": "Plane (Flat)",
                    "thickness_mm": 0.70,
                    "radius_mm": 0.3,
                    "area_mm2": 9600.0,
                    "depth_mm": 0.0,
                    "width_mm": 120.0,
                    "curvature_min": 0.0,
                    "curvature_max": 0.0,
                    "draft_angle_deg": 0.5,
                    "parent_wall_mm": 1.2,
                    "centroid_dist_mm": 45.0
                },
                {
                    "face_id": "F2",
                    "face_type_label": "Cylinder (Round)",
                    "thickness_mm": 1.10,
                    "radius_mm": 2.1,
                    "area_mm2": 1200.0,
                    "depth_mm": 8.0,
                    "width_mm": 4.2,
                    "curvature_min": 0.24,
                    "curvature_max": 0.24,
                    "draft_angle_deg": 1.2,
                    "parent_wall_mm": 1.1,
                    "centroid_dist_mm": 30.0
                },
                {
                    "face_id": "F3",
                    "face_type_label": "Plane (Flat)",
                    "thickness_mm": 1.15,
                    "radius_mm": 0.8,
                    "area_mm2": 4800.0,
                    "depth_mm": 0.0,
                    "width_mm": 80.0,
                    "curvature_min": 0.0,
                    "curvature_max": 0.0,
                    "draft_angle_deg": 0.8,
                    "parent_wall_mm": 1.15,
                    "centroid_dist_mm": 20.0
                },
                {
                    "face_id": "F4",
                    "face_type_label": "Plane (Flat)",
                    "thickness_mm": 0.30,
                    "radius_mm": 0.2,
                    "area_mm2": 600.0,
                    "depth_mm": 2.0,
                    "width_mm": 15.0,
                    "curvature_min": 0.0,
                    "curvature_max": 0.0,
                    "draft_angle_deg": 0.3,
                    "parent_wall_mm": 0.7,
                    "centroid_dist_mm": 55.0
                }
            ],
            "adjacency": [[0, 1], [1, 2], [2, 3], [3, 0]]
        },
        {
            "name": "Die Cast Housing (WARNING)",
            "description": "Aluminium die cast part with draft angle issues",
            "process": "die_cast_al",
            "material": "Al-380",
            "nominal_wall_mm": 3.0,
            "faces": [
                {
                    "face_id": "F1",
                    "face_type_label": "Plane (Flat)",
                    "thickness_mm": 2.50,
                    "radius_mm": 2.0,
                    "area_mm2": 15000.0,
                    "depth_mm": 0.0,
                    "width_mm": 150.0,
                    "curvature_min": 0.0,
                    "curvature_max": 0.0,
                    "draft_angle_deg": 0.2,
                    "parent_wall_mm": 2.5,
                    "centroid_dist_mm": 20.0
                },
                {
                    "face_id": "F2",
                    "face_type_label": "Cylinder (Round)",
                    "thickness_mm": 3.00,
                    "radius_mm": 1.5,
                    "area_mm2": 2000.0,
                    "depth_mm": 15.0,
                    "width_mm": 10.0,
                    "curvature_min": 0.66,
                    "curvature_max": 0.66,
                    "draft_angle_deg": 0.5,
                    "parent_wall_mm": 3.0,
                    "centroid_dist_mm": 15.0
                },
                {
                    "face_id": "F3",
                    "face_type_label": "Plane (Flat)",
                    "thickness_mm": 2.50,
                    "radius_mm": 2.0,
                    "area_mm2": 15000.0,
                    "depth_mm": 0.0,
                    "width_mm": 150.0,
                    "curvature_min": 0.0,
                    "curvature_max": 0.0,
                    "draft_angle_deg": 1.5,
                    "parent_wall_mm": 2.5,
                    "centroid_dist_mm": 20.0
                },
                {
                    "face_id": "F4",
                    "face_type_label": "Plane (Flat)",
                    "thickness_mm": 2.50,
                    "radius_mm": 2.0,
                    "area_mm2": 8000.0,
                    "depth_mm": 0.0,
                    "width_mm": 80.0,
                    "curvature_min": 0.0,
                    "curvature_max": 0.0,
                    "draft_angle_deg": 1.5,
                    "parent_wall_mm": 2.5,
                    "centroid_dist_mm": 10.0
                }
            ],
            "adjacency": [[0, 1], [1, 2], [2, 3], [3, 0]]
        },
        {
            "name": "Compliant CNC Part (PASS)",
            "description": "CNC machined steel bracket — all rules pass",
            "process": "cnc",
            "material": "Steel",
            "nominal_wall_mm": 5.0,
            "faces": [
                {
                    "face_id": "F1",
                    "face_type_label": "Plane (Flat)",
                    "thickness_mm": 8.00,
                    "radius_mm": 5.0,
                    "area_mm2": 12000.0,
                    "depth_mm": 0.0,
                    "width_mm": 100.0,
                    "curvature_min": 0.0,
                    "curvature_max": 0.0,
                    "draft_angle_deg": 0.0,
                    "parent_wall_mm": 8.0,
                    "centroid_dist_mm": 10.0
                },
                {
                    "face_id": "F2",
                    "face_type_label": "Plane (Flat)",
                    "thickness_mm": 8.00,
                    "radius_mm": 5.0,
                    "area_mm2": 12000.0,
                    "depth_mm": 0.0,
                    "width_mm": 100.0,
                    "curvature_min": 0.0,
                    "curvature_max": 0.0,
                    "draft_angle_deg": 0.0,
                    "parent_wall_mm": 8.0,
                    "centroid_dist_mm": 10.0
                },
                {
                    "face_id": "F3",
                    "face_type_label": "Cylinder (Round)",
                    "thickness_mm": 8.00,
                    "radius_mm": 4.0,
                    "area_mm2": 3000.0,
                    "depth_mm": 10.0,
                    "width_mm": 8.0,
                    "curvature_min": 0.25,
                    "curvature_max": 0.25,
                    "draft_angle_deg": 0.0,
                    "parent_wall_mm": 8.0,
                    "centroid_dist_mm": 5.0
                },
                {
                    "face_id": "F4",
                    "face_type_label": "Cylinder (Round)",
                    "thickness_mm": 8.00,
                    "radius_mm": 4.0,
                    "area_mm2": 3000.0,
                    "depth_mm": 10.0,
                    "width_mm": 8.0,
                    "curvature_min": 0.25,
                    "curvature_max": 0.25,
                    "draft_angle_deg": 0.0,
                    "parent_wall_mm": 8.0,
                    "centroid_dist_mm": 5.0
                }
            ],
            "adjacency": [[0, 2], [1, 2], [0, 3], [1, 3]]
        }
    ]
