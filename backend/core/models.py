from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from enum import Enum

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"

class Category(str, Enum):
    INJECTION_MOLDING = "injection_molding"
    DIE_CASTING = "die_casting"
    CNC = "cnc"
    GDT = "gdt"
    ASSEMBLY = "assembly"

class FaceType(str, Enum):
    PLANE = "Plane"
    CYLINDER = "Cylinder"
    SPHERE = "Sphere"
    CONE = "Cone"
    TORUS = "Torus"
    BSPLINE = "BSpline"
    OTHER = "Other"

class FaceGeometry(BaseModel):
    face_id: str
    face_type: FaceType = FaceType.OTHER
    area_mm2: float = 0.0
    thickness_mm: Optional[float] = None
    draft_angle_deg: Optional[float] = None
    radius_mm: Optional[float] = None
    depth_mm: Optional[float] = None
    width_mm: Optional[float] = None
    parent_wall_thickness_mm: Optional[float] = None
    sw_feature_name: str = ""
    sw_feature_type: str = ""
    curvature_min: float = 0.0
    curvature_max: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    centroid_z: float = 0.0
    normal_x: float = 0.0
    normal_y: float = 0.0
    normal_z: float = 0.0

class FeatureTreeItem(BaseModel):
    feature_name: str
    feature_type: str
    children: list = []

class PartMetadata(BaseModel):
    filename: str = ""
    solidworks_part_number: str = ""
    process: str = ""
    material: str = ""
    bounding_box_mm: str = ""
    face_count: int = 0
    feature_tree_depth: int = 0
    volume_mm3: Optional[float] = None
    surface_area_mm2: Optional[float] = None
    faces: list[FaceGeometry] = []
    feature_tree: list[FeatureTreeItem] = []
    assembly_gaps: list = []
    nominal_wall_mm: Optional[float] = 1.5
    classification: Optional[str] = "structural"
    pull_direction: Optional[str] = "auto"
    class_a_face_ids: list[str] = []

class PartStats(BaseModel):
    part_id: str = ""
    source_dataset: str = "abc"
    face_count: int = 0
    surf_count: int = 0
    min_wall_thickness_mm: Optional[float] = None
    mean_wall_thickness_mm: Optional[float] = None
    min_draft_angle_deg: Optional[float] = None
    mean_draft_angle_deg: Optional[float] = None
    min_fillet_radius_mm: Optional[float] = None
    hole_count: int = 0
    rib_count: int = 0
    pocket_count: int = 0
    max_depth_width_ratio: float = 0.0
    adjacency_edge_count: int = 0
    bounding_volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0
    gaussian_curvature: float = 0.0
    mean_curvature: float = 0.0
    surface_types: list[str] = []
    curve_types: list[str] = []

class Violation(BaseModel):
    rule_id: str
    category: str
    severity: Severity
    face_ids: list[str] = []
    solidworks_feature_name: str = ""
    measured_value: str = ""
    required_value: str = ""
    standard_reference: str = ""
    description: str = ""
    fix_suggestion: str = ""
    solidworks_fix_path: str = ""
    unaddressed_risk_score: int = 5
    unaddressed_risk_reasoning: str = ""
    
    # Gemini-enriched fields
    current_value_mm: Optional[float] = None
    minimum_required_mm: Optional[float] = None
    optimal_value_mm: Optional[float] = None
    fix_delta_mm: Optional[float] = None
    plain_english: Optional[str] = ""
    fix_instruction: Optional[str] = ""
    highlight_color: Optional[str] = ""
    status: Optional[str] = "ACTIVE"

class FaceSnapshot(BaseModel):
    face_id: str
    snapshot_b64: str          # base64 PNG from SolidWorks viewport
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    centroid_z: float = 0.0
    area_mm2: float = 0.0
    face_type: str = ""

class ValidationResult(BaseModel):
    part_id: str = ""
    solidworks_document_path: str = ""
    overall_manufacturability_score: int = 100
    risk_summary: dict = {}
    violations: list[Violation] = []
    ml_anomaly_flags: list = []
    passed_checks: list[str] = []
    engineer_review_required: bool = False
    confidence: float = 1.0
    data_gaps: list[str] = []
    gnn_risk_score: float = 0.0
    gnn_anomaly: Optional[dict] = None
    gemini_enriched: bool = False
    process: str = ""
    material: str = ""
    face_health: dict = {}
    screenshot_png_base64: Optional[str] = ""
    gemini_narrative: Optional[str] = ""
    
    # NEW — optional, populated when add-in sends snapshots
    face_snapshots: Optional[Dict[str, str]] = {}
    faces_geometry: Optional[List[dict]] = []

class FixSuggestion(BaseModel):
    rule_id: str = ""
    root_cause: str = ""
    immediate_fix: dict = {}
    alternative_fixes: list[dict] = []
    downstream_impacts: list[str] = []
    verification_method: str = ""
    estimated_time_minutes: int = 15

class TrainingLabel(BaseModel):
    part_id: str
    label: int
    label_text: str = ""
    confidence: float = 0.0
    primary_evidence: list[str] = []
    label_rationale: str = ""
    needs_human_review: bool = False

class FeedbackEvent(BaseModel):
    event_id: str = ""
    timestamp: str = ""
    engineer_id: str = ""
    part_id: str = ""
    feedback_type: str = ""
    corrected_label: int = 0
    engineer_comment: str = ""
