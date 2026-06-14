"""
Eureka DFM 3.0 — Autodesk Fusion 360 Add-in
Full-featured DFM validation connector with:
  - Geometry extraction from BRep faces
  - Backend REST communication (validate, report, feedback)
  - Face color overlay by severity
  - Face highlighting with dimension annotations
  - HTML palette UI panel
  - Backend auto-start
"""

import adsk.core
import adsk.fusion
import traceback
import os
import sys
import json
import math
import threading
import subprocess

# ─── Global State ───────────────────────────────────────────────────────────────
app: adsk.core.Application = None
ui: adsk.core.UserInterface = None
handlers = []
face_registry = {}          # face_id_str -> BRepFace
face_original_appearances = {}  # face_id_str -> original Appearance
last_result = None           # ValidationResult dict
last_part_metadata = None    # PartMetadata dict
palette = None
custom_graphics_group = None
BACKEND_URL = os.environ.get('EUREKA_BACKEND_URL', 'http://localhost:8001')
API_KEY = os.environ.get('EUREKA_API_KEY', 'eureka-dev-key-change-me')

# ─── Command Definitions ────────────────────────────────────────────────────────
CMD_ID = 'EurekaDFMCommand'
CMD_NAME = 'Eureka DFM'
CMD_DESC = 'Run Design for Manufacturability (DFM) verification'
PALETTE_ID = 'EurekaDFMPalette'
PALETTE_NAME = 'Eureka DFM 3.0'

# ─── Logging ─────────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(os.environ.get('TEMP', '/tmp'), 'eureka_fusion.log')


def log(msg):
    """Thread-safe logging to temp file."""
    try:
        import datetime
        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f'[{ts}] {msg}\n')
    except Exception:
        pass


# ─── REST Client ─────────────────────────────────────────────────────────────────
class RestClient:
    """HTTP client for communicating with the Eureka DFM backend."""

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url or BACKEND_URL
        self.api_key = api_key or API_KEY

    def _post(self, path, payload, timeout=90):
        """POST JSON to the backend with retry on 429/503."""
        import urllib.request
        import urllib.error
        import time

        url = f'{self.base_url}{path}'
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }

        retry_delays = [1.0, 2.0, 4.0]
        last_error = None

        for attempt in range(len(retry_delays) + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode('utf-8')
                    return json.loads(body)
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < len(retry_delays):
                    time.sleep(retry_delays[attempt])
                    last_error = e
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < len(retry_delays):
                    time.sleep(retry_delays[attempt])
                    continue
                raise

        raise last_error

    def _get(self, path, timeout=10):
        """GET from the backend (no API key for health)."""
        import urllib.request
        url = f'{self.base_url}{path}'
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8')
            return json.loads(body)

    def validate(self, part_metadata):
        return self._post('/validate', part_metadata)

    def health(self):
        return self._get('/health')

    def report(self, result):
        return self._post('/report', result)

    def report_pdf(self, payload):
        """Returns PDF bytes."""
        import urllib.request
        url = f'{self.base_url}/report/pdf'
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()

    def feedback(self, payload):
        return self._post('/feedback', payload)


client = RestClient()


# ─── Geometry Extraction ─────────────────────────────────────────────────────────
def build_part_metadata(config=None):
    """Extract full face geometry from the active Fusion design, matching
    the exact schema expected by the SolidWorks/CATIA backend endpoint."""
    global face_registry
    face_registry.clear()
    face_original_appearances.clear()

    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError('No active Fusion Design. Open a part first.')

    root = design.rootComponent

    # Gather solid bodies
    bodies = []
    for b in root.bRepBodies:
        if b.isSolid:
            bodies.append(b)
    for occ in root.allOccurrences:
        if not occ.isLightBulbOn:
            continue
        comp = occ.component
        for b in comp.bRepBodies:
            if b.isSolid:
                bodies.append(b)

    if not bodies:
        raise RuntimeError('No solid bodies found in the active design.')

    body = bodies[0]
    log(f'Using body: {body.name}, faces: {body.faces.count}')

    # Bounding box
    bbox = body.boundingBox
    min_pt = bbox.minPoint
    max_pt = bbox.maxPoint
    bx = abs(max_pt.x - min_pt.x) * 10.0  # cm -> mm
    by = abs(max_pt.y - min_pt.y) * 10.0
    bz = abs(max_pt.z - min_pt.z) * 10.0

    volume_mm3 = body.volume * 1000.0       # cm³ -> mm³
    surface_area_mm2 = body.area * 100.0    # cm² -> mm²
    material_name = body.material.name if body.material else "Unknown"

    faces_list = list(body.faces)
    if len(faces_list) > 500:
        raise RuntimeError(f'Part has {len(faces_list)} faces (max 500). Use a simpler part.')

    faces_geom = []
    face_index = 1

    for face in faces_list:
        fid = str(face_index)
        face_registry[fid] = face

        # Save original appearance for cleanup
        try:
            if face.appearance:
                face_original_appearances[fid] = face.appearance
        except Exception:
            pass

        face_index += 1

        geom = face.geometry
        area_mm2 = face.area * 100.0  # cm² -> mm²

        # Surface type
        type_map = {
            adsk.core.Plane.classType(): "Plane",
            adsk.core.Cylinder.classType(): "Cylinder",
            adsk.core.Sphere.classType(): "Sphere",
            adsk.core.Cone.classType(): "Cone",
            adsk.core.Torus.classType(): "Torus",
        }
        surf_type = type_map.get(geom.objectType, "BSpline")

        # Centroid and Normal
        evaluator = face.evaluator
        cx, cy, cz = 0.0, 0.0, 0.0
        nx, ny, nz = 0.0, 0.0, 1.0

        try:
            pt = face.pointOnFace
            cx, cy, cz = pt.x * 10.0, pt.y * 10.0, pt.z * 10.0
        except Exception:
            pass

        try:
            range_box = evaluator.parametricRange()
            min_p = range_box.minPoint
            max_p = range_box.maxPoint
            mid_p = adsk.core.Point2D.create(
                (min_p.x + max_p.x) / 2.0,
                (min_p.y + max_p.y) / 2.0
            )
            ret_val, normal = evaluator.getNormalAtParameter(mid_p)
            if ret_val:
                nx, ny, nz = normal.x, normal.y, normal.z
                if face.isParamReversed:
                    nx, ny, nz = -nx, -ny, -nz
        except Exception as e:
            log(f'Failed to get normal for face: {e}')

        # Radius estimation
        radius_mm = 1.0
        if surf_type in ("Cylinder", "Sphere"):
            try:
                radius_mm = geom.radius * 10.0
            except Exception:
                pass

        # Thickness estimation
        thickness = math.sqrt(area_mm2) * 0.08
        thickness = max(0.3, min(8.0, thickness))

        width = math.sqrt(area_mm2)
        depth = radius_mm * 2.0 if surf_type == "Cylinder" else (thickness * 0.5 if surf_type == "Plane" else 0.0)
        curv = 1.0 / radius_mm if (surf_type in ("Cylinder", "Sphere") and radius_mm > 0.001) else 0.0

        # Draft angle relative to +Z pull direction
        dot = abs(nz)
        draft_angle_deg = math.asin(min(dot, 1.0)) * 180.0 / math.pi

        faces_geom.append({
            "face_id": fid,
            "face_type": surf_type,
            "area_mm2": area_mm2,
            "thickness_mm": thickness,
            "draft_angle_deg": draft_angle_deg,
            "radius_mm": radius_mm,
            "depth_mm": depth,
            "width_mm": width,
            "sw_feature_name": f"Face_{fid}",
            "sw_feature_type": surf_type,
            "curvature_min": curv,
            "curvature_max": curv,
            "centroid_x": cx,
            "centroid_y": cy,
            "centroid_z": cz,
            "normal_x": nx,
            "normal_y": ny,
            "normal_z": nz,
        })

    # Parent wall thickness = median
    if faces_geom:
        sorted_t = sorted(f["thickness_mm"] for f in faces_geom)
        median_t = sorted_t[len(sorted_t) // 2]
        for f in faces_geom:
            f["parent_wall_thickness_mm"] = median_t

    # Topological adjacency (edge-sharing)
    edge_to_faces = {}
    for idx, face in enumerate(faces_list):
        for edge in face.edges:
            try:
                eid = edge.tempId
                edge_to_faces.setdefault(eid, []).append(idx)
            except Exception:
                pass

    adjacency = []
    seen = set()
    for eid, findices in edge_to_faces.items():
        if len(findices) >= 2:
            for i in range(len(findices)):
                for j in range(i + 1, len(findices)):
                    pair = (min(findices[i], findices[j]), max(findices[i], findices[j]))
                    if pair not in seen:
                        seen.add(pair)
                        adjacency.append([pair[0], pair[1]])

    # Build metadata dict
    doc_name = app.activeDocument.name
    filename = os.path.basename(doc_name)
    part_number = os.path.splitext(filename)[0]

    cfg = config or {}
    part_meta = {
        "filename": filename,
        "solidworks_part_number": part_number,
        "process": cfg.get("process", "injection_moulding"),
        "material": cfg.get("material", material_name),
        "nominal_wall_mm": cfg.get("nominal_wall_mm", 1.5),
        "bounding_box_mm": f"{bx:.1f} x {by:.1f} x {bz:.1f}",
        "face_count": len(faces_geom),
        "volume_mm3": volume_mm3,
        "surface_area_mm2": surface_area_mm2,
        "faces": faces_geom,
        "adjacency": adjacency,
        "pull_direction": cfg.get("pull_direction", "auto"),
    }

    log(f'Extracted {len(faces_geom)} faces, bbox={part_meta["bounding_box_mm"]}')
    return part_meta


# ─── Face Coloring ───────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    'CRITICAL': (224, 82, 82),     # Red
    'WARNING': (224, 154, 58),     # Orange
    'INFO': (224, 200, 58),        # Yellow
}
PASS_COLOR = (82, 196, 122)       # Green


def _find_or_create_appearance(design, name, r, g, b):
    """Find or create a colored appearance in the design's appearances collection."""
    app_name = f'EurekaDFM_{name}'

    # Check existing
    existing = design.appearances.itemByName(app_name)
    if existing:
        return existing

    # Create new appearance by copying the default
    try:
        # Use the design's material library to find a base appearance
        mat_lib = app.materialLibraries.itemByName('Fusion 360 Appearance Library')
        if mat_lib:
            base_app = None
            for i in range(mat_lib.appearances.count):
                a = mat_lib.appearances.item(i)
                if 'Plastic' in a.name or 'Generic' in a.name:
                    base_app = a
                    break
            if base_app:
                new_app = design.appearances.addByCopy(base_app, app_name)
                # Set the color
                color_prop = None
                for prop in new_app.appearanceProperties:
                    if hasattr(prop, 'value') and isinstance(getattr(prop, 'value', None), adsk.core.Color):
                        color_prop = prop
                        break
                if color_prop:
                    color_prop.value = adsk.core.Color.create(r, g, b, 255)
                return new_app
    except Exception as e:
        log(f'Failed to create appearance from library: {e}')

    # Fallback: try to create with a simpler method
    try:
        # If no library available, try creating a basic appearance
        existing_apps = design.appearances
        if existing_apps.count > 0:
            base = existing_apps.item(0)
            new_app = existing_apps.addByCopy(base, app_name)
            for prop in new_app.appearanceProperties:
                if hasattr(prop, 'value') and isinstance(getattr(prop, 'value', None), adsk.core.Color):
                    prop.value = adsk.core.Color.create(r, g, b, 255)
                    break
            return new_app
    except Exception as e:
        log(f'Fallback appearance creation failed: {e}')

    return None


def apply_face_overlay(result):
    """Color all faces by severity, matching SolidWorks OverlayRenderer behavior."""
    global last_result
    last_result = result
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        return

    # Map face_id -> highest severity
    face_severities = {}
    priority_map = {'CRITICAL': 3, 'WARNING': 2, 'INFO': 1}
    if result.get('violations'):
        for v in result['violations']:
            fids = v.get('face_ids', [])
            sev = (v.get('severity') or '').upper()
            rule = (v.get('rule_id') or '').upper()
            if rule.startswith('GNN-RISK'):
                sev = 'WARNING'
            elif rule.startswith('GNN-WATCH') or rule.startswith('GNN-ANOMALY'):
                sev = 'INFO'
            for fid in fids:
                cur = face_severities.get(fid)
                if cur is None or priority_map.get(sev, 0) > priority_map.get(cur, 0):
                    face_severities[fid] = sev

    # Create appearances for each severity
    appearances = {}
    for sev_name, (r, g, b) in SEVERITY_COLORS.items():
        a = _find_or_create_appearance(design, sev_name, r, g, b)
        if a:
            appearances[sev_name] = a

    pass_app = _find_or_create_appearance(design, 'PASS', *PASS_COLOR)
    if pass_app:
        appearances['PASS'] = pass_app

    # Apply to faces
    colored = 0
    for fid, face in face_registry.items():
        try:
            if not face.isValid:
                continue
            sev = face_severities.get(fid)
            target_app = appearances.get(sev) if sev else appearances.get('PASS')
            if target_app:
                face.appearance = target_app
                colored += 1
        except Exception as e:
            log(f'Failed to color face {fid}: {e}')

    # Refresh viewport
    try:
        adsk.doEvents()
        app.activeViewport.refresh()
    except Exception:
        pass

    log(f'Colored {colored} faces (red:{len([f for f in face_severities.values() if f=="CRITICAL"])}, '
        f'orange:{len([f for f in face_severities.values() if f=="WARNING"])}, '
        f'yellow:{len([f for f in face_severities.values() if f=="INFO"])}, '
        f'green:{colored - len(face_severities)})')


def clear_face_overlay():
    """Remove all DFM color overlays, restoring original appearances."""
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        return

    for fid, face in face_registry.items():
        try:
            if not face.isValid:
                continue
            orig = face_original_appearances.get(fid)
            if orig:
                face.appearance = orig
            else:
                # Remove override to revert to body/component appearance
                face.appearance = None
        except Exception as e:
            log(f'Failed to clear face {fid}: {e}')

    # Clear annotations
    clear_annotations()

    try:
        app.activeViewport.refresh()
    except Exception:
        pass

    log('Overlay cleared')


# ─── Annotations / Custom Graphics ──────────────────────────────────────────────
def clear_annotations():
    """Remove all custom graphics from the root component."""
    global custom_graphics_group
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design and custom_graphics_group:
            custom_graphics_group.deleteMe()
            custom_graphics_group = None
    except Exception as e:
        log(f'clear_annotations error: {e}')


def highlight_violation(violation):
    """Highlight specific faces and show an annotation with dimensions."""
    clear_annotations()
    global custom_graphics_group

    face_ids = violation.get('face_ids', [])
    if not face_ids:
        return

    # Select faces in viewport
    ui.activeSelections.clear()
    first_face = None
    for fid in face_ids:
        face = face_registry.get(str(fid))
        if face and face.isValid:
            ui.activeSelections.add(face)
            if first_face is None:
                first_face = face

    if not first_face:
        return

    # Build annotation text
    rule_id = violation.get('rule_id', '')
    severity = (violation.get('severity', '')).upper()
    message = violation.get('plain_english') or violation.get('description', '')

    # Determine unit
    is_angle = any(k in rule_id.lower() for k in ('draft', 'angle', 'undercut'))
    unit = '\u00B0' if is_angle else 'mm'

    # Measured / Required values
    measured = violation.get('current_value_mm')
    if measured is None:
        measured = _extract_number(violation.get('measured_value'))
    required = violation.get('minimum_required_mm')
    if required is None:
        required = _extract_number(violation.get('required_value'))

    lines = [f'{severity}: {rule_id}']
    if message:
        lines.append(message[:80])
    if measured is not None:
        lines.append(f'Measured: {measured:.2f}{unit}')
    if required is not None:
        lines.append(f'Required: {required:.2f}{unit}')
    fix = violation.get('fix_instruction') or violation.get('fix_suggestion', '')
    if fix:
        lines.append(f'Fix: {fix[:60]}')

    annotation_text = '\n'.join(lines)

    # Create custom graphics annotation near the face
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        root = design.rootComponent
        custom_graphics_group = root.customGraphicsGroups.add()

        # Get face centroid for annotation position
        centroid = first_face.pointOnFace
        if centroid:
            # Offset annotation slightly from face
            pos = adsk.core.Point3D.create(
                centroid.x + 1.0,  # Offset 1cm = 10mm
                centroid.y + 1.0,
                centroid.z + 0.5
            )

            transform = adsk.core.Matrix3D.create()
            transform.translation = adsk.core.Vector3D.create(pos.x, pos.y, pos.z)

            # Create text annotation
            billboard = adsk.fusion.CustomGraphicsBillBoard.create(pos)
            text = custom_graphics_group.addText(
                annotation_text,
                'Arial',
                0.4,  # font height in cm
                transform
            )
            text.billBoarding = billboard
            text.isSelectable = False

            # Set color based on severity
            if severity == 'CRITICAL':
                text.color = adsk.fusion.CustomGraphicsSolidColorEffect.create(
                    adsk.core.Color.create(224, 82, 82, 255))
            elif severity == 'WARNING':
                text.color = adsk.fusion.CustomGraphicsSolidColorEffect.create(
                    adsk.core.Color.create(224, 154, 58, 255))
            else:
                text.color = adsk.fusion.CustomGraphicsSolidColorEffect.create(
                    adsk.core.Color.create(74, 158, 224, 255))

            # Also add a line from centroid to annotation
            line_coords = adsk.fusion.CustomGraphicsCoordinates.create([
                centroid.x, centroid.y, centroid.z,
                pos.x, pos.y, pos.z
            ])
            line = custom_graphics_group.addLines(line_coords, [0, 1], False)
            line.weight = 2.0
            line.color = adsk.fusion.CustomGraphicsSolidColorEffect.create(
                adsk.core.Color.create(74, 128, 224, 200))

        app.activeViewport.refresh()
    except Exception as e:
        log(f'highlight_violation annotation error: {e}')

    log(f'Highlighted face(s) {face_ids} for rule {rule_id}')


def _extract_number(s):
    """Extract first numeric value from a string."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    import re
    m = re.search(r'[-+]?\d*\.?\d+', str(s))
    return float(m.group()) if m else None


# ─── Backend Management ──────────────────────────────────────────────────────────
def check_backend_health():
    """Check if backend is reachable. Returns True/False."""
    try:
        client.health()
        return True
    except Exception:
        return False


def try_start_backend():
    """Try to launch start_backend.ps1 from the project root."""
    try:
        # Walk up from add-in dir to find project root with start_backend.ps1
        current = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        candidates = [current]
        # Also try the known project location
        candidates.append('F:\\Varroc')

        for d in candidates:
            script = os.path.join(d, 'start_backend.ps1')
            if os.path.exists(script):
                log(f'Starting backend from {script}')
                subprocess.Popen(
                    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', 'start_backend.ps1'],
                    cwd=d,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                # Wait for backend to be ready
                import time
                for i in range(10):
                    time.sleep(2)
                    if check_backend_health():
                        log(f'Backend started on attempt {i+1}')
                        return True
                log('Backend did not start after 20s')
                return False

        log('start_backend.ps1 not found')
        return False
    except Exception as e:
        log(f'try_start_backend error: {e}')
        return False


# ─── Validation Pipeline ────────────────────────────────────────────────────────
def run_validation(config):
    """Run the complete validation pipeline in a background thread."""
    global last_result, last_part_metadata

    def _worker():
        global last_result, last_part_metadata
        try:
            import time
            start = time.time()

            # Step 1: Extract geometry
            send_to_palette('progress', {'stage': 'extracting'})
            try:
                part_meta = build_part_metadata(config)
                last_part_metadata = part_meta
            except Exception as e:
                log(f'Geometry extraction failed: {e}')
                send_to_palette('error', {'message': str(e)})
                return

            # Step 2: Rules engine
            send_to_palette('progress', {'stage': 'rules'})
            try:
                result = client.validate(part_meta)
            except Exception as e:
                log(f'Backend validation failed: {e}')
                send_to_palette('error', {'message': f'Backend error: {e}'})
                return

            # Step 3: GNN (already done server-side, update progress)
            send_to_palette('progress', {'stage': 'gnn'})

            last_result = result
            elapsed = time.time() - start

            # Step 4: Apply overlay colors on the main thread
            # Note: Fusion API calls must happen on the main thread
            # We'll use adsk.doEvents and a flag approach
            try:
                apply_face_overlay(result)
            except Exception as e:
                log(f'apply_face_overlay error: {e}')

            # Send results to palette
            send_to_palette('load_results', result)
            send_to_palette('validation_time', {'time_str': f'{elapsed:.1f}s'})
            send_to_palette('progress', {'stage': 'done'})

            log(f'Validation complete: score={result.get("overall_manufacturability_score")}, '
                f'violations={len(result.get("violations", []))}, time={elapsed:.1f}s')

        except Exception as e:
            log(f'run_validation error: {e}\n{traceback.format_exc()}')
            send_to_palette('error', {'message': str(e)})

    # Run on main thread since Fusion API is not thread-safe
    _worker()


def submit_feedback(is_accurate):
    """Submit engineer feedback to the backend."""
    global last_result, last_part_metadata

    def _worker():
        try:
            if not last_result:
                send_to_palette('feedback_error', {'message': 'No active validation result'})
                return

            gnn_score = last_result.get('gnn_risk_score', 0.0)
            predicted_label = 1 if gnn_score >= 0.5 else 0
            engineer_label = predicted_label if is_accurate else (1 - predicted_label)

            yml_path = None
            rs = last_result.get('risk_summary')
            if rs and isinstance(rs, dict):
                yml_path = rs.get('yml_path')

            part_id = last_result.get('part_id', '')
            if not part_id and last_part_metadata:
                part_id = last_part_metadata.get('filename', 'unknown_part')

            payload = {
                'part_id': part_id,
                'yml_path': yml_path,
                'predicted_label': predicted_label,
                'predicted_score': gnn_score,
                'engineer_label': engineer_label,
            }

            resp = client.feedback(payload)
            message = 'Feedback recorded!'
            if isinstance(resp, dict) and 'stats' in resp:
                stats = resp['stats']
                if isinstance(stats, dict) and 'corrections_pending' in stats:
                    message = f'Feedback logged. {stats["corrections_pending"]}/30 until next fine-tune.'

            send_to_palette('feedback_ok', {'message': message})
            log(f'Feedback submitted: is_accurate={is_accurate}')

        except Exception as e:
            log(f'submit_feedback error: {e}')
            send_to_palette('feedback_error', {'message': str(e)})

    _worker()


# ─── Palette Communication ───────────────────────────────────────────────────────
def send_to_palette(action, data):
    """Send data from Python to the HTML palette."""
    global palette
    try:
        if palette and palette.isValid:
            palette.sendInfoToHTML(action, json.dumps(data))
    except Exception as e:
        log(f'send_to_palette error: {e}')


# ─── Event Handlers ──────────────────────────────────────────────────────────────
class HTMLEventHandler(adsk.core.HTMLEventHandler):
    """Handle events from the HTML palette (JS -> Python)."""

    def __init__(self):
        super().__init__()

    def notify(self, args):
        try:
            html_args = adsk.core.HTMLEventArgs.cast(args)
            action = html_args.action
            data_str = html_args.data

            log(f'HTML event: action={action}')

            if action == 'validate':
                data = json.loads(data_str) if data_str else {}
                run_validation(data)

            elif action == 'clear':
                clear_face_overlay()
                ui.activeSelections.clear()

            elif action == 'highlight_face':
                data = json.loads(data_str) if data_str else {}
                violation = data.get('violation', {})
                if violation:
                    highlight_violation(violation)

            elif action == 'clear_highlight':
                clear_annotations()
                ui.activeSelections.clear()
                # Re-apply overlay
                if last_result:
                    apply_face_overlay(last_result)

            elif action == 'feedback':
                data = json.loads(data_str) if data_str else {}
                is_accurate = data.get('is_accurate', True)
                submit_feedback(is_accurate)

            elif action == 'export_report':
                export_report()

        except Exception:
            log(f'HTMLEventHandler error:\n{traceback.format_exc()}')


class CommandExecuteHandler(adsk.core.CommandEventHandler):
    """Handler for when the command button is clicked — opens the palette."""

    def __init__(self):
        super().__init__()

    def notify(self, args):
        try:
            global palette
            palette = ui.palettes.itemById(PALETTE_ID)
            if not palette:
                # Compute HTML path
                current_dir = os.path.dirname(os.path.realpath(__file__))
                html_path = os.path.join(current_dir, 'palette.html')
                html_url = 'file:///' + html_path.replace('\\', '/')

                palette = ui.palettes.add(
                    PALETTE_ID,
                    PALETTE_NAME,
                    html_url,
                    True,   # isVisible
                    True,   # showCloseButton
                    True,   # isResizable
                    350,    # width
                    900     # height
                )
                palette.dockingState = adsk.core.PaletteDockingStates.PaletteDockStateRight

                # Add HTML event handler
                on_html = HTMLEventHandler()
                palette.incomingFromHTML.add(on_html)
                handlers.append(on_html)

                # Add close handler
                on_close = PaletteCloseHandler()
                palette.closed.add(on_close)
                handlers.append(on_close)

                log('Created palette')

            palette.isVisible = True

            # Check backend health and report to palette
            def _check():
                import time
                time.sleep(1)  # Brief delay to let palette load
                connected = check_backend_health()
                if not connected:
                    connected = try_start_backend()
                send_to_palette('backend_status', {'connected': connected})

            t = threading.Thread(target=_check, daemon=True)
            t.start()

        except Exception:
            log(f'CommandExecuteHandler error:\n{traceback.format_exc()}')
            if ui:
                ui.messageBox(f'Eureka DFM command failed:\n{traceback.format_exc()}')


class CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    """Handler for command creation — wires up the execute handler."""

    def __init__(self):
        super().__init__()

    def notify(self, args):
        try:
            cmd = args.command
            on_exec = CommandExecuteHandler()
            cmd.execute.add(on_exec)
            handlers.append(on_exec)
        except Exception:
            log(f'CommandCreatedHandler error:\n{traceback.format_exc()}')
            if ui:
                ui.messageBox(f'Command creation failed:\n{traceback.format_exc()}')


class PaletteCloseHandler(adsk.core.UserInterfaceGeneralEventHandler):
    """Handle palette close event."""

    def __init__(self):
        super().__init__()

    def notify(self, args):
        try:
            log('Palette closed')
        except Exception:
            pass


# ─── Report Export ───────────────────────────────────────────────────────────────
def export_report():
    """Export validation report as PDF."""
    try:
        if not last_result:
            ui.messageBox('No validation results to export.')
            return

        # Use Fusion's file dialog
        dlg = ui.createFileDialog()
        dlg.title = 'Export DFM Report'
        dlg.filter = 'PDF Files (*.pdf);;HTML Files (*.html);;Markdown Files (*.md)'
        dlg.filterIndex = 0
        dlg.initialFilename = f'EUREKA_DFM_Report_{last_result.get("part_id", "part")}'

        result = dlg.showSave()
        if result != adsk.core.DialogResults.DialogOK:
            return

        filepath = dlg.filename

        if filepath.endswith('.pdf'):
            payload = {
                'result': last_result,
                'face_snapshots': {},
                'faces_geometry': last_part_metadata.get('faces', []) if last_part_metadata else []
            }
            pdf_bytes = client.report_pdf(payload)
            with open(filepath, 'wb') as f:
                f.write(pdf_bytes)
        else:
            resp = client.report(last_result)
            content = resp.get('report', '') if isinstance(resp, dict) else str(resp)
            if filepath.endswith('.html'):
                html = (
                    '<html><head><style>body { font-family: "Segoe UI", sans-serif; '
                    'margin: 40px; color: #333; } h2 { color: #1565C0; border-bottom: '
                    '2px solid #1565C0; padding-bottom: 5px; }</style></head><body>'
                    + content.replace('\n', '<br/>') + '</body></html>'
                )
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

        ui.messageBox(f'Report exported to:\n{filepath}')
        log(f'Report exported to {filepath}')

    except Exception as e:
        log(f'export_report error: {e}')
        ui.messageBox(f'Export failed: {e}')


# ─── Add-in Lifecycle ────────────────────────────────────────────────────────────
def run(context):
    """Add-in entry point — registers command and toolbar button."""
    global app, ui
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        log('='*60)
        log('Eureka DFM 3.0 Fusion Add-in starting')

        # Create command definition
        cmd_defs = ui.commandDefinitions
        cmd_def = cmd_defs.itemById(CMD_ID)
        if not cmd_def:
            cmd_def = cmd_defs.addButtonDefinition(CMD_ID, CMD_NAME, CMD_DESC, '')

        on_created = CommandCreatedHandler()
        cmd_def.commandCreated.add(on_created)
        handlers.append(on_created)

        # Add to the UTILITIES tab > ADD-INS panel  (Fusion 360 standard)
        # Try the standard panels
        panel_ids = ['SolidScriptsAddinsPanel', 'SolidAddinsPanel', 'ToolsAddinsPanel']
        added = False
        for pid in panel_ids:
            panel = ui.allToolbarPanels.itemById(pid)
            if panel:
                existing_ctrl = panel.controls.itemById(CMD_ID)
                if not existing_ctrl:
                    panel.controls.addCommand(cmd_def)
                added = True
                log(f'Added command to panel: {pid}')

        if not added:
            # Fallback: add to any available panel
            if ui.allToolbarPanels.count > 0:
                panel = ui.allToolbarPanels.item(0)
                existing_ctrl = panel.controls.itemById(CMD_ID)
                if not existing_ctrl:
                    panel.controls.addCommand(cmd_def)
                log(f'Added command to fallback panel: {panel.id}')

        log('Eureka DFM 3.0 started successfully')
        try:
            cmd_def.execute()
            log('Auto-executed command definition')
        except Exception as e:
            log(f'Failed to auto-execute command: {e}')

    except Exception:
        log(f'run() error:\n{traceback.format_exc()}')
        if ui:
            ui.messageBox(f'Eureka DFM Add-in failed to start:\n{traceback.format_exc()}')


def stop(context):
    """Add-in cleanup — removes palette, command, and toolbar items."""
    global handlers, palette
    try:
        log('Eureka DFM 3.0 stopping')

        # Clear overlays
        try:
            clear_face_overlay()
        except Exception:
            pass

        # Remove palette
        p = ui.palettes.itemById(PALETTE_ID)
        if p:
            p.deleteMe()
            palette = None

        # Remove command from panels
        panel_ids = ['SolidScriptsAddinsPanel', 'SolidAddinsPanel', 'ToolsAddinsPanel']
        for pid in panel_ids:
            panel = ui.allToolbarPanels.itemById(pid)
            if panel:
                ctrl = panel.controls.itemById(CMD_ID)
                if ctrl:
                    ctrl.deleteMe()

        # Remove command definition
        cmd_def = ui.commandDefinitions.itemById(CMD_ID)
        if cmd_def:
            cmd_def.deleteMe()

        handlers = []
        log('Eureka DFM 3.0 stopped')

    except Exception:
        log(f'stop() error:\n{traceback.format_exc()}')
        if ui:
            ui.messageBox(f'Add-in stop failed:\n{traceback.format_exc()}')
