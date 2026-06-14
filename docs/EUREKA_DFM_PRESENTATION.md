# Eureka DFM 3.0 — System Presentation

> **AI-Powered Design for Manufacturability Validation**  
> SolidWorks Add-in × Python FastAPI × GNN Neural Network × Gemini 2.5 Flash

---

## 1. Executive Summary

Eureka DFM 3.0 is a real-time manufacturability validation system that lives **inside SolidWorks**. When an engineer clicks **Analyse Part**, the system:

1. Extracts geometry data from every face of the CAD part
2. Runs a fast rule-based engine (designed to scale to 150+ rules, with 15+ core rules implemented for competition scope)
3. Passes all face features through a **Graph Neural Network** trained on real defect data
4. Sends the results to **Gemini 2.5 Flash** for plain-English diagnosis and fix instructions
5. Highlights every problematic face in the SolidWorks viewport with colour-coded overlays
6. Displays a structured violation report with actionable fixes — in under 10 seconds

**Problem it solves:** Design engineers typically discover manufacturability problems only when the part reaches the toolroom — costing weeks of rework. Eureka DFM catches these issues at the design stage, inside the tool the engineer already uses.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SolidWorks Desktop                            │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  EurekaAddin.dll  (C# COM Add-in, TaskPane UI)               │   │
│  │                                                               │   │
│  │  ┌────────────┐   ┌───────────────┐   ┌──────────────────┐  │   │
│  │  │ SwAddin.cs │   │  TaskPane.cs  │   │OverlayRenderer.cs│  │   │
│  │  │ Face data  │──▶│ Violations UI │   │ Face highlights  │  │   │
│  │  │ extraction │   │ Score card    │   │ (RED/ORANGE/     │  │   │
│  │  │ REST calls │   │ Detail panel  │   │  YELLOW/GREEN)   │  │   │
│  │  └────────────┘   └───────────────┘   └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │ HTTP POST (JSON)                           │
└──────────────────────────┼──────────────────────────────────────────┘
                           │ x-api-key header
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Python FastAPI Backend  (localhost:8001)                 │
│                                                                       │
│   POST /validate                                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  1. Input validation & process alias normalisation            │  │
│   │  2. Rules Engine (150+ design scale / 15+ core implemented)  →│  │
│   │  3. GNN Inference (PyTorch)       →  gnn_score, face_scores  │  │
│   │  4. Gemini Enrichment             →  plain_english, fixes     │  │
│   │  5. GNN Anomaly Explanation       →  gemini_explanation       │  │
│   │  6. GNN-ANOMALY-001 injection     →  if gnn_score > 0.6      │  │
│   │  7. Face health classification    →  critical/at_risk/watch   │  │
│   │  8. Assemble ValidationResult     →  return JSON              │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│   POST /report  →  Markdown report with AI Anomaly Diagnosis section │
│   POST /fix-suggestion  →  Deep-dive fix for single violation        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┴─────────────────┐
          ▼                                   ▼
┌──────────────────┐               ┌─────────────────────┐
│  Gemini 2.5 Flash│               │  GNN Model          │
│  REST API        │               │  (PyTorch, CPU)     │
│  googleapis.com  │               │  12-feature vectors │
│                  │               │  per face node      │
│  • DFM enrichment│               │  Graph edges from   │
│  • Anomaly       │               │  face adjacency     │
│    diagnosis     │               │                     │
│  • Fix instruct. │               │                     │
└──────────────────┘               └─────────────────────┘
```

---

## 3. Components In Detail

### 3.1 SolidWorks Add-in (`EurekaAddin.dll`)

The add-in is a **C# COM component** registered in the Windows registry. SolidWorks loads it at startup, adding the "Eureka DFM" TaskPane on the right side of the window.

#### `SwAddin.cs` — The Bridge
- Hooks into SolidWorks via the `ISwAddin` COM interface
- Iterates every `IFace2` object in the active part document
- For each face, computes: area, centroid, normal vector, curvature, and uses a geometric proxy for wall thickness (`√area × 0.08`)
- Identifies SolidWorks feature names (`Boss-Extrude1`, `Fillet2`, etc.) from the feature manager tree
- Bundles everything into a `PartMetadata` JSON payload and POSTs to `/validate`
- After validation, calls `ApplyFaceHighlights()` using colour codes mapped from `highlight_color` (RED → crimson, ORANGE → orange, YELLOW → amber, GREEN → lime)

#### `TaskPane.cs` — The UI (1,100+ lines)
The UI is built entirely in WinForms within the SolidWorks TaskPane host:

| Section | Description |
|---------|-------------|
| **Score Card** | Circular gauge 0–100 score; colour changes red → amber → green |
| **Health Chips** | Four chips: Critical / At Risk / Watch / Clean face counts |
| **Gemini Badge** | "Gemini AI: enriched N finding(s) · M strength(s) identified" |
| **Violations Grid** | DataGridView: Severity, Rule ID, Description, Measured, Required, Δ, Fix |
| **Detail Panel** | Slides in on row click; shows `plain_english` and `fix_instruction` |
| **What's Good Panel** | Green-tinted card listing Gemini's positive design observations |
| **Export PDF Button** | Calls `/report` and renders Markdown → HTML for printing |

#### `OverlayRenderer.cs` — Face Highlighting
Implements `ISwAddinCallback` to draw transparent colour overlays on selected faces in the SolidWorks viewport.

---

### 3.2 FastAPI Backend

#### `main.py` — App Bootstrap
- Loads `.env` via `python-dotenv` so `GEMINI_API_KEY` and `EUREKA_API_KEY` are always available
- Registers three routers: `/validate`, `/fix-suggestion`, `/report`
- Applies API key middleware (except `/health` and `/processes`)
- Loads the GNN model into memory at startup via the `DFMInferenceEngine`

#### `router_validate.py` — The Orchestrator
A single `POST /validate` request triggers an 8-step pipeline:

**Step 1 — Input Validation**
Checks face count ≤ 500, process and material present, wall thickness > 0.

**Step 2 — Rules Engine**
`engine.validate(part)` runs all registered rules for the given process.

**Step 3 — GNN Inference**
`gnn_engine.predict(part)` builds a graph where nodes are faces (12-feature vectors) and edges connect adjacent faces. Returns `part_risk_score` (0–1) and `face_scores` dict.

**Step 4 — Gemini Enrichment**
`enrich_validation_with_gemini()` sends all violations + high-risk faces to Gemini 2.5 Flash. Returns `plain_english`, `fix_instruction`, `highlight_color`, `confidence`, and GREEN positive items.

**Step 5 — GNN Anomaly Explanation**
If `gnn_score > 0.3`, calls `explain_anomaly(gnn_score, part)`:
- Runs heuristic multi-feature analysis (thin wall + low draft, thick wall + slender core, etc.)
- Builds a process-aware prompt for Gemini 2.5 Flash
- Gets back a 3-bullet diagnosis: root cause + 2 recommended fixes
- Stored in `gnn_anomaly["gemini_explanation"]`

**Step 6 — GNN-ANOMALY-001 Injection**
If `gnn_score > 0.6`, creates a virtual `Violation`:
```python
Violation(
  rule_id         = "GNN-ANOMALY-001",
  severity        = WARNING,
  highlight_color = "ORANGE",
  face_ids        = [involved faces],
  plain_english   = gemini_explanation  # the 3-bullet AI diagnosis
)
```
This flows into the add-in's violations grid and detail panel automatically.

**Step 7 — Face Health Classification**
Every face classified into: Critical / At Risk / Watch / Clean.

**Step 8 — Assemble Result**
Returns `ValidationResult` JSON with all violations, passed checks, GNN data, face health counts, and Gemini badge text.

---

### 3.3 Rules Engine

The rules engine uses a **registration pattern** — each module registers checks at import time. The architecture is engineered to scale to **150+ rules** across various manufacturing domains, with **15+ core rules** fully implemented for the competition scope (covering Injection Moulding, Die Casting, GD&T, and Assembly).

#### Rules Coverage

| Process | Rules | Key Checks |
|---------|-------|-----------|
| **Injection Moulding** | 11 | Wall thickness min/max, uneven walls, draft angle, Class A draft, corner radii, hole size, flow length, rib thickness/height/root, boss wall/height/isolation |
| **Die Casting (Al/Zn/Mg)** | 8 | Wall thickness, draft angle, corner radii, hole size — with process-specific thresholds |
| **GD&T** | 4 | Position tolerance, cylindricity, flatness, perpendicularity |
| **Assembly** | 2 | Gap clearance, interference detection |

---

### 3.4 GNN Model Architecture

The Graph Neural Network treats the CAD part as a **graph**:
- **Nodes** = faces (each face → 12-dimensional feature vector)
- **Edges** = adjacency between all face pairs

#### 12 Node Features per Face

| # | Feature | Why it matters |
|---|---------|----------------|
| 0 | face_type (encoded) | Plane/Cylinder/Sphere affect moulding differently |
| 1 | thickness_mm | Core DFM parameter |
| 2 | radius_mm | Fillet / corner radius |
| 3 | area_mm2 | Large thin areas warp |
| 4 | depth_mm | Pocket / rib depth |
| 5 | width_mm | For aspect ratio |
| 6 | curvature_min | Surface complexity |
| 7 | curvature_max | Surface complexity |
| 8 | draft_angle_deg | Ejection risk |
| 9 | parent_wall_thickness_mm | Surrounding context |
| 10 | centroid_dist | Spatial position |
| 11 | aspect_ratio | depth / width |

#### Architecture: `DFMGNN`
```
Input (N faces × 12 features)
    ↓
GraphConv layer 1  (12 → 128)  + ReLU
    ↓
GraphConv layer 2  (128 → 128) + ReLU
    ↓
Global Mean Pool   (N faces → 1 × 128 part vector)
    ↓
Linear (128 → 1)  + Sigmoid
    ↓
part_risk_score ∈ [0, 1]
```

The model runs in **PyTorch on CPU** — no GPU required. Inference on a 50-face part: < 200ms.

---

### 3.5 Gemini 2.5 Flash Integration

Two separate Gemini calls happen per validation:

#### Call A — DFM Violation Enrichment
**When:** Always (if API key set) | **Temperature:** 0.1 | **Timeout:** 45s
**Output per violation:**
- `plain_english` — 2-sentence explanation at junior-engineer level
- `fix_instruction` — exact SolidWorks menu path + target dimension
- `highlight_color` — RED / ORANGE / YELLOW / GREEN
- `confidence` — 0.0–1.0 certainty score
- GREEN items — up to 3 positive design observations

#### Call B — GNN Anomaly Diagnosis
**When:** Only if `gnn_score > 0.3` | **Temperature:** 0.2 | **Retries:** 4 with exponential backoff

**Prompt instructs Gemini to respond in exactly 3 bullet points:**
```
• Root Cause: [what the geometric combination causes]
• Recommended Fix 1: [primary corrective action with dimension]
• Recommended Fix 2: [secondary corrective action with dimension]
```
**Model:** `gemini-2.5-flash:generateContent` via direct REST API (no SDK)

---

### 3.6 Report Generation

`POST /report` returns a Markdown document with:

1. **Executive Summary** — score, violation counts, GNN alert
2. **AI Anomaly Diagnosis** — Gemini's 3-bullet root cause analysis *(when GNN risk > 0.3)*
3. **Critical Findings** — detailed list of CRITICAL violations
4. **Warning Findings** — WARNING violations with risk scores
5. **Manufacturability Score** — interpretation and tooling readiness
6. **Recommended Actions** — top 5 violations with SolidWorks fix paths
7. **Sign-Off Checklist** — checkboxes for tooling release approval

---

## 4. End-to-End Data Flow

```
Engineer clicks "Analyse Part"
           │
           ▼
SwAddin.cs extracts IFace2 objects
  └─ face_id, face_type, area, thickness,
     draft angle, radius, depth, width,
     centroid (x/y/z), normal (x/y/z),
     curvature min/max, sw_feature_name
           │
           ▼  POST /validate
           │
    ┌──────┴──────────────────────────────────────────┐
    │  Rules engine    → violations[]                  │
    │  GNN inference   → part_risk_score, face_scores  │
    │  if score > 0.3: Gemini Call B → 3-bullet diag. │
    │  Gemini Call A:  enrich violations + GREEN items │
    │  if score > 0.6: inject GNN-ANOMALY-001          │
    │  Compute face_health, passed_checks              │
    └──────────────────────────────────────────────────┘
           │
           ▼  ValidationResult JSON
           │
    TaskPane renders score, chips, grid, badge
    User clicks violation → detail panel slides in
    OverlayRenderer colours faces in SolidWorks viewport
```

---

## 5. Manufacturability Score

Score starts at **100**; deductions applied per violation:

| Violation | Deduction |
|-----------|-----------|
| CRITICAL violation | −15 pts |
| WARNING violation | −8 pts |
| INFO violation | −3 pts |
| GNN risk > 0.8 | Additional −10 pts |

| Score | Status |
|-------|--------|
| 80–100 | ✅ Ready for tooling |
| 60–79 | ⚠️ Design changes needed |
| 40–59 | ❌ Significant redesign required |
| 0–39 | 🚫 Must redesign before tooling |

---

## 6. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **REST API over COM** | Keeps AI/ML decoupled; backend updates without DLL rebuild |
| **GNN over pure rules** | Rules check faces in isolation; GNN detects cross-feature interaction risks |
| **Gemini via REST, not SDK** | No FutureWarning, simpler retry logic, faster cold start |
| **Two separate Gemini calls** | Structured JSON output (enrichment) vs free-form 3-bullet text (anomaly) |
| **GNN-ANOMALY-001 as Violation** | Zero C# changes — reuses existing violations grid and face highlighting |
| **PyTorch on CPU** | No GPU dependency on engineering workstations |

---

## 7. Running the System

### One-Click Launch
Double-click **`START SERVER.bat`** in `f:\Varroc\`.
The launcher: kills port 8001 → loads `.env` → starts uvicorn with `--reload`.

```
API Docs:  http://localhost:8001/docs
Health:    http://localhost:8001/health
```

### SolidWorks Add-in
**Tools → Add-ins → Eureka DFM → ✓ Active** (already registered).

---

## 8. File Structure

```
f:\Varroc\
├── START SERVER.bat              ← Double-click to launch backend
├── .env                          ← GEMINI_API_KEY, EUREKA_API_KEY
│
├── addin\EurekaAddin\
│   ├── SwAddin.cs                ← COM add-in, face extraction, REST calls
│   ├── TaskPane.cs               ← All UI: score, grid, detail panel, export
│   ├── Models.cs                 ← C# data models
│   ├── RestClient.cs             ← HTTP client
│   ├── OverlayRenderer.cs        ← SolidWorks face colouring
│   └── build.ps1                 ← Rebuild DLL (run as Admin)
│
└── backend\
    ├── main.py                   ← FastAPI app, middleware
    ├── api\
    │   ├── router_validate.py    ← 8-step validation pipeline
    │   ├── router_fix.py         ← Fix suggestion endpoint
    │   └── router_report.py      ← Report generation
    ├── rules\
    │   ├── engine.py             ← Rule registration and execution
    │   ├── injection.py          ← 11 injection moulding rules
    │   ├── die_casting.py        ← 8 die casting rules
    │   ├── gdt.py                ← GD&T rules
    │   └── assembly.py           ← Assembly rules
    ├── ml\
    │   ├── gnn_model.py          ← DFMGNN PyTorch architecture
    │   ├── inference.py          ← DFMInferenceEngine
    │   └── anomaly_explain.py    ← GNN anomaly → Gemini diagnosis
    └── services\
        ├── gemini_dfm_service.py ← Violation enrichment + GREEN items
        └── __init__.py           ← GNN engine singleton
```

---

## 9. Competitive Differentiators

| Feature | Eureka DFM 3.0 | Traditional DFM Tools |
|---------|----------------|----------------------|
| **AI Root Cause** | Gemini explains *why* and *how to fix* | Rule ID only, no explanation |
| **Multi-feature GNN** | Detects interaction risks rules miss | Rules check features in isolation |
| **Face Highlighting** | Every face colour-coded in SolidWorks viewport | External reports only |
| **Plain English** | Junior-engineer-friendly explanations | Technical codes and thresholds |
| **What's Good** | Positive feedback for well-designed features | No positive feedback |
| **Process-aware AI** | Prompt tailored to Injection Moulding vs Die Casting | Generic rules |
| **Inside SolidWorks** | No file export, no context switch | Separate application |
| **Open architecture** | Add new rules or ML models easily | Black-box proprietary |

---

*Eureka DFM 3.0 | GNN: DFMGNN-12D-128H | AI: Gemini 2.5 Flash | Session 3 — 2026-06-11*
