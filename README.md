# 🏭 Eureka DFM 3.0 — AI-Powered Design for Manufacturability

> **Eureka DFM** is an intelligent CAD validation system that catches manufacturability issues *before* they hit the shop floor. It combines a Graph Neural Network (GNN) for geometric risk detection with a rule-based engine and Google Gemini AI for plain-English explanations.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **GNN Risk Scoring** | Face-level and part-level geometric anomaly detection using a Graph Attention Network (GATv2) |
| **Rules Engine** | 30+ parametric rules for Injection Moulding, Die Casting (Al/Zn/Mg), CNC Machining, GD&T, and Assembly |
| **Gemini AI Enrichment** | Google Gemini 2.5 Flash generates plain-English explanations, fix instructions with exact SolidWorks menu paths, and identifies design strengths |
| **Hybrid Inference** | GNN + XGBoost ensemble for calibrated confidence scores |
| **PDF Reports** | One-click professional DFM reports with executive summary, violation details, and AI diagnosis |
| **SolidWorks Add-in** | Native C# TaskPane UI with face highlighting, health dashboard, and real-time validation |
| **Fusion 360 Add-in** | Python-based palette UI for Autodesk Fusion 360 |
| **CATIA Add-in** | C# connector for Dassault Systèmes CATIA V5 |
| **Flywheel Learning** | Feedback loop with automatic model fine-tuning and A/B registry |
| **Web Interface** | Standalone HTML interface for browser-based validation |

---

## 🏗️ Architecture

```
┌─────────────────────────┐     HTTP/JSON      ┌──────────────────────────┐
│   CAD Add-ins           │ ◄─────────────────► │   FastAPI Backend        │
│  ┌───────────────┐      │     Port 8001       │  ┌────────────────────┐  │
│  │ SolidWorks    │      │                     │  │ Rules Engine       │  │
│  │ (C# TaskPane) │      │                     │  │  • Injection Mould │  │
│  ├───────────────┤      │                     │  │  • Die Casting     │  │
│  │ Fusion 360    │      │                     │  │  • CNC Machining   │  │
│  │ (Python)      │      │                     │  │  • GD&T / Assembly │  │
│  ├───────────────┤      │                     │  ├────────────────────┤  │
│  │ CATIA V5      │      │                     │  │ GNN Inference      │  │
│  │ (C#)          │      │                     │  │  • PyTorch / ONNX  │  │
│  └───────────────┘      │                     │  │  • XGBoost Hybrid  │  │
│                         │                     │  ├────────────────────┤  │
│  ┌───────────────┐      │                     │  │ Gemini AI          │  │
│  │ Web Interface │ ─────┘                     │  │  • Enrichment      │  │
│  │ (HTML/JS)     │                            │  │  • Anomaly Explain │  │
│  └───────────────┘                            │  └────────────────────┘  │
└─────────────────────────┘                     └──────────────────────────┘
```

---

## 📁 Project Structure

```
eureka-dfm/
├── backend/                    # Python FastAPI backend
│   ├── main.py                 # App entry point, middleware, lifespan
│   ├── api/                    # API route handlers
│   │   ├── router_validate.py  # POST /validate — full DFM validation
│   │   ├── router_fix.py       # POST /fix-suggestion — AI fix advice
│   │   ├── router_report.py    # POST /report — PDF report generation
│   │   ├── router_web.py       # Web interface API endpoints
│   │   └── router_flywheel.py  # Feedback & model fine-tuning endpoints
│   ├── core/                   # Configuration, database, data models
│   │   ├── config.py           # Pydantic settings (env-based config)
│   │   ├── database.py         # Async SQLAlchemy + SQLite
│   │   └── models.py           # PartMetadata, FaceGeometry, Violation, etc.
│   ├── ml/                     # Machine learning modules
│   │   ├── gnn_model.py        # GATv2-based GNN architecture
│   │   ├── inference.py        # DFMInferenceEngine (PyTorch/ONNX/Hybrid)
│   │   ├── anomaly_explain.py  # Gemini-powered anomaly explanations
│   │   ├── feedback_store.py   # SQLite feedback & model registry
│   │   └── fine_tune_trigger.py# Background fine-tuning pipeline
│   ├── rules/                  # Parametric rule definitions
│   │   ├── engine.py           # RulesEngine — register/validate/score
│   │   ├── injection.py        # 14 injection moulding rules (INJ-*)
│   │   ├── die_casting.py      # 8 die casting rules (DC-*)
│   │   ├── cnc.py              # CNC machining rules (CNC-*)
│   │   ├── gdt.py              # GD&T rules (GDT-*)
│   │   └── assembly.py         # Assembly rules (ASM-*)
│   ├── services/               # Shared service instances
│   │   ├── __init__.py         # GNN engine singleton
│   │   └── gemini_dfm_service.py # Gemini 2.5 Flash API integration
│   ├── utils/                  # Utility modules
│   └── tests/                  # Unit tests
├── addin/                      # CAD platform add-ins
│   ├── EurekaAddin/            # SolidWorks C# add-in
│   │   ├── SwAddin.cs          # COM add-in entry point, face highlighting
│   │   ├── TaskPane.cs         # Full UI (1100+ lines) — dashboard, grid, cards
│   │   ├── Models.cs           # C# data models
│   │   ├── RestClient.cs       # HTTP client to FastAPI
│   │   ├── OverlayRenderer.cs  # Viewport overlay rendering
│   │   ├── EurekaAddin.csproj  # MSBuild project file
│   │   └── build.ps1           # Build script (requires VS Build Tools)
│   ├── FusionAddin/            # Autodesk Fusion 360 Python add-in
│   │   ├── EurekaDFM.py        # Main add-in logic
│   │   ├── palette.html        # HTML palette UI
│   │   ├── EurekaDFM.manifest  # Add-in manifest
│   │   └── install.ps1         # Installation script
│   ├── CatiaAddin/             # CATIA V5 C# connector
│   │   ├── CatiaConnector.cs   # Full CATIA automation
│   │   ├── EurekaDfmCatia.csproj
│   │   └── build.ps1
│   └── EurekaAddin.sln         # Visual Studio solution
├── scripts/                    # Training & data pipeline scripts
│   ├── train_hybrid.py         # Hybrid GNN+XGBoost training
│   ├── train_gnn.py            # GNN-only training
│   ├── build_dataset.py        # Dataset construction
│   ├── export_onnx.py          # ONNX model export
│   └── ...                     # Additional training utilities
├── data/
│   └── models/                 # Trained model weights
│       ├── gnn_model_real_cpu.pt    # Primary GNN model (PyTorch)
│       ├── gnn_model_real.onnx      # ONNX-optimized model
│       ├── hybrid_xgb.json          # XGBoost ensemble model
│       ├── threshold_real.txt       # Decision threshold
│       ├── calibration_real.json    # GNN calibration params
│       └── calibration_hybrid.json  # Hybrid calibration params
├── test_cad/                   # Sample CAD files for testing
├── docs/                       # Documentation & presentations
├── eureka_web.html             # Standalone web interface
├── start_backend.ps1           # Backend startup script (PowerShell)
├── START SERVER.bat             # One-click server launcher (Windows)
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
├── PROJECT_LOG.md              # Development session log
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- **SolidWorks 2020+** (for the SolidWorks add-in — optional)
- **Visual Studio Build Tools 2022** (for building C# add-ins — optional)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/eureka-dfm.git
cd eureka-dfm

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example env file
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac

# Edit .env and set your keys:
# EUREKA_API_KEY=your-api-key-here
# GEMINI_API_KEY=your-google-gemini-api-key
```

### 3. Start the Backend

**Option A — Batch file (Windows, recommended):**
```cmd
"START SERVER.bat"
```

**Option B — PowerShell:**
```powershell
.\start_backend.ps1
```

**Option C — Manual:**
```bash
python -m uvicorn backend.main:app --port 8001 --host 0.0.0.0
```

### 4. Verify

```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "gnn_available": true,
  "inference_mode": "hybrid",
  "gemini_configured": true
}
```

---

## 📡 API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | None | Server status, model info, Gemini status |
| `/processes` | GET | None | List supported manufacturing processes |
| `/validate` | POST | `x-api-key` | Full DFM validation (rules + GNN + Gemini) |
| `/fix-suggestion` | POST | `x-api-key` | AI-powered fix suggestion for a violation |
| `/report` | POST | `x-api-key` | Generate PDF DFM report |
| `/web/presets` | GET | None | Web UI manufacturing process presets |
| `/flywheel` | GET | None | Model metrics & fine-tuning status |
| `/feedback` | POST | `x-api-key` | Submit engineer feedback for learning |

**Authentication:** Include `x-api-key` header with your API key (default: `eureka-dev-key-change-me`).

**Interactive Docs:** When `EUREKA_DEV_MODE=true`, visit `http://localhost:8001/docs` for Swagger UI.

---

## 🧠 AI Stack

### Graph Neural Network (GNN)
- **Architecture:** GATv2Conv (Graph Attention Network v2) with global mean+max pooling
- **Input:** Per-face geometric features (area, curvature, thickness, draft angle, etc.)
- **Output:** Part-level risk score + per-face risk scores
- **Training:** Trained on ABC Dataset (real CAD geometry) with focal loss

### Hybrid Inference
- GNN embeddings are passed through an **XGBoost** classifier for calibrated probability scores
- Platt scaling calibration for reliable confidence thresholds

### Gemini AI Integration
- **Model:** Google Gemini 2.5 Flash
- **Enrichment:** Plain-English explanations, SolidWorks-specific fix instructions
- **Anomaly Diagnosis:** Multi-feature interaction analysis when GNN risk > 0.3
- **Strengths Detection:** Identifies well-designed aspects of the part

---

## 🔧 Building the SolidWorks Add-in

```powershell
cd addin\EurekaAddin
.\build.ps1
# Requires Visual Studio 2022 Build Tools
# Run as Administrator for COM registration
```

---

## 🚫 Files Not Included & Why

The following files and directories from the development environment are **intentionally excluded** from this repository:

### 🔐 Secrets & Environment

| File | Reason |
|------|--------|
| `.env` | Contains **API keys** (`GEMINI_API_KEY`, `EUREKA_API_KEY`). Never commit secrets to version control. Use `.env.example` as a template instead. |

### 🗄️ Databases & Runtime State

| File | Reason |
|------|--------|
| `eureka.db` | SQLite database generated at runtime by the backend. Auto-created on first startup via `init_db()`. |
| `data/feedback/feedback.db` | SQLite database storing engineer feedback submissions. Contains user-generated data, not source code. |
| `server.log` | Runtime server log. Regenerated every time the backend starts. |

### 📦 Build Artifacts & Compiled Binaries

| File / Directory | Reason |
|------------------|--------|
| `addin/EurekaAddin/bin/` | C# compiled output (`.dll`, `.pdb`). Rebuilt by running `build.ps1`. |
| `addin/EurekaAddin/obj/` | C# intermediate build files. Auto-generated by MSBuild. |
| `addin/CatiaAddin/bin/` | Same as above — CATIA add-in compiled output. |
| `addin/CatiaAddin/obj/` | Same as above — CATIA intermediate build files. |
| `Newtonsoft.Json.dll` | Third-party NuGet DLL. Should be restored via NuGet package manager, not checked into source. |
| `test_deser.exe` | Compiled test executable. Build artifact, not source code. |
| `test_bypass.onnx` | Throwaway ONNX file used during development testing. Not needed for production. |

### 🐍 Python Cache

| Directory | Reason |
|-----------|--------|
| `__pycache__/` (all instances) | Python bytecode cache (`.pyc` files). Auto-generated by the Python interpreter. Present in root, `backend/`, `scripts/`, and every sub-package. |
| `.cursor/` | IDE workspace settings (Cursor editor). Machine-specific, not part of source code. |

### 📊 Training Data (Large Files)

| File / Directory | Size | Reason |
|------------------|------|--------|
| `data/abc_raw/` | **~8.5 GB** | Raw ABC Dataset archives (`.7z` files with CAD feature vectors, metadata, and statistics). Far too large for GitHub (100 MB file limit). Should be downloaded separately from the [ABC Dataset](https://deep-geometry.github.io/abc-dataset/). |
| `data/processed/` | **~165 MB** | Preprocessed training tensors (`.pt` files), label CSVs, and pickle files. Generated by running scripts in `scripts/`. |
| `data/synthetic/` | — | Synthetic training data directory. Empty in current build, reserved for future data augmentation. |
| `data/labels/` | — | Label directory. Empty — labels are stored in `data/processed/` as CSVs. |
| `data/feedback/parts/` | — | Uploaded CAD part files from feedback submissions. User-generated content. |
| `data/*.log` | — | Training log files (`gate_output.log`, `rebuild_14d.log`, `train_14d.log`). Output of training runs, not source code. |

### 🧠 Redundant / Backup Model Files

| File | Size | Reason |
|------|------|--------|
| `data/models/gnn_model.pt` | 27 KB | Early-stage GNN model (synthetic data only). Superseded by `gnn_model_real_cpu.pt`. |
| `data/models/gnn_model_real.pt` | 27 KB | GPU-only model variant. The CPU-compatible version (`gnn_model_real_cpu.pt`) is included instead. |
| `data/models/gnn_focal_10d.pt` | 128 KB | Experimental 10-dimensional focal-loss model. Not used in production. |
| `data/models/gnn_model_real_cpu.pt.bak` | 299 KB | Backup of a previous model version. |
| `data/models/gnn_model_real.onnx.stale8d` | 34 KB | Stale ONNX export from an older 8-dimensional model. |
| `data/models/calibration_focal_10d.json` | < 1 KB | Calibration for the experimental focal model. |
| `data/models/calibration_curve.png` | 32 KB | Visualization image from calibration analysis. Not needed for inference. |
| `data/models/hybrid_feature_importance.png` | 35 KB | XGBoost feature importance chart. Training diagnostic, not source code. |

### 🧪 Test & Debug Artifacts

| File | Reason |
|------|--------|
| `test_fallback_report.pdf` | Generated test PDF — output of `test_pdf_generation.py`. |
| `test_generated_report.pdf` | Generated test PDF — output of report endpoint testing. |
| `test_report.pdf` | Generated test PDF — manual testing output. |
| `test_snapshots_report.pdf` | Generated test PDF — snapshot-based report test. |
| `test_deser.cs` | One-off C# deserialization test script. Not part of the product. |
| `rendered_page_1.png`, `rendered_page_2.png` | PNG renders from `render_pdf_to_images.py`. Debug output. |
| `inspect_pdf_images.py` | One-off debug script for inspecting images embedded in PDFs. |
| `render_pdf_to_images.py` | One-off utility script for PDF-to-image conversion testing. |
| `verify_images_in_pdf.py` | One-off debug script for verifying PDF image embedding. |
| `verify_live.py` | One-off integration test script for live API verification. |
| `test_pdf_generation.py` | Standalone test script (not part of the `backend/tests/` suite). |

### 🔧 Machine-Specific Registry & Build Scripts

| File | Reason |
|------|--------|
| `_clean_clsid.ps1` | PowerShell script to clean Windows COM registry entries. Machine-specific, requires admin privileges. |
| `_fix_registry.ps1` | PowerShell script to fix COM registration issues. Machine-specific. |
| `_register_admin.ps1` | PowerShell script for admin-level COM registration. Machine-specific. |
| `register_addin.ps1` | SolidWorks add-in COM registration script. Machine-specific paths and CLSIDs. |
| `run_build.bat` | Wrapper batch file for build. Machine-specific paths. |
| `run_register_addin.bat` | Wrapper batch file for registration. Machine-specific paths. |
| `build_log.txt` | Output log from the last C# build run. |
| `reg_log.txt` | Output log from COM registration. |
| `register_log.txt` | Output log from add-in registration. |

### ✅ What IS Included (for reference)

The **6 trained model files** needed for inference are included:
- `gnn_model_real_cpu.pt` — Primary GNN model (PyTorch, CPU-compatible)
- `gnn_model_real.onnx` — ONNX-optimized model for faster inference
- `hybrid_xgb.json` — XGBoost ensemble model
- `threshold_real.txt` — Decision threshold value
- `calibration_real.json` — GNN probability calibration parameters
- `calibration_hybrid.json` — Hybrid model calibration parameters

---

## 📜 License

This project is proprietary. All rights reserved.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
