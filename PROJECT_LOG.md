# Eureka DFM 3.0 — Project Log

> **Living document** — updated after every session.  
> The AI model must update this file whenever it makes changes, fixes errors, or discovers new issues.

---

## Project Overview

| Item | Value |
|------|-------|
| **Product** | Eureka DFM 3.0 — SolidWorks DFM Validation Add-in |
| **Architecture** | C# SolidWorks Add-in (TaskPane UI) <-> Python FastAPI backend (port 8001) |
| **AI Stack** | GNN (PyTorch/ONNX) for face-level risk + Gemini 2.5 Flash for plain-English enrichment |
| **Rules Engine** | Python rules for Injection Moulding, Die Casting (Al/Zn/Mg), GD&T, Assembly |
| **Backend start** | `.\start_backend.ps1` from `f:\Varroc\` |
| **Add-in rebuild** | `.\build.ps1` from `f:\Varroc\addin\EurekaAddin\` |

---

## Session Log

---

### Session 3 — 2026-06-11 (Gemini GNN Anomaly Explanation Integration)

#### Objective
Complete the Gemini AI integration for GNN geometric anomaly explanations. When the GNN part-level risk score exceeds 0.3, Gemini 2.5 Flash now diagnoses the multi-feature interaction and returns a 3-bullet-point plain-English explanation stored in `gnn_anomaly["gemini_explanation"]`. When the score exceeds 0.6, a `GNN-ANOMALY-001` violation is injected so the explanation appears in the SolidWorks add-in's detail card and all involved faces are highlighted ORANGE.

#### Changes Made

##### `f:\Varroc\backend\ml\anomaly_explain.py`
- **Removed `google-generativeai` SDK** — eliminated the `FutureWarning`-generating import of `google.generativeai`. The module now uses `requests.post` REST calls directly, matching the pattern in `gemini_dfm_service.py`.
- **Updated model** — changed from deprecated `gemini-1.5-flash` to `gemini-2.5-flash`.
- **Rate-limit retries** — added exponential backoff (1s → 2s → 4s) on HTTP 429 responses, up to 4 attempts.
- **Robust face formatting** — `_build_gemini_prompt()` now guards every `Optional[float]` field (thickness, draft, radius, depth, width) with `None` checks, formatting them as `"N/A"` when absent.
- **Process-aware prompt** — prompt now maps process IDs (`injection_moulding`, `die_cast_al`, etc.) to friendly names ("Injection Moulding", "Die Casting", "CNC Machining") and tailors the root-cause explanation accordingly.

##### `f:\Varroc\backend\api\router_validate.py`
- **Import `explain_anomaly`** — added `from ..ml.anomaly_explain import explain_anomaly`.
- **`gnn_anomaly_data` dict** — replaced the minimal `{"gnn_risk_score": ..., "face_scores": ...}` payload with a rich dict that is progressively populated as GNN inference, then Gemini explanation, complete. `face_scores` is always preserved.
- **Gemini invocation gate** — if `gnn_score > 0.3`, calls `explain_anomaly(gnn_score, part)` and merges all returned keys except `face_scores` into `gnn_anomaly_data`.
- **`GNN-ANOMALY-001` injection** — if `gnn_score > 0.6`, appends a part-level `Violation(rule_id="GNN-ANOMALY-001", severity=WARNING, highlight_color="ORANGE")` whose `plain_english` is the Gemini explanation and `face_ids` are the involved faces detected by the heuristic (falling back to all faces scoring > 0.6).

##### `f:\Varroc\backend\api\router_report.py`
- **AI Anomaly Diagnosis section** — after the executive summary block, if `result.gnn_anomaly` contains a non-null `gemini_explanation`, appends a `### AI Anomaly Diagnosis` heading followed by the explanation text.

#### Verification

| Test | Result |
|------|--------|
| `test_gnn_explain.py` unit test (mocked GNN + Gemini) | **PASS** — all assertions green |
| `gnn_anomaly["gemini_explanation"]` populated | ✓ |
| `face_scores` preserved inside `gnn_anomaly` | ✓ |
| `GNN-ANOMALY-001` violation present, severity=WARNING, color=ORANGE | ✓ |
| Report contains `### AI Anomaly Diagnosis` | ✓ |
| Live Gemini REST API (direct ping) | **200 OK** — `gemini-2.5-flash` responding |

---

### Session 2 — 2026-06-11 (Dynamic Layout + Loop Optimization + Auto Port Kill + Delta Sign Fix)


#### Changes Made

##### `f:\Varroc\addin\EurekaAddin\TaskPane.cs`
- **Dynamic layout helper** — Created `LayoutPostGridControls()` to dynamic-position all post-grid controls (Violations Detail Card, "What's Looking Good" Panel, and Export PDF Button) based on visibility. Eliminates blank gaps and overlaps.
- **Delta formatting bug** — Fixed a bug where negative values were prefixed with `+` unconditionally (e.g. producing `+-0.15 mm`).

##### `f:\Varroc\addin\EurekaAddin\SwAddin.cs`
- **Highlight loop optimization (T5)** — Refactored `ApplyFaceHighlights()` to build a single face-to-color lookup dictionary in a single pass. Replaced the nested inner loop over all violations.

##### `f:\Varroc\start_backend.ps1`
- **Auto port-kill (T3 / N5)** — Added PowerShell code to query and terminate any process listening on port 8001 using `Get-NetTCPConnection` before launching uvicorn.

---

### Session 1 — 2026-06-11 (UI Fix + Gemini Key + Enrichment Upgrade)

#### Changes Made

##### `f:\Varroc\addin\EurekaAddin\TaskPane.cs`
- **Health chips clipping** — `BuildHealthChip()` had a hardcoded 80px width. Four chips at 88px spacing overflowed ContentWidth (352px) and labels ("Critical", "At Risk", "Watch", "Good") were truncated to "tical", "Risk", "atch", "d".
  - **Fix:** Added `width` parameter to `BuildHealthChip()`; chips now computed as `(cw-6)/4` ~86px each with 2px gaps.
- **Legend items overlapping** — "GNN Risk" at x=88 and "Watch" at x=168 with 80px columns were overlapping.
  - **Fix:** Legend items now evenly distributed at `cw/4` intervals.
- **Score card too cramped** — Height 78->84px; `_lblViolationStats` width `cw-100`->`cw-90`.
- **Review banner** — Added warning prefix; fixed text clipping with explicit `Size`.
- **Detail panel clipping** — Height 128->148px; issue text box 32->42px; fix box 34->38px.
- **Violations grid columns** — Severity 72->80px; Delta 48->54px.
- **"What's Good" section added** — New `_pnlGoodDesign` (green-tinted card) + `_flpGoodItems` (FlowLayoutPanel) appear below violations when backend returns `passed_checks`. Up to 8 items shown.
- **`C_GREEN_LIGHT`** colour added (`#E8F5E9`) for "What's Good" background.
- **Gemini badge** — Now reads `risk_summary["gemini_badge"]` from backend for richer text; colour goes muted when Gemini is inactive.
- **Delta column** — Now prefixed with `+` when positive (e.g. `+0.10`).

##### `f:\Varroc\backend\services\gemini_dfm_service.py`
- **Completely rewritten** with a much richer prompt:
  - Asks Gemini to emit **GREEN items** ("What's good") for up to 3 well-executed design aspects.
  - `plain_english`: 2-sentence max, junior-engineer reading level, explains WHAT and WHY.
  - `fix_instruction`: must include exact SolidWorks menu path and target dimension.
  - Adds `confidence` field (0.0-1.0) per item.
  - Added `RULE_LABELS` and `PASSED_LABELS` dicts for friendly rule display names.
  - Temperature lowered to 0.1 for more deterministic output.
  - Timeout raised 30s -> 45s to handle larger parts.
  - Improved logging: reports GREEN/RED counts on success.

##### `f:\Varroc\backend\api\router_validate.py`
- **Completely rewritten** to:
  - Separate Gemini response into `problem_items` and `gemini_good_items` (GREEN).
  - Build a human-friendly `passed_checks` list combining rules-engine passes and Gemini GREEN items (prefixed with checkmark).
  - Emit `risk_summary["gemini_badge"]` — rich string e.g. `"Gemini AI: enriched 3 finding(s) * 2 strength(s) identified"`.
  - Use `get_passed_label()` / `get_rule_label()` for friendly names.
  - Richer `plain_english` / `fix_instruction` fallbacks when Gemini skips a violation.

##### `f:\Varroc\.env` *(created)*
```
EUREKA_API_KEY=eureka-dev-key-change-me
GEMINI_API_KEY=AIzaSyAb8RN6JIIBh9h6NyO88OgKneLWXexLyGsiNtUdLsVhpCs0x4Yw
```

##### `f:\Varroc\start_backend.ps1`
- Added `Set-Item -Path "env:$name" -Value $value` alongside existing `SetEnvironmentVariable` — ensures current PowerShell process gets env var (Windows quirk where `[Environment]::SetEnvironmentVariable("Process")` doesn't always populate `$env:` in same scope).
- Added `$env:EUREKA_DEV_MODE = "true"` default.

---

#### Errors Encountered & Fixed

| # | Error | Root Cause | Fix Applied |
|---|-------|------------|-------------|
| E1 | Build: `CS0579 Duplicate AssemblyFileVersionAttribute` | `Models.cs` has `[assembly: AssemblyVersion]` AND csproj auto-generates it | **FIXED** — added `<GenerateAssemblyInfo>false</GenerateAssemblyInfo>` to `EurekaAddin.csproj` |
| E2 | Build: `CS7027 Error signing — EurekaAddin.snk not found` | Strong-name signing key missing from repo | **FIXED** — set `<SignAssembly>false</SignAssembly>` in `EurekaAddin.csproj` |
| E3 | `gemini_configured: false` even with key in `.env` | Old server (PID 43548) held port 8001; `Start-Process` creates isolated env | **FIXED** — killed port-8001 PID; start uvicorn in same shell with explicit `$env:` vars |
| E4 | `[Errno 10048] Only one usage of socket address` | Second uvicorn bound to port already held by zombie | **FIXED** — `netstat -ano` + `Stop-Process` to kill occupying PID |
| E5 | `MSB4803: RegisterAssembly not supported on .NET Core MSBuild` | `dotnet build` uses .NET Core MSBuild which can't do COM registration | **WORKED AROUND** — use `build.ps1` which calls `csc.exe` from VS 2022 Build Tools directly |
| E6 | `CS0103: The name 'cw' does not exist in the current context` (TaskPane.cs:1092, 1099) | `cw` is a local var in `BuildUI()`; used inside `UpdateResults()` which is a different method | **FIXED** — replaced `cw` with `ContentWidth` (class-level property) |
| E7 | `RegAsm : error RA0000 — registry access denied` | COM registration requires admin | **Known** — run `build.ps1` as Administrator to complete COM registration |
| E8 | Delta value sign display error | Delta formatted unconditionally with `+` prefix | **FIXED** — added conditional formatting check for positive values only |

---

#### Final Status After Session 2

| Component | Status |
|-----------|--------|
| Backend server | RUNNING on `http://localhost:8001` |
| GNN inference | Active — `pytorch` path, threshold `0.200` |
| Gemini enrichment | **ACTIVE** — `gemini_configured: true` |
| "What's Good" section | Implemented in TaskPane + dynamically laid out |
| Health chips | Fixed — no more label truncation |
| Legend items | Fixed — evenly spaced |
| Add-in DLL (`EurekaAddin.dll`) | **BUILD SUCCESS** — deployed to SolidWorks addins folder |
| COM registration | Active — CLSID `{E7A8B9C0-D1E2-F3A4-B5C6-D7E8F9A0B1C2}` registered in `HKLM` |

---

## Outstanding TODOs / Known Issues

### Should Fix (Quality / UX)

| # | Issue | File | Notes |
|---|-------|------|-------|
| T4 | Face thickness is a proxy (`sqrt(area) * 0.08`) — ~60% accurate | `SwAddin.cs` `BuildPartMetadata()` | Use SolidWorks ray-cast or `IFace2` APIs for true thickness. Low priority until API is confirmed available in license. |
| T6 | Export saves naive Markdown->HTML (string replace) | `SwAddin.cs` `ExportPDF()` | Use Markdig NuGet package for proper HTML rendering |

### Nice to Have

| # | Feature | Notes |
|---|---------|-------|
| N1 | `--reload` flag in dev uvicorn | Auto-reloads backend Python on file save |
| N2 | Gemini response caching by part hash | Avoid repeated API calls for same geometry |
| N3 | Confidence bar in detail panel | Show Gemini's `confidence` field as visual bar |
| N4 | Scrollable passed_checks list | Currently capped at 8 items with `.Take(8)` |

---

## Architecture Reference

```
f:\Varroc\
|- .env                         <- API keys (GEMINI_API_KEY, EUREKA_API_KEY)
|- start_backend.ps1            <- Run this to start the backend
|- PROJECT_LOG.md               <- This file
|- addin\EurekaAddin\
|   |- TaskPane.cs              <- All UI logic (1100+ lines)
|   |- SwAddin.cs               <- SolidWorks COM add-in, face highlighting
|   |- Models.cs                <- C# data models (Violation, ValidationResult...)
|   |- RestClient.cs            <- HTTP client to FastAPI
|   |- OverlayRenderer.cs       <- SolidWorks viewport overlay
|   |- build.ps1                <- Build the .DLL (currently BLOCKED by T1+T2)
|   `- EurekaAddin.csproj       <- Project file
`- backend\
    |- main.py                  <- FastAPI app, .env loader, middleware
    |- api\
    |   |- router_validate.py   <- POST /validate (rules + GNN + Gemini)
    |   |- router_fix.py        <- POST /fix-suggestion
    |   `- router_report.py     <- POST /report
    |- rules\
    |   |- engine.py            <- RulesEngine register/validate/score
    |   |- injection.py         <- 14 injection moulding rules (INJ-*)
    |   |- die_casting.py       <- 8 die casting rules (DC-*)
    |   |- gdt.py               <- GD&T rules (GDT-*)
    |   `- assembly.py          <- Assembly rules (ASM-*)
    `- services\
        |- gemini_dfm_service.py <- Gemini 2.5 Flash enrichment + GREEN items
        `- __init__.py           <- GNN engine loader (PyTorch / ONNX)
```

---

## How to Start / Rebuild

### Start Backend (no rebuild needed)
```powershell
cd f:\Varroc
.\start_backend.ps1
# Verify: curl http://localhost:8001/health -> gemini_configured should be true
```

### Fix Build Errors T1+T2 Then Rebuild Add-in
1. Open `f:\Varroc\addin\EurekaAddin\EurekaAddin.csproj`
2. Inside the first `<PropertyGroup>` block, add:
   ```xml
   <GenerateAssemblyInfo>false</GenerateAssemblyInfo>
   <SignAssembly>false</SignAssembly>
   ```
3. Then run:
   ```powershell
   cd f:\Varroc\addin\EurekaAddin
   .\build.ps1
   ```
4. Reload add-in in SolidWorks: **Tools -> Add-ins -> Eureka DFM -> Reload**

---

## API Quick Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | None | Server status + `gemini_configured` flag |
| `/processes` | GET | None | List supported manufacturing processes |
| `/validate` | POST | `x-api-key` | Full DFM validation (rules + GNN + Gemini) |
| `/fix-suggestion` | POST | `x-api-key` | AI fix for a single violation |
| `/report` | POST | `x-api-key` | Generate Markdown/HTML report |

**Default API key:** `eureka-dev-key-change-me`  
**Gemini key:** set in `.env` as `GEMINI_API_KEY`
