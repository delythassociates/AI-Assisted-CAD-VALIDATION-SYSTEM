using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using SolidWorks.Interop.swpublished;

namespace EurekaAddin
{
    [Guid("E7A8B9C0-D1E2-F3A4-B5C6-D7E8F9A0B1C2")]
    [ComVisible(true)]
    public class SwAddin : ISwAddin
    {
        private SldWorks _swApp;
        private TaskPane _taskPane;
        private RestClient _client;
        private Control _marshalControl;
        private readonly Dictionary<string, IFace2> _faceRegistry = new Dictionary<string, IFace2>();

        private OverlayRenderer _renderer;
        private ValidationResult _lastResult;
        private PartMetadata _lastPartMetadata;

        private static readonly List<ProcessInfo> _defaultProcesses = new List<ProcessInfo>
        {
            new ProcessInfo { id = "injection_moulding", name = "Injection Moulding",        description = "Thermoplastic part design",      rule_count = 11 },
            new ProcessInfo { id = "die_cast_al",        name = "Die Casting (Aluminium)",   description = "Aluminium die-cast design",     rule_count = 8  },
            new ProcessInfo { id = "die_cast_zn",        name = "Die Casting (Zinc)",        description = "Zinc die-cast design",          rule_count = 8  },
            new ProcessInfo { id = "die_cast_mg",        name = "Die Casting (Magnesium)",   description = "Magnesium die-cast design",     rule_count = 8  }
        };

        private void Log(string msg)
        {
            try 
            { 
                string logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hermes_addin.log");
                System.IO.File.AppendAllText(logPath, string.Format("{0:HH:mm:ss} {1}\r\n", DateTime.Now, msg)); 
            } 
            catch { }
        }

        public bool ConnectToSW(object ThisSwApp, int Cookie)
        {
            Log(string.Format("EUREKA DFM ConnectToSW called, Cookie={0}", Cookie));

            _client = new RestClient();
            Log("Created RestClient");

            _marshalControl = new Control();
            _marshalControl.CreateControl();
            Log("Created marshal control");

            // Defer ALL SW API calls to avoid COM threading issues
            _marshalControl.BeginInvoke((Action)(() =>
            {
                try
                {
                    _swApp = (SldWorks)ThisSwApp;
                    Log("Got SW app");

                    _renderer = new OverlayRenderer(_swApp);
                    Log("Created OverlayRenderer");

                    _taskPane = new TaskPane();
                    _taskPane.Dock = DockStyle.Fill;
                    
                    // Wire TaskPane events
                    _taskPane.ValidateClicked += (sender, e) => {
                        _marshalControl.BeginInvoke((Action)(async () => await RunValidation()));
                    };

                    _taskPane.ClearClicked += (sender, e) => {
                        IModelDoc2 doc = (IModelDoc2)_swApp.IActiveDoc2;
                        if (doc != null) {
                            _renderer.ClearOverlay(doc);
                            _renderer.ClearHighlight(doc);
                        }
                        _taskPane.ClearResults();
                    };

                    _taskPane.FaceSelected += (sender, faceId) => {
                        IModelDoc2 doc = (IModelDoc2)_swApp.IActiveDoc2;
                        if (doc == null) return;
                        if (string.IsNullOrEmpty(faceId)) {
                            _renderer.ClearHighlight(doc);
                            return;
                        }
                        Violation v = _lastResult?.Violations?.FirstOrDefault(x => x.FaceId == faceId);
                        if (v != null) _renderer.HighlightViolation(v, doc);
                    };


                    _taskPane.ExportReportClicked += (sender, e) => {
                        if (_lastResult != null) {
                            _marshalControl.BeginInvoke((Action)(async () => await ExportPDF(_lastResult)));
                        }
                    };

                    _taskPane.FeedbackClicked += (sender, isAccurate) => {
                        _marshalControl.BeginInvoke((Action)(async () => await SubmitFeedback(isAccurate)));
                    };

                    // Display UserControl in SolidWorks TaskPane
                    string iconPath = CreateIcon();
                    var view = _swApp.CreateTaskpaneView2(iconPath ?? "", "   Eureka DFM 3.0");
                    var taskPaneView = view as ITaskpaneView;
                    if (taskPaneView != null)
                    {
                        taskPaneView.DisplayWindowFromHandlex64(_taskPane.Handle.ToInt64());
                    }
                    Log("Created and hosted task pane UserControl");

                    // Register callback handler
                    try { _swApp.SetAddinCallbackInfo2(0, this, Cookie); } catch { Log("SetAddinCallbackInfo2 failed"); }
                    Log("SetAddinCallbackInfo2 done");
                    int cmdId = _swApp.AddMenuItem4((int)swDocumentTypes_e.swDocPART, Cookie, "EUREKA Validate Part@1", 0, "RunValidationMenu", "", "", "");
                    Log(string.Format("Added menu item, cmdId={0}", cmdId));

                    // Subscribe to document activation and modification change events
                    SubscribeToDocEvents();

                    System.Threading.Tasks.Task.Run(async () =>
                    {
                        await CheckAndStartBackend();
                    });

                    Log("ConnectToSW success");
                }
                catch (Exception ex)
                {
                    Log(string.Format("ConnectToSW INIT ERROR: {0}", ex.ToString()));
                }
            }));

            return true;
        }

        private int _debounceEpoch = 0;
        private PartDoc _subscribedDoc = null;

        private void SubscribeToDocEvents()
        {
            try
            {
                if (_swApp == null) return;
                
                // Track active document changes
                _swApp.ActiveDocChangeNotify += OnActiveDocChangeNotify;
                Log("Subscribed to ActiveDocChangeNotify");

                // Check if there is an existing active doc to hook onto
                HookActiveDoc();
            }
            catch (Exception ex)
            {
                Log("SubscribeToDocEvents error: " + ex.Message);
            }
        }

        private void HookActiveDoc()
        {
            try
            {
                UnhookDoc();

                var doc = _swApp.IActiveDoc2 as ModelDoc2;
                if (doc != null && doc.GetType() == (int)swDocumentTypes_e.swDocPART)
                {
                    _subscribedDoc = doc as PartDoc;
                    if (_subscribedDoc != null)
                    {
                        // Hook into ModifyNotify
                        _subscribedDoc.ModifyNotify += OnModifyNotify;
                        Log("Hooked ModifyNotify for part document");
                    }
                }
            }
            catch (Exception ex)
            {
                Log("HookActiveDoc error: " + ex.Message);
            }
        }

        private void UnhookDoc()
        {
            try
            {
                if (_subscribedDoc != null)
                {
                    _subscribedDoc.ModifyNotify -= OnModifyNotify;
                    Log("Unhooked ModifyNotify for part document");
                    _subscribedDoc = null;
                }
            }
            catch (Exception ex)
            {
                Log("UnhookDoc error: " + ex.Message);
            }
        }

        private int OnActiveDocChangeNotify()
        {
            Log("Active document changed");
            _marshalControl?.BeginInvoke((Action)(() =>
            {
                HookActiveDoc();
            }));
            return 0;
        }

        private int OnModifyNotify()
        {
            Log("Document modified");
            return 0;
        }

        private void TriggerDebouncedValidation()
        {
            int epoch = System.Threading.Interlocked.Increment(ref _debounceEpoch);
            Task.Delay(2000).ContinueWith(t =>
            {
                if (epoch == Volatile.Read(ref _debounceEpoch))
                {
                    _marshalControl?.BeginInvoke((Action)(async () =>
                    {
                        Log("Debounce complete. Triggering RunValidation.");
                        await RunValidation();
                    }));
                }
            });
        }

        public bool DisconnectFromSW()
        {
            try
            {
                _marshalControl?.BeginInvoke((Action)(() =>
                {
                    try
                    {
                        if (_swApp != null)
                        {
                            _swApp.ActiveDocChangeNotify -= OnActiveDocChangeNotify;
                        }
                        UnhookDoc();
                    }
                    catch (Exception ex)
                    {
                        Log("DisconnectFromSW event cleanup error: " + ex.Message);
                    }

                    if (_taskPane != null)
                    {
                        _taskPane.Dispose();
                        _taskPane = null;
                    }
                }));
                _marshalControl?.Dispose();
                _marshalControl = null;
            }
            catch { }
            return true;
        }

        // Target for menu item click callback
        public void RunValidationMenu()
        {
            _marshalControl.BeginInvoke((Action)(async () => await RunValidation()));
        }

        public async Task RunValidation()
        {
            try
            {
                Log("RunValidation called");
                var startTime = DateTime.Now;
                var doc = _swApp.IActiveDoc2 as ModelDoc2;
                if (doc == null)
                {
                    Log("RunValidation: no active document");
                    _swApp.SendMsgToUser("No active document. Open a part file first.");
                    return;
                }

                _taskPane.SetProgress("extracting");

                PartMetadata part;
                try
                {
                    part = BuildPartMetadata(doc);
                    _lastPartMetadata = part;
                }
                catch (Exception ex)
                {
                    Log(string.Format("RunValidation: BuildPartMetadata error: {0}: {1}", ex.GetType().Name, ex.Message));
                    _swApp.SendMsgToUser("EUREKA read error: " + ex.Message);
                    _taskPane.SetProgress("error");
                    return;
                }

                _taskPane.SetProgress("rules");

                ValidationResult result = null;
                string errorMsg = null;
                try
                {
                    result = await _client.ValidatePart(part);
                    Log(string.Format("ValidatePart OK: score={0}, violations={1}", 
                        result?.overall_manufacturability_score, result?.violations?.Count));
                }
                catch (Exception ex)
                {
                    errorMsg = ex.Message;
                    Log(string.Format("ValidatePart FAILED: {0}: {1}", ex.GetType().Name, ex.Message));
                }

                _marshalControl.BeginInvoke((Action)(() =>
                {
                    try
                    {
                        if (errorMsg != null)
                        {
                            _taskPane.SetProgress("error");
                            _swApp.SendMsgToUser("Validation request failed. Ensure server is running on http://localhost:8001");
                            return;
                        }

                        _taskPane.SetProgress("gnn");

                        if (result == null)
                        {
                            _taskPane.SetProgress("error");
                            _swApp.SendMsgToUser("Validation result was empty. Check server logs.");
                            return;
                        }

                        _lastResult = result;

                        // Apply overlay via new OverlayRenderer
                        _renderer.ApplyOverlay(result.Violations, doc, _faceRegistry);
                        _taskPane.UpdateResults(result);

                        var elapsed = DateTime.Now - startTime;
                        _taskPane.SetValidationTime(elapsed.TotalSeconds);
                    }
                    catch (Exception innerEx)
                    {
                        Log("RunValidation UI callback error: " + innerEx.Message);
                        _taskPane.SetProgress("error");
                    }
                }));
            }
            catch (Exception ex)
            {
                Log("RunValidation error: " + ex.Message);
                _taskPane.SetProgress("error");
            }
        }

        public async Task SubmitFeedback(bool isAccurate)
        {
            try
            {
                Log(string.Format("SubmitFeedback called, isAccurate={0}", isAccurate));
                if (_lastResult == null)
                {
                    Log("SubmitFeedback: no active validation result");
                    return;
                }

                // Determine GNN predicted label and score
                double gnnScore = _lastResult.gnn_risk_score;
                int predictedLabel = gnnScore >= 0.5 ? 1 : 0;
                int engineerLabel = isAccurate ? predictedLabel : (1 - predictedLabel);

                string ymlPath = null;
                if (_lastResult.risk_summary != null && _lastResult.risk_summary.TryGetValue("yml_path", out object ymlObj))
                {
                    ymlPath = ymlObj?.ToString();
                }

                string partId = _lastResult.part_id;
                if (string.IsNullOrEmpty(partId))
                {
                    partId = _lastPartMetadata?.filename ?? "unknown_part";
                }

                var payload = new
                {
                    part_id = partId,
                    yml_path = ymlPath,
                    predicted_label = predictedLabel,
                    predicted_score = gnnScore,
                    engineer_label = engineerLabel
                };

                string responseJson = await _client.SubmitFeedback(payload);
                Log("SubmitFeedback successful response: " + responseJson);

                string message = "Feedback logged successfully!";
                try
                {
                    var resObj = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, object>>(responseJson);
                    if (resObj != null && resObj.TryGetValue("stats", out object statsObj) && statsObj != null)
                    {
                        var stats = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(statsObj.ToString());
                        if (stats != null && stats.TryGetValue("corrections_pending", out int pending))
                        {
                            message = string.Format("Feedback logged. {0}/30 corrections until next fine-tune.", pending);
                        }
                    }
                }
                catch (Exception jsonEx)
                {
                    Log("SubmitFeedback JSON parse warning: " + jsonEx.Message);
                }

                _marshalControl.BeginInvoke((Action)(() =>
                {
                    _taskPane.ShowFeedbackRecorded(message);
                }));
            }
            catch (Exception ex)
            {
                Log("SubmitFeedback error: " + ex.Message);
                _marshalControl.BeginInvoke((Action)(() =>
                {
                    _taskPane.ShowFeedbackError("Failed to submit feedback: " + ex.Message);
                }));
            }
        }

        public async Task ExportPDF(ValidationResult result)
        {
            try
            {
                Log("ExportPDF called");
                
                var doc = _swApp.IActiveDoc2 as ModelDoc2;
                if (doc != null)
                {
                    string tempPng = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "solidworks_view.png");
                    try
                    {
                        int errors = 0;
                        int warnings = 0;
                        bool success = doc.SaveAs4(tempPng, (int)swSaveAsVersion_e.swSaveAsCurrentVersion, (int)swSaveAsOptions_e.swSaveAsOptions_Silent, ref errors, ref warnings);
                        if (success && System.IO.File.Exists(tempPng))
                        {
                            byte[] bytes = System.IO.File.ReadAllBytes(tempPng);
                            result.screenshot_png_base64 = Convert.ToBase64String(bytes);
                            Log("Screenshot captured successfully, base64 length: " + result.screenshot_png_base64.Length);
                            try { System.IO.File.Delete(tempPng); } catch { }
                        }
                        else
                        {
                            Log(string.Format("SaveAs3 PNG failed, errors={0}, warnings={1}", errors, warnings));
                        }
                    }
                    catch (Exception ex)
                    {
                        Log("Screenshot capture failed: " + ex.Message);
                    }
                }

                using (var sfd = new SaveFileDialog())
                {
                    sfd.Filter = "PDF Files (*.pdf)|*.pdf|HTML Files (*.html)|*.html|Markdown Files (*.md)|*.md";
                    sfd.FileName = string.Format("EUREKA_DFM_Report_{0}", result.part_id);
                    if (sfd.ShowDialog() == DialogResult.OK)
                    {
                        if (sfd.FileName.EndsWith(".pdf"))
                        {
                            var faceSnapshots = new Dictionary<string, string>();

                            if (result.violations != null)
                            {
                                foreach (var violation in result.violations)
                                {
                                    if (violation.face_ids == null || violation.face_ids.Count == 0)
                                        continue;

                                    if (violation.severity == "INFO")
                                        continue;

                                    foreach (var faceId in violation.face_ids)
                                    {
                                        if (faceSnapshots.ContainsKey(faceId))
                                            continue;

                                        if (_renderer.FaceCache.TryGetValue(faceId, out IFace2 face))
                                        {
                                            string snapshot = _renderer.CaptureViolationSnapshot(face, faceId);
                                            if (snapshot != null)
                                                faceSnapshots[faceId] = snapshot;
                                        }
                                    }
                                }
                            }

                            if (doc != null)
                            {
                                doc.ViewZoomtofit2();
                                doc.GraphicsRedraw2();
                            }

                            var reportPayload = new {
                                result = result,
                                face_snapshots = faceSnapshots,
                                faces_geometry = _lastPartMetadata != null ? _lastPartMetadata.faces : new List<FaceGeometry>()
                            };

                            var pdfBytes = await _client.GeneratePdfReport(reportPayload);
                            System.IO.File.WriteAllBytes(sfd.FileName, pdfBytes);
                        }
                        else
                        {
                            var reportContent = await _client.GenerateReport(result);
                            if (sfd.FileName.EndsWith(".html"))
                            {
                                string html = "<html><head><style>body { font-family: 'Segoe UI', sans-serif; margin: 40px; color: #333; } h2 { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 5px; } ul { line-height: 1.6; } li { margin-bottom: 10px; }</style></head><body>" 
                                              + reportContent.Replace("\n", "<br/>").Replace("## ", "<h2>").Replace("**", "<b>").Replace("- [ ] ", "<li>[ ] ").Replace("- ", "<li>") 
                                              + "</body></html>";
                                System.IO.File.WriteAllText(sfd.FileName, html);
                            }
                            else
                            {
                                System.IO.File.WriteAllText(sfd.FileName, reportContent);
                            }
                        }
                        _swApp.SendMsgToUser("Report exported successfully!");
                    }
                }
            }
            catch (Exception ex)
            {
                Log("ExportPDF error: " + ex.Message);
                _swApp.SendMsgToUser("Export failed: " + ex.Message);
            }
        }

        public List<string> GetSelectedFaceIds()
        {
            var list = new List<string>();
            try
            {
                var doc = _swApp.IActiveDoc2 as ModelDoc2;
                if (doc != null)
                {
                    var selMgr = doc.ISelectionManager as SelectionMgr;
                    if (selMgr != null)
                    {
                        int count = selMgr.GetSelectedObjectCount2(-1);
                        for (int i = 1; i <= count; i++)
                        {
                            int type = selMgr.GetSelectedObjectType3(i, -1);
                            if (type == (int)swSelectType_e.swSelFACES)
                            {
                                var face = selMgr.GetSelectedObject6(i, -1) as Face2;
                                if (face != null)
                                {
                                    list.Add(face.GetFaceId().ToString());
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Log("GetSelectedFaceIds error: " + ex.Message);
            }
            return list;
        }

        private PartMetadata BuildPartMetadata(ModelDoc2 doc)
        {
            var part = doc as PartDoc;
            if (part == null)
                throw new InvalidOperationException("Only part documents are supported");

            _faceRegistry.Clear();
            _renderer.InvalidateCache(); // Invalidate overlay renderer cache on new validation run

            string path = doc.GetPathName();
            string name = System.IO.Path.GetFileName(path);

            var box = (double[])part.GetPartBox(false);
            if (box == null || box.Length < 6)
                throw new InvalidOperationException("Could not retrieve part bounding box");
            double bx = Math.Abs(box[3] - box[0]) * 1000;
            double by = Math.Abs(box[4] - box[1]) * 1000;
            double bz = Math.Abs(box[5] - box[2]) * 1000;

            var massProp = (double[])doc.GetMassProperties();
            double volume = massProp != null && massProp.Length > 0 ? massProp[0] : 0;
            double surfaceArea = massProp != null && massProp.Length > 2 ? massProp[2] : 0;
            double partCOMx = massProp != null && massProp.Length >= 4 ? massProp[1] : 0;
            double partCOMy = massProp != null && massProp.Length >= 4 ? massProp[2] : 0;
            double partCOMz = massProp != null && massProp.Length >= 4 ? massProp[3] : 0;

            var faces = new List<FaceGeometry>();
            object[] bodies = (object[])part.GetBodies2((int)swBodyType_e.swSolidBody, false);
            if (bodies != null)
            {
                int faceIndex = 1;
                foreach (var bodyObj in bodies)
                {
                    var body = bodyObj as Body2;
                    if (body == null) continue;
                    object[] faceEntities = (object[])body.GetFaces();
                    if (faceEntities == null) continue;

                    foreach (var fe in faceEntities)
                    {
                        var face = fe as Face2;
                        if (face == null) continue;

                        string faceIdStr = faceIndex.ToString();
                        _faceRegistry[faceIdStr] = face;
                        faceIndex++;

                        var surfType = GetSurfaceType(face);
                        var area = face.GetArea() * 1_000_000;

                        double[] normal = (double[])face.Normal;
                        double nx = normal != null && normal.Length >= 3 ? normal[0] : 0;
                        double ny = normal != null && normal.Length >= 3 ? normal[1] : 0;
                        double nz = normal != null && normal.Length >= 3 ? normal[2] : 0;

                        double thickness = Math.Sqrt(area) * 0.08;
                        if (thickness < 0.3) thickness = 0.3;
                        if (thickness > 8.0) thickness = 8.0;

                        double width = Math.Sqrt(area);
                        double depth = 0;
                        double curvMin = 0;
                        double curvMax = 0;
                        double radius = EstimateRadius(face);

                        if (surfType == "Cylinder" && radius > 0.001)
                        {
                            depth = radius * 2.0;
                            curvMin = 1.0 / radius;
                            curvMax = 1.0 / radius;
                        }
                        else if (surfType == "Plane")
                        {
                            depth = thickness * 0.5;
                        }

                        var cp = (double[])face.GetClosestPointOn(partCOMx, partCOMy, partCOMz);
                        double cx = cp != null ? cp[0] * 1000 : 0;
                        double cy = cp != null ? cp[1] * 1000 : 0;
                        double cz = cp != null ? cp[2] * 1000 : 0;

                        faces.Add(new FaceGeometry
                        {
                            face_id = faceIdStr,
                            face_type = surfType,
                            area_mm2 = area,
                            thickness_mm = thickness,
                            draft_angle_deg = EstimateDraftAngle(nx, ny, nz),
                            radius_mm = radius,
                            depth_mm = depth,
                            width_mm = width,
                            sw_feature_name = string.Format("Face_{0}", faceIdStr),
                            sw_feature_type = surfType,
                            curvature_min = curvMin,
                            curvature_max = curvMax,
                            centroid_x = cx,
                            centroid_y = cy,
                            centroid_z = cz,
                            normal_x = nx,
                            normal_y = ny,
                            normal_z = nz
                        });
                    }
                }
            }

            if (faces.Count > 500)
            {
                throw new InvalidOperationException("Part has more than 500 faces (Count: " + faces.Count + "). Please select a simpler component.");
            }

            // Post-process: set parent_wall_thickness_mm to median face thickness
            if (faces.Count > 0)
            {
                var sorted = faces.Select(f => f.thickness_mm ?? 0).OrderBy(v => v).ToList();
                double median = sorted[sorted.Count / 2];
                foreach (var f in faces)
                    f.parent_wall_thickness_mm = median;
            }

            Log(string.Format("BuildPartMetadata: {0} faces, bbox {1:F1}x{2:F1}x{3:F1}mm", faces.Count, bx, by, bz));

            var partMeta = new PartMetadata
            {
                filename = name,
                solidworks_part_number = System.IO.Path.GetFileNameWithoutExtension(path),
                bounding_box_mm = string.Format("{0:F1} x {1:F1} x {2:F1}", bx, by, bz),
                face_count = faces.Count,
                volume_mm3 = volume * 1_000_000_000,
                surface_area_mm2 = surfaceArea * 1_000_000,
                faces = faces
            };

            partMeta.process = _taskPane.GetSelectedProcess();
            partMeta.material = _taskPane.GetSelectedMaterial();
            partMeta.nominal_wall_mm = _taskPane.GetNominalWall();
            partMeta.classification = _taskPane.GetClassification();
            partMeta.pull_direction = _taskPane.GetPullDirection();
            partMeta.class_a_face_ids = GetSelectedFaceIds();

            return partMeta;
        }

        private string GetSurfaceType(Face2 face)
        {
            var surf = face.GetSurface() as Surface;
            if (surf == null) return "Plane";

            if (surf.IsPlane()) return "Plane";
            if (surf.IsCylinder()) return "Cylinder";
            if (surf.IsSphere()) return "Sphere";
            if (surf.IsCone()) return "Cone";
            if (surf.IsTorus()) return "Torus";
            return "BSpline";
        }

        private double EstimateDraftAngle(double nx, double ny, double nz)
        {
            double dot = Math.Abs(nz);
            return Math.Asin(Math.Min(dot, 1.0)) * 180.0 / Math.PI;
        }

        private double EstimateRadius(Face2 face)
        {
            var surf = face.GetSurface() as Surface;
            if (surf == null) return 1.0;

            if (surf.IsCylinder())
            {
                var cylParams = surf.CylinderParams as double[];
                if (cylParams != null && cylParams.Length >= 7)
                    return Math.Abs(cylParams[6]) * 1000;
            }

            return 1.0;
        }

        private async Task CheckAndStartBackend()
        {
            Log("Checking backend health...");
            bool isAlive = false;
            try
            {
                string health = await _client.GetHealth();
                Log("Backend is already running: " + health);
                isAlive = true;
            }
            catch (Exception)
            {
                Log("Backend not responding. Attempting to start server...");
            }

            if (!isAlive)
            {
                string dllPath = System.Reflection.Assembly.GetExecutingAssembly().Location;
                string dir = System.IO.Path.GetDirectoryName(dllPath);
                string projectRoot = null;
                string current = dir;
                while (!string.IsNullOrEmpty(current))
                {
                    if (System.IO.File.Exists(System.IO.Path.Combine(current, "start_backend.ps1")))
                    {
                        projectRoot = current;
                        break;
                    }
                    current = System.IO.Path.GetDirectoryName(current);
                }

                if (projectRoot != null)
                {
                    try
                    {
                        Log("Launching start_backend.ps1 at " + projectRoot);
                        var psi = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "powershell.exe",
                            Arguments = "-ExecutionPolicy Bypass -File start_backend.ps1",
                            WorkingDirectory = projectRoot,
                            CreateNoWindow = true,
                            UseShellExecute = false
                        };
                        System.Diagnostics.Process.Start(psi);
                        
                        for (int i = 0; i < 5; i++)
                        {
                            await Task.Delay(2000);
                            try
                            {
                                string health = await _client.GetHealth();
                                Log(string.Format("Backend started successfully on attempt {0}: {1}", i + 1, health));
                                isAlive = true;
                                break;
                            }
                            catch
                            {
                                Log(string.Format("Backend not ready yet on attempt {0}...", i + 1));
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Log("Failed to start backend process: " + ex.Message);
                    }
                }
                else
                {
                    Log("Could not find start_backend.ps1 in parent directories of " + dir);
                }
            }

            if (_taskPane != null)
            {
                _taskPane.SetBackendConnected(isAlive);
            }
        }

        private string CreateIcon()
        {
            var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "eureka_icon.bmp");
            try
            {
                using (var bmp = new System.Drawing.Bitmap(16, 16))
                using (var g = System.Drawing.Graphics.FromImage(bmp))
                {
                    g.Clear(System.Drawing.Color.FromArgb(74, 128, 224));
                    using (var b = new System.Drawing.SolidBrush(System.Drawing.Color.White))
                        g.FillRectangle(b, 2, 2, 12, 12);
                    using (var f = new System.Drawing.Font("Segoe UI", 8, System.Drawing.FontStyle.Bold))
                    using (var b2 = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(74, 128, 224)))
                        g.DrawString("E", f, b2, 3, 2);
                    bmp.Save(path, System.Drawing.Imaging.ImageFormat.Bmp);
                }
                return path;
            }
            catch { return null; }
        }
    }
}
