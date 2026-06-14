using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace EurekaAddin
{
    public class CatiaConnector
    {
        private object _catiaApp;
        private TaskPane _taskPane;
        private RestClient _client;
        private Dictionary<string, object> _faceRegistry = new Dictionary<string, object>();
        private Dictionary<string, double[]> _faceNormals = new Dictionary<string, double[]>();
        private ValidationResult _lastResult;
        private object _activeAnnotation;

        private void Log(string msg)
        {
            try
            {
                string logPath = Path.Combine(Path.GetTempPath(), "eureka_catia.log");
                File.AppendAllText(logPath, string.Format("[{0:HH:mm:ss.fff}] {1}\r\n", DateTime.Now, msg));
            }
            catch { }
        }

        [STAThread]
        public static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            
            var connector = new CatiaConnector();
            connector.Run();
        }

        public void Run()
        {
            _client = new RestClient();
            
            // Show main WinForms Form containing TaskPane
            Form form = new Form
            {
                Text = "Eureka DFM 3.0 — CATIA Connector",
                Width = 330,
                Height = 850,
                StartPosition = FormStartPosition.CenterScreen,
                FormBorderStyle = FormBorderStyle.FixedSingle
            };
            
            _taskPane = new TaskPane();
            _taskPane.Dock = DockStyle.Fill;
            form.Controls.Add(_taskPane);
            
            // Wire TaskPane event hooks
            _taskPane.ValidateClicked += (sender, e) => {
                form.BeginInvoke((Action)(async () => await RunValidation()));
            };
            
            _taskPane.ClearClicked += (sender, e) => {
                try {
                    Log("ClearClicked handler");
                    ClearHighlight();
                    _taskPane.ClearResults();
                } catch (Exception ex) { Log("ClearClicked error: " + ex.Message); }
            };
            
            _taskPane.FaceSelected += (sender, faceId) => {
                try {
                    Log("FaceSelected: faceId=" + faceId);
                    if (string.IsNullOrEmpty(faceId)) {
                        ClearHighlight();
                        return;
                    }
                    Violation v = _lastResult?.Violations?.FirstOrDefault(x => x.FaceId == faceId);
                    if (v != null) HighlightViolation(v);
                    else Log("FaceSelected: no violation found for face " + faceId);
                } catch (Exception ex) { Log("FaceSelected error: " + ex.ToString()); }
            };
            
            _taskPane.ExportReportClicked += (sender, e) => {
                if (_lastResult != null) {
                    form.BeginInvoke((Action)(async () => await ExportPDF(_lastResult)));
                }
            };
            
            // Async health check and start backend
            System.Threading.Tasks.Task.Run(async () => {
                await CheckAndStartBackend();
            });
            
            Application.Run(form);
        }

        private object GetActiveSelection()
        {
            if (_catiaApp == null) return null;
            try
            {
                object doc = _catiaApp.GetType().InvokeMember("ActiveDocument",
                    System.Reflection.BindingFlags.GetProperty, null, _catiaApp, null);
                return doc.GetType().InvokeMember("Selection",
                    System.Reflection.BindingFlags.GetProperty, null, doc, null);
            }
            catch
            {
                return null;
            }
        }

        private void ApplyOverlay(ValidationResult result)
        {
            Log("ApplyOverlay starting");
            object selection = GetActiveSelection();
            if (selection == null) { Log("ApplyOverlay: selection object is null"); return; }

            try
            {
                // Group face IDs by severity
                var faceSeverities = new Dictionary<string, string>();
                if (result.violations != null)
                {
                    foreach (var v in result.violations)
                    {
                        if (v.face_ids == null) continue;
                        string sev = (v.severity ?? "").ToUpper();
                        string rule = (v.rule_id ?? "").ToUpper();
                        if (rule.StartsWith("GNN-RISK")) sev = "WARNING";
                        else if (rule.StartsWith("GNN-WATCH") || rule.StartsWith("GNN-ANOMALY")) sev = "INFO";

                        foreach (var fid in v.face_ids)
                        {
                            int currentPriority = 0;
                            if (faceSeverities.TryGetValue(fid, out string curSev))
                            {
                                if (curSev == "CRITICAL") currentPriority = 3;
                                else if (curSev == "WARNING") currentPriority = 2;
                                else if (curSev == "INFO") currentPriority = 1;
                            }

                            int newPriority = 0;
                            if (sev == "CRITICAL") newPriority = 3;
                            else if (sev == "WARNING") newPriority = 2;
                            else if (sev == "INFO") newPriority = 1;

                            if (newPriority > currentPriority)
                            {
                                faceSeverities[fid] = sev;
                            }
                        }
                    }
                }
                Log("ApplyOverlay: face severities grouped. Count = " + faceSeverities.Count);

                // Group faces by RGB components
                var redGroup = new List<object>();
                var orangeGroup = new List<object>();
                var yellowGroup = new List<object>();
                var greenGroup = new List<object>();

                foreach (var kvp in _faceRegistry)
                {
                    string fid = kvp.Key;
                    object face = kvp.Value;

                    if (faceSeverities.TryGetValue(fid, out string sev))
                    {
                        if (sev == "CRITICAL") redGroup.Add(face);
                        else if (sev == "WARNING") orangeGroup.Add(face);
                        else yellowGroup.Add(face);
                    }
                    else
                    {
                        greenGroup.Add(face);
                    }
                }
                Log(string.Format("ApplyOverlay grouping: red={0}, orange={1}, yellow={2}, green={3}", redGroup.Count, orangeGroup.Count, yellowGroup.Count, greenGroup.Count));

                // Apply colors in groups to minimize COM calls
                ApplyGroupColor(selection, redGroup, 224, 82, 82);
                ApplyGroupColor(selection, orangeGroup, 224, 154, 58);
                ApplyGroupColor(selection, yellowGroup, 224, 200, 58);
                ApplyGroupColor(selection, greenGroup, 82, 196, 122);

                try
                {
                    object activeDoc = _catiaApp.GetType().InvokeMember("ActiveDocument", System.Reflection.BindingFlags.GetProperty, null, _catiaApp, null);
                    object part = activeDoc.GetType().InvokeMember("Part", System.Reflection.BindingFlags.GetProperty, null, activeDoc, null);
                    part.GetType().InvokeMember("Update", System.Reflection.BindingFlags.InvokeMethod, null, part, null);
                    Log("ApplyOverlay: Part update complete.");
                }
                catch (Exception upEx) { Log("ApplyOverlay part update error: " + upEx.Message); }
            }
            catch (Exception ex) { Log("ApplyOverlay error: " + ex.ToString()); }
        }

        private void ApplyGroupColor(object selection, List<object> faces, int r, int g, int b)
        {
            if (faces.Count == 0) return;
            try
            {
                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                foreach (var face in faces)
                {
                    selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { face });
                }
                object visProps = selection.GetType().InvokeMember("VisProperties", System.Reflection.BindingFlags.GetProperty, null, selection, null);
                visProps.GetType().InvokeMember("SetRealColor", System.Reflection.BindingFlags.InvokeMethod, null, visProps, new object[] { r, g, b, 1 });
                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                Log(string.Format("ApplyGroupColor: colored {0} faces to ({1},{2},{3})", faces.Count, r, g, b));
            }
            catch (Exception ex) { Log(string.Format("ApplyGroupColor error for {0} faces: {1}", faces.Count, ex.Message)); }
        }

        private void ClearOverlay()
        {
            Log("ClearOverlay starting.");
            object selection = GetActiveSelection();
            if (selection == null) { Log("ClearOverlay: no selection object"); return; }
            try
            {
                ClearAnnotation(selection);

                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                foreach (var face in _faceRegistry.Values)
                {
                    selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { face });
                }
                object visProps = selection.GetType().InvokeMember("VisProperties", System.Reflection.BindingFlags.GetProperty, null, selection, null);
                visProps.GetType().InvokeMember("ResetRealColor", System.Reflection.BindingFlags.InvokeMethod, null, visProps, null);
                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);

                UpdatePart();
                Log("ClearOverlay completed.");
            }
            catch (Exception ex) { Log("ClearOverlay error: " + ex.Message); }
        }

        /// <summary>Clears annotations and re-applies the color overlay (like SolidWorks ClearHighlight)</summary>
        private void ClearHighlight()
        {
            Log("ClearHighlight starting.");
            try
            {
                object selection = GetActiveSelection();
                if (selection != null) ClearAnnotation(selection);

                // Re-apply overlay to restore colors (matches SolidWorks behavior)
                if (_lastResult != null)
                {
                    ApplyOverlay(_lastResult);
                }
                Log("ClearHighlight completed.");
            }
            catch (Exception ex) { Log("ClearHighlight error: " + ex.Message); }
        }

        private void ClearAnnotation(object selection)
        {
            if (_activeAnnotation != null)
            {
                try
                {
                    selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                    selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { _activeAnnotation });
                    selection.GetType().InvokeMember("Delete", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                    Log("Deleted previous annotation.");
                }
                catch (Exception ex) { Log("ClearAnnotation delete error: " + ex.Message); }
                _activeAnnotation = null;
            }
        }

        private void UpdatePart()
        {
            try
            {
                object activeDoc = _catiaApp.GetType().InvokeMember("ActiveDocument", System.Reflection.BindingFlags.GetProperty, null, _catiaApp, null);
                object part = activeDoc.GetType().InvokeMember("Part", System.Reflection.BindingFlags.GetProperty, null, activeDoc, null);
                part.GetType().InvokeMember("Update", System.Reflection.BindingFlags.InvokeMethod, null, part, null);
            }
            catch (Exception ex) { Log("UpdatePart error: " + ex.Message); }
        }

        private string GetSeverityForFace(string faceId)
        {
            if (_lastResult == null || _lastResult.Violations == null) return null;
            string highest = null;
            int highPri = 0;
            foreach (var v in _lastResult.Violations)
            {
                if (v.face_ids == null || !v.face_ids.Contains(faceId)) continue;
                string sev = (v.severity ?? "").ToUpper();
                string rule = (v.rule_id ?? "").ToUpper();
                if (rule.StartsWith("GNN-RISK")) sev = "WARNING";
                else if (rule.StartsWith("GNN-WATCH") || rule.StartsWith("GNN-ANOMALY")) sev = "INFO";
                int pri = sev == "CRITICAL" ? 3 : sev == "WARNING" ? 2 : sev == "INFO" ? 1 : 0;
                if (pri > highPri) { highPri = pri; highest = sev; }
            }
            return highest;
        }

        private void GetSeverityColor(string severity, out int r, out int g, out int b)
        {
            switch ((severity ?? "").ToUpper())
            {
                case "CRITICAL": r = 224; g = 82;  b = 82;  return;
                case "WARNING":  r = 224; g = 154; b = 58;  return;
                case "INFO":     r = 74;  g = 158; b = 224; return;
                default:         r = 82;  g = 196; b = 122; return;
            }
        }

        /// <summary>Zooms to face, orients camera normal-to, re-applies severity color</summary>
        private void ZoomToFace(string faceId)
        {
            Log("ZoomToFace: " + faceId);
            if (!_faceRegistry.TryGetValue(faceId, out object face))
            {
                Log("ZoomToFace: face not found in registry: " + faceId);
                return;
            }

            object selection = GetActiveSelection();
            if (selection == null) return;

            try
            {
                // Select the face
                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { face });

                // Reframe (zoom) to the selected face
                object win = _catiaApp.GetType().InvokeMember("ActiveWindow", System.Reflection.BindingFlags.GetProperty, null, _catiaApp, null);
                object viewer = win.GetType().InvokeMember("ActiveViewer", System.Reflection.BindingFlags.GetProperty, null, win, null);
                viewer.GetType().InvokeMember("Reframe", System.Reflection.BindingFlags.InvokeMethod, null, viewer, null);
                Log("ZoomToFace: reframed.");

                // Orient camera normal-to the face (like SolidWorks ShowNamedView2 "*Normal To")
                try
                {
                    if (_faceNormals.TryGetValue(faceId, out double[] normal) && normal != null && normal.Length >= 3)
                    {
                        double nx = normal[0], ny = normal[1], nz = normal[2];
                        double nlen = Math.Sqrt(nx*nx + ny*ny + nz*nz);
                        if (nlen > 0.001)
                        {
                            nx /= nlen; ny /= nlen; nz /= nlen;
                            object viewpoint = viewer.GetType().InvokeMember("Viewpoint3D", System.Reflection.BindingFlags.GetProperty, null, viewer, null);

                            // Get current eye position (origin)
                            object origin = viewpoint.GetType().InvokeMember("Origin", System.Reflection.BindingFlags.GetProperty, null, viewpoint, null);
                            double ox = (double)origin.GetType().InvokeMember("X", System.Reflection.BindingFlags.GetProperty, null, origin, null);
                            double oy = (double)origin.GetType().InvokeMember("Y", System.Reflection.BindingFlags.GetProperty, null, origin, null);
                            double oz = (double)origin.GetType().InvokeMember("Z", System.Reflection.BindingFlags.GetProperty, null, origin, null);

                            // Set sight direction to face normal (look along the negative normal toward the face)
                            object sightDir = viewpoint.GetType().InvokeMember("SightDirection", System.Reflection.BindingFlags.GetProperty, null, viewpoint, null);
                            sightDir.GetType().InvokeMember("X", System.Reflection.BindingFlags.SetProperty, null, sightDir, new object[] { -nx });
                            sightDir.GetType().InvokeMember("Y", System.Reflection.BindingFlags.SetProperty, null, sightDir, new object[] { -ny });
                            sightDir.GetType().InvokeMember("Z", System.Reflection.BindingFlags.SetProperty, null, sightDir, new object[] { -nz });
                            viewpoint.GetType().InvokeMember("SightDirection", System.Reflection.BindingFlags.SetProperty, null, viewpoint, new object[] { sightDir });

                            // Compute up direction: perpendicular to sight direction
                            double ux, uy, uz;
                            if (Math.Abs(nz) < 0.9) { ux = 0; uy = 0; uz = 1; }
                            else { ux = 0; uy = 1; uz = 0; }
                            // Cross product: up = right x sight = (arbitrary x (-normal))
                            double rx = uy * (-nz) - uz * (-ny);
                            double ry = uz * (-nx) - ux * (-nz);
                            double rz = ux * (-ny) - uy * (-nx);
                            double rlen = Math.Sqrt(rx*rx + ry*ry + rz*rz);
                            if (rlen > 0.001) { rx /= rlen; ry /= rlen; rz /= rlen; }
                            // up = sight x right
                            ux = (-ny) * rz - (-nz) * ry;
                            uy = (-nz) * rx - (-nx) * rz;
                            uz = (-nx) * ry - (-ny) * rx;
                            double ulen = Math.Sqrt(ux*ux + uy*uy + uz*uz);
                            if (ulen > 0.001) { ux /= ulen; uy /= ulen; uz /= ulen; }

                            object upDir = viewpoint.GetType().InvokeMember("UpDirection", System.Reflection.BindingFlags.GetProperty, null, viewpoint, null);
                            upDir.GetType().InvokeMember("X", System.Reflection.BindingFlags.SetProperty, null, upDir, new object[] { ux });
                            upDir.GetType().InvokeMember("Y", System.Reflection.BindingFlags.SetProperty, null, upDir, new object[] { uy });
                            upDir.GetType().InvokeMember("Z", System.Reflection.BindingFlags.SetProperty, null, upDir, new object[] { uz });
                            viewpoint.GetType().InvokeMember("UpDirection", System.Reflection.BindingFlags.SetProperty, null, viewpoint, new object[] { upDir });

                            viewer.GetType().InvokeMember("Update", System.Reflection.BindingFlags.InvokeMethod, null, viewer, null);
                            Log("ZoomToFace: oriented camera normal-to face.");
                        }
                    }
                }
                catch (Exception vpEx) { Log("ZoomToFace: viewpoint orient error: " + vpEx.Message); }

                // Re-apply severity color to this face (like SolidWorks does in ZoomToFace)
                string sev = GetSeverityForFace(faceId);
                GetSeverityColor(sev, out int cr, out int cg, out int cb);
                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { face });
                object vp = selection.GetType().InvokeMember("VisProperties", System.Reflection.BindingFlags.GetProperty, null, selection, null);
                vp.GetType().InvokeMember("SetRealColor", System.Reflection.BindingFlags.InvokeMethod, null, vp, new object[] { cr, cg, cb, 1 });
                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                Log(string.Format("ZoomToFace: re-applied color ({0},{1},{2}) for severity={3}", cr, cg, cb, sev));
            }
            catch (Exception ex) { Log("ZoomToFace error: " + ex.Message); }
        }

        /// <summary>Matches SolidWorks HighlightViolation: zoom + dimension note</summary>
        private void HighlightViolation(Violation v)
        {
            Log(string.Format("HighlightViolation: rule={0}, face={1}", v.rule_id, v.FaceId));
            try
            {
                // 1. Zoom to face and rotate
                ZoomToFace(v.FaceId);

                // 2. Show annotation callout note
                ShowDimensionNote(v);
            }
            catch (Exception ex) { Log("HighlightViolation error: " + ex.Message); }
        }

        /// <summary>Matches SolidWorks ShowDimensionNote: creates 3D annotation with dimension text</summary>
        private void ShowDimensionNote(Violation v)
        {
            Log("ShowDimensionNote for face: " + v.FaceId);
            object selection = GetActiveSelection();
            if (selection == null) return;

            try
            {
                // Clear previous annotation
                ClearAnnotation(selection);

                if (!_faceRegistry.TryGetValue(v.FaceId, out object face))
                {
                    Log("ShowDimensionNote: face not in registry: " + v.FaceId);
                    return;
                }

                // Select the face for annotation attachment
                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { face });

                // Build note text
                string noteText;
                bool isGnn = (v.rule_id ?? "").StartsWith("GNN-");
                
                if (isGnn)
                {
                    // GNN violations: show risk percentage (not 0.00mm → ≥0.00mm which is useless)
                    string riskPercent = "";
                    string desc = v.description ?? "";
                    int riskIdx = desc.IndexOf("risk ");
                    if (riskIdx >= 0)
                    {
                        int pctIdx = desc.IndexOf("%", riskIdx);
                        if (pctIdx > riskIdx)
                            riskPercent = desc.Substring(riskIdx + 5, pctIdx - riskIdx - 5 + 1);
                    }
                    if (string.IsNullOrEmpty(riskPercent)) riskPercent = "N/A";
                    noteText = string.Format("{0}\nNeural Risk: {1}\nReview wall & draft geometry", v.rule_id, riskPercent);
                }
                else
                {
                    // Rules violations: exact same format as SolidWorks OverlayRenderer.ShowDimensionNote but using ASCII for CATIA COM safety
                    double delta = v.RequiredValue - v.MeasuredValue;
                    string sign = delta >= 0 ? "+" : "";
                    string relation = ">=";
                    if (v.MeasuredValue > v.RequiredValue) relation = "<=";

                    string unitStr = (v.Unit == "°" || v.Unit == "deg") ? "deg" : (v.Unit ?? "mm");
                    noteText = string.Format("{0}\n{1:F2}{2} -> {3}{4:F2}{5}\nFix: {6}{7:F2}{8}",
                        v.Id ?? v.rule_id,
                        v.MeasuredValue, unitStr,
                        relation, v.RequiredValue, unitStr,
                        sign, delta, unitStr);
                }

                Log("ShowDimensionNote text: " + noteText.Replace("\n", " | "));

                // Create 3D annotation via CATIA AnnotationSets API
                try
                {
                    object activeDoc = _catiaApp.GetType().InvokeMember("ActiveDocument", System.Reflection.BindingFlags.GetProperty, null, _catiaApp, null);
                    object part = activeDoc.GetType().InvokeMember("Part", System.Reflection.BindingFlags.GetProperty, null, activeDoc, null);

                    object annSets = part.GetType().InvokeMember("AnnotationSets", System.Reflection.BindingFlags.GetProperty, null, part, null);
                    object annSet = null;
                    int setsCount = (int)annSets.GetType().InvokeMember("Count", System.Reflection.BindingFlags.GetProperty, null, annSets, null);
                    if (setsCount > 0)
                        annSet = annSets.GetType().InvokeMember("Item", System.Reflection.BindingFlags.InvokeMethod, null, annSets, new object[] { 1 });
                    else
                        annSet = annSets.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, annSets, new object[] { "ISO" });

                    object userSurfs = part.GetType().InvokeMember("UserSurfaces", System.Reflection.BindingFlags.GetProperty, null, part, null);
                    object selectedItem = selection.GetType().InvokeMember("Item2", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { 1 });
                    object reference = selectedItem.GetType().InvokeMember("Reference", System.Reflection.BindingFlags.GetProperty, null, selectedItem, null);
                    object userSurf = userSurfs.GetType().InvokeMember("Generate", System.Reflection.BindingFlags.InvokeMethod, null, userSurfs, new object[] { reference });
                    object factory = annSet.GetType().InvokeMember("AnnotationFactory", System.Reflection.BindingFlags.GetProperty, null, annSet, null);
                    object annotation = factory.GetType().InvokeMember("CreateText", System.Reflection.BindingFlags.InvokeMethod, null, factory, new object[] { userSurf });
                    _activeAnnotation = annotation;

                    object textProp = Microsoft.VisualBasic.Interaction.CallByName(annotation, "Text", Microsoft.VisualBasic.CallType.Get);
                    Microsoft.VisualBasic.Interaction.CallByName(textProp, "Text", Microsoft.VisualBasic.CallType.Let, noteText);

                    part.GetType().InvokeMember("Update", System.Reflection.BindingFlags.InvokeMethod, null, part, null);
                    Log("ShowDimensionNote: annotation created successfully.");
                }
                catch (Exception annEx)
                {
                    Log("ShowDimensionNote annotation error: " + annEx.ToString());
                    // Fallback: show the note in a message for now
                    Log("ShowDimensionNote FALLBACK: noteText = " + noteText);
                }

                // Keep face selected for visual feedback
                selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { face });
            }
            catch (Exception ex) { Log("ShowDimensionNote error: " + ex.ToString()); }
        }

        private PartMetadata BuildPartMetadata()
        {
            // Connect to CATIA if not already connected
            if (_catiaApp == null)
            {
                try
                {
                    _catiaApp = Marshal.GetActiveObject("CATIA.Application");
                }
                catch (Exception)
                {
                    throw new InvalidOperationException("Could not connect to CATIA. Ensure CATIA is running with an active part document.");
                }
            }
            
            object catia = _catiaApp;
            object activeDoc = null;
            try
            {
                activeDoc = catia.GetType().InvokeMember("ActiveDocument",
                    System.Reflection.BindingFlags.GetProperty, null, catia, null);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("No active document found in CATIA. Ensure a part is open. Details: " + ex.Message);
            }

            object partDoc = null;
            if (activeDoc != null)
            {
                string docTypeName = Microsoft.VisualBasic.Information.TypeName(activeDoc);
                if (docTypeName == "PartDocument")
                {
                    partDoc = activeDoc;
                }
                else
                {
                    // Fall back to finding the first PartDocument in the session
                    try
                    {
                        object docs = catia.GetType().InvokeMember("Documents",
                            System.Reflection.BindingFlags.GetProperty, null, catia, null);
                        int count = (int)docs.GetType().InvokeMember("Count",
                            System.Reflection.BindingFlags.GetProperty, null, docs, null);
                        for (int j = 1; j <= count; j++)
                        {
                            object doc = docs.GetType().InvokeMember("Item",
                                System.Reflection.BindingFlags.InvokeMethod, null, docs, new object[] { j });
                            if (Microsoft.VisualBasic.Information.TypeName(doc) == "PartDocument")
                            {
                                partDoc = doc;
                                break;
                            }
                        }
                    }
                    catch { }
                }
            }

            if (partDoc == null)
            {
                throw new InvalidOperationException("No open Part Document found. Please ensure a part is open in CATIA.");
            }
            
            object part = partDoc.GetType().InvokeMember("Part",
                System.Reflection.BindingFlags.GetProperty, null, partDoc, null);
            string path = "";
            try { path = (string)partDoc.GetType().InvokeMember("FullName", System.Reflection.BindingFlags.GetProperty, null, partDoc, null); } catch { }
            string name = "CATIAPart";
            try { name = (string)partDoc.GetType().InvokeMember("Name", System.Reflection.BindingFlags.GetProperty, null, partDoc, null); } catch { }
            
            _faceRegistry.Clear();
            _faceNormals.Clear();
            
            // Get SPAWorkbench for measurements
            object spaWorkbench = partDoc.GetType().InvokeMember("GetWorkbench",
                System.Reflection.BindingFlags.InvokeMethod, null, partDoc, new object[] { "SPAWorkbench" });
            
            double volume = 0;
            double surfaceArea = 0;
            double bx = 0, by = 0, bz = 0;
            
            // Collect all faces using selection search
            List<object> catiaFaces = new List<object>();
            List<object> catiaFaceRefs = new List<object>();
            object selection = partDoc.GetType().InvokeMember("Selection",
                System.Reflection.BindingFlags.GetProperty, null, partDoc, null);
            
            selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
            selection.GetType().InvokeMember("Search", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { "Topology.CGMFace,all" });
            int searchCount = (int)selection.GetType().InvokeMember("Count2", System.Reflection.BindingFlags.GetProperty, null, selection, null);
            for (int i = 1; i <= searchCount; i++)
            {
                object element = selection.GetType().InvokeMember("Item2", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { i });
                object face = element.GetType().InvokeMember("Value", System.Reflection.BindingFlags.GetProperty, null, element, null);
                object faceRef = element.GetType().InvokeMember("Reference", System.Reflection.BindingFlags.GetProperty, null, element, null);
                if (face != null && faceRef != null)
                {
                    catiaFaces.Add(face);
                    catiaFaceRefs.Add(faceRef);
                }
            }
            selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
            
            if (catiaFaces.Count > 500)
            {
                throw new InvalidOperationException("Part has more than 500 faces (Count: " + catiaFaces.Count + "). Please select a simpler component.");
            }
            
            // Measure the main solid body for part properties if available
            string partBodyName = null;
            try
            {
                object bodies = part.GetType().InvokeMember("Bodies", System.Reflection.BindingFlags.GetProperty, null, part, null);
                int bodiesCount = (int)bodies.GetType().InvokeMember("Count", System.Reflection.BindingFlags.GetProperty, null, bodies, null);
                if (bodiesCount > 0)
                {
                    object mainBody = bodies.GetType().InvokeMember("Item", System.Reflection.BindingFlags.InvokeMethod, null, bodies, new object[] { 1 });
                    partBodyName = (string)mainBody.GetType().InvokeMember("Name", System.Reflection.BindingFlags.GetProperty, null, mainBody, null);
                    object bodyRef = part.GetType().InvokeMember("CreateReferenceFromObject", System.Reflection.BindingFlags.InvokeMethod, null, part, new object[] { mainBody });
                    object measurable = spaWorkbench.GetType().InvokeMember("GetMeasurable", System.Reflection.BindingFlags.InvokeMethod, null, spaWorkbench, new object[] { bodyRef });
                    volume = (double)measurable.GetType().InvokeMember("Volume", System.Reflection.BindingFlags.GetProperty, null, measurable, null);
                    surfaceArea = (double)measurable.GetType().InvokeMember("Area", System.Reflection.BindingFlags.GetProperty, null, measurable, null);
                }
            }
            catch { }
            
            // Bounding box approximation using face centroids
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;
            double minZ = double.MaxValue, maxZ = double.MinValue;
            
            var faces = new List<FaceGeometry>();
            
            for (int i = 0; i < catiaFaces.Count; i++)
            {
                object face = catiaFaces[i];
                object faceRef = catiaFaceRefs[i];
                string faceIdStr = (i + 1).ToString();
                _faceRegistry[faceIdStr] = face;
                _faceNormals[faceIdStr] = null; // will be populated below
                
                try
                {
                    object measurable = spaWorkbench.GetType().InvokeMember("GetMeasurable",
                        System.Reflection.BindingFlags.InvokeMethod, null, spaWorkbench, new object[] { faceRef });
                    
                    double area = (double)measurable.GetType().InvokeMember("Area",
                        System.Reflection.BindingFlags.GetProperty, null, measurable, null);
                    double area_mm2 = area * 1_000_000.0;
                    
                    object[] cog = new object[3];
                    object[] cogArgs = new object[] { cog };
                    InvokeByRef(measurable, "GetCOG", cogArgs);
                    object[] cogResult = (object[])cogArgs[0];
                    double fcx = Convert.ToDouble(cogResult[0]);
                    double fcy = Convert.ToDouble(cogResult[1]);
                    double fcz = Convert.ToDouble(cogResult[2]);
                    
                    // Track bounding box coordinates
                    if (fcx < minX) minX = fcx; if (fcx > maxX) maxX = fcx;
                    if (fcy < minY) minY = fcy; if (fcy > maxY) maxY = fcy;
                    if (fcz < minZ) minZ = fcz; if (fcz > maxZ) maxZ = fcz;
                    
                    // Get Face type name via Microsoft.VisualBasic.Information.TypeName
                    string comTypeName = Microsoft.VisualBasic.Information.TypeName(face);
                    string surfType = "Plane";
                    if (comTypeName == "PlanarFace") surfType = "Plane";
                    else if (comTypeName == "CylindricalFace") surfType = "Cylinder";
                    else if (comTypeName == "SphericalFace") surfType = "Sphere";
                    else if (comTypeName == "ConicalFace") surfType = "Cone";
                    else if (comTypeName == "ToroidalFace") surfType = "Torus";
                    else surfType = "BSpline";
                    
                    double nx = 0, ny = 0, nz = 1;
                    double radius = 1.0;
                    try
                    {
                        if (surfType == "Cylinder" || surfType == "Sphere")
                        {
                            radius = (double)measurable.GetType().InvokeMember("Radius",
                                System.Reflection.BindingFlags.GetProperty, null, measurable, null);
                        }
                    }
                    catch { }
                    
                    try
                    {
                        if (surfType == "Plane")
                        {
                            object[] plane = new object[9];
                            object[] planeArgs = new object[] { plane };
                            InvokeByRef(measurable, "GetPlane", planeArgs);
                            object[] pData = (object[])planeArgs[0];
                            if (pData != null && pData.Length >= 9)
                            {
                                double ux = Convert.ToDouble(pData[3]);
                                double uy = Convert.ToDouble(pData[4]);
                                double uz = Convert.ToDouble(pData[5]);
                                double vx = Convert.ToDouble(pData[6]);
                                double vy = Convert.ToDouble(pData[7]);
                                double vz = Convert.ToDouble(pData[8]);
                                nx = uy * vz - uz * vy;
                                ny = uz * vx - ux * vz;
                                nz = ux * vy - uy * vx;
                                double nlen = Math.Sqrt(nx*nx + ny*ny + nz*nz);
                                if (nlen > 0.0001) { nx /= nlen; ny /= nlen; nz /= nlen; }
                            }
                        }
                        else if (surfType == "Cylinder" || surfType == "Cone")
                        {
                            object[] origin = new object[3];
                            object[] originArgs = new object[] { origin };
                            InvokeByRef(face, "GetOrigin", originArgs);
                            object[] oData = (object[])originArgs[0];

                            object[] direction = new object[3];
                            object[] directionArgs = new object[] { direction };
                            InvokeByRef(face, "GetDirection", directionArgs);
                            object[] dData = (object[])directionArgs[0];

                            if (oData != null && oData.Length >= 3 && dData != null && dData.Length >= 3)
                            {
                                double ax = Convert.ToDouble(oData[0]);
                                double ay = Convert.ToDouble(oData[1]);
                                double az = Convert.ToDouble(oData[2]);
                                double dx = Convert.ToDouble(dData[0]);
                                double dy = Convert.ToDouble(dData[1]);
                                double dz = Convert.ToDouble(dData[2]);
                                double len = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                                if (len > 0.0001) { dx /= len; dy /= len; dz /= len; }
                                double acx = fcx - ax, acy = fcy - ay, acz = fcz - az;
                                double dot = acx * dx + acy * dy + acz * dz;
                                double px = ax + dot * dx, py = ay + dot * dy, pz = az + dot * dz;
                                nx = fcx - px; ny = fcy - py; nz = fcz - pz;
                                double nlen = Math.Sqrt(nx * nx + ny * ny + nz * nz);
                                if (nlen > 0.0001) { nx /= nlen; ny /= nlen; nz /= nlen; }
                            }
                        }
                        else if (surfType == "Sphere")
                        {
                            object[] center = new object[3];
                            object[] centerArgs = new object[] { center };
                            InvokeByRef(measurable, "GetCenter", centerArgs);
                            object[] cData = (object[])centerArgs[0];
                            if (cData != null && cData.Length >= 3)
                            {
                                double sx = Convert.ToDouble(cData[0]);
                                double sy = Convert.ToDouble(cData[1]);
                                double sz = Convert.ToDouble(cData[2]);
                                nx = fcx - sx; ny = fcy - sy; nz = fcz - sz;
                                double nlen = Math.Sqrt(nx*nx + ny*ny + nz*nz);
                                if (nlen > 0.0001) { nx /= nlen; ny /= nlen; nz /= nlen; }
                            }
                        }
                    }
                    catch (Exception ex) { Log("Error calculating normal for face " + faceIdStr + ": " + ex.Message); }
                    
                    double thickness = Math.Sqrt(area_mm2) * 0.08;
                    if (thickness < 0.3) thickness = 0.3;
                    if (thickness > 8.0) thickness = 8.0;
                    
                    double width = Math.Sqrt(area_mm2);
                    double depth = 0;
                    double curvMin = 0;
                    double curvMax = 0;
                    
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

                    if (!string.IsNullOrEmpty(partBodyName) && (surfType == "Cylinder" || surfType == "Cone" || surfType == "Sphere"))
                    {
                        EnsureOutwardNormal(part, partDoc, partBodyName, fcx, fcy, fcz, ref nx, ref ny, ref nz);
                    }
                    
                    faces.Add(new FaceGeometry
                    {
                        face_id = faceIdStr,
                        face_type = surfType,
                        area_mm2 = area_mm2,
                        thickness_mm = thickness,
                        draft_angle_deg = EstimateDraftAngle(nx, ny, nz),
                        radius_mm = radius,
                        depth_mm = depth,
                        width_mm = width,
                        sw_feature_name = string.Format("Face_{0}", faceIdStr),
                        sw_feature_type = surfType,
                        curvature_min = curvMin,
                        curvature_max = curvMax,
                        centroid_x = fcx,
                        centroid_y = fcy,
                        centroid_z = fcz,
                        normal_x = nx,
                        normal_y = ny,
                        normal_z = nz
                    });
                    
                    // Store face normal for camera orientation in ZoomToFace
                    _faceNormals[faceIdStr] = new double[] { nx, ny, nz };
                }
                catch (Exception faceEx) { Log("BuildPartMetadata face " + faceIdStr + " error: " + faceEx.Message + "\n" + faceEx.StackTrace); }
            }
            
            // Post-process parent wall thickness (median face thickness)
            if (faces.Count > 0)
            {
                var sorted = faces.Select(f => f.thickness_mm ?? 0).OrderBy(v => v).ToList();
                double median = sorted[sorted.Count / 2];
                foreach (var f in faces)
                    f.parent_wall_thickness_mm = median;
            }
            
            // Estimate bounding box dimensions
            if (maxX > minX) bx = maxX - minX;
            if (maxY > minY) by = maxY - minY;
            if (maxZ > minZ) bz = maxZ - minZ;
            
            if (bx < 0.1) bx = 50.0;
            if (by < 0.1) by = 50.0;
            if (bz < 0.1) bz = 10.0;
            
            var partMeta = new PartMetadata
            {
                filename = name,
                solidworks_part_number = System.IO.Path.GetFileNameWithoutExtension(name),
                bounding_box_mm = string.Format("{0:F1} x {1:F1} x {2:F1}", bx, by, bz),
                face_count = faces.Count,
                volume_mm3 = volume * 1_000_000_000.0,
                surface_area_mm2 = surfaceArea * 1_000_000.0,
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

        public List<string> GetSelectedFaceIds()
        {
            var list = new List<string>();
            try
            {
                object selection = GetActiveSelection();
                if (selection == null) return list;
                int count = (int)selection.GetType().InvokeMember("Count2", System.Reflection.BindingFlags.GetProperty, null, selection, null);
                for (int i = 1; i <= count; i++)
                {
                    object element = selection.GetType().InvokeMember("Item2", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { i });
                    object val = element.GetType().InvokeMember("Value", System.Reflection.BindingFlags.GetProperty, null, element, null);
                    if (val != null)
                    {
                        foreach (var kvp in _faceRegistry)
                        {
                            if (kvp.Value == val || kvp.Value.Equals(val))
                            {
                                list.Add(kvp.Key);
                                break;
                            }
                        }
                    }
                }
            }
            catch (Exception ex) { Log("GetSelectedFaceIds error: " + ex.Message); }
            return list;
        }

        private static void InvokeByRef(object target, string methodName, object[] args)
        {
            var type = target.GetType();
            System.Reflection.ParameterModifier[] modifiers = new System.Reflection.ParameterModifier[1];
            modifiers[0] = new System.Reflection.ParameterModifier(args.Length);
            for (int i = 0; i < args.Length; i++)
            {
                modifiers[0][i] = true;
            }
            type.InvokeMember(methodName, System.Reflection.BindingFlags.InvokeMethod, null, target, args, modifiers, null, null);
        }

        private void EnsureOutwardNormal(object part, object partDoc, string partBodyName, double fcx, double fcy, double fcz, ref double nx, ref double ny, ref double nz)
        {
            object tempBody = null;
            object formula = null;
            object distParam = null;
            try
            {
                object hybridBodies = part.GetType().InvokeMember("HybridBodies", System.Reflection.BindingFlags.GetProperty, null, part, null);
                tempBody = hybridBodies.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, hybridBodies, null);
                Microsoft.VisualBasic.Interaction.CallByName(tempBody, "Name", Microsoft.VisualBasic.CallType.Let, "Eureka_Normal_Temp");

                object hybridShapeFactory = part.GetType().InvokeMember("HybridShapeFactory", System.Reflection.BindingFlags.GetProperty, null, part, null);
                double px = fcx + 0.1 * nx;
                double py = fcy + 0.1 * ny;
                double pz = fcz + 0.1 * nz;

                object pt = hybridShapeFactory.GetType().InvokeMember("AddNewPointCoord",
                    System.Reflection.BindingFlags.InvokeMethod, null, hybridShapeFactory, new object[] { px, py, pz });
                tempBody.GetType().InvokeMember("AppendHybridShape", System.Reflection.BindingFlags.InvokeMethod, null, tempBody, new object[] { pt });

                part.GetType().InvokeMember("Update", System.Reflection.BindingFlags.InvokeMethod, null, part, null);

                object parameters = part.GetType().InvokeMember("Parameters", System.Reflection.BindingFlags.GetProperty, null, part, null);
                object relations = part.GetType().InvokeMember("Relations", System.Reflection.BindingFlags.GetProperty, null, part, null);

                distParam = parameters.GetType().InvokeMember("CreateDimension",
                    System.Reflection.BindingFlags.InvokeMethod, null, parameters, new object[] { "", "LENGTH", 0.0 });
                
                string ptName = (string)pt.GetType().InvokeMember("Name", System.Reflection.BindingFlags.GetProperty, null, pt, null);
                string formulaStr = string.Format("distance(`Eureka_Normal_Temp\\{0}`, `{1}`)", ptName, partBodyName);

                formula = relations.GetType().InvokeMember("CreateFormula",
                    System.Reflection.BindingFlags.InvokeMethod, null, relations, new object[] { "Eureka_Normal_Formula", "", distParam, formulaStr });

                part.GetType().InvokeMember("Update", System.Reflection.BindingFlags.InvokeMethod, null, part, null);
                double val = (double)distParam.GetType().InvokeMember("Value", System.Reflection.BindingFlags.GetProperty, null, distParam, null);

                if (val < 0.0001)
                {
                    nx = -nx;
                    ny = -ny;
                    nz = -nz;
                }
            }
            catch (Exception ex)
            {
                Log("EnsureOutwardNormal error: " + ex.Message);
            }
            finally
            {
                try
                {
                    object selection = partDoc.GetType().InvokeMember("Selection", System.Reflection.BindingFlags.GetProperty, null, partDoc, null);
                    selection.GetType().InvokeMember("Clear", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                    
                    if (tempBody != null) selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { tempBody });
                    if (formula != null) selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { formula });
                    if (distParam != null) selection.GetType().InvokeMember("Add", System.Reflection.BindingFlags.InvokeMethod, null, selection, new object[] { distParam });
                    
                    int count = (int)selection.GetType().InvokeMember("Count2", System.Reflection.BindingFlags.GetProperty, null, selection, null);
                    if (count > 0)
                    {
                        selection.GetType().InvokeMember("Delete", System.Reflection.BindingFlags.InvokeMethod, null, selection, null);
                        part.GetType().InvokeMember("Update", System.Reflection.BindingFlags.InvokeMethod, null, part, null);
                    }
                }
                catch (Exception ex)
                {
                    Log("EnsureOutwardNormal cleanup error: " + ex.Message);
                }
            }
        }

        private double EstimateDraftAngle(double nx, double ny, double nz)
        {
            double dot = Math.Abs(nz);
            return Math.Asin(Math.Min(dot, 1.0)) * 180.0 / Math.PI;
        }

        public async System.Threading.Tasks.Task RunValidation()
        {
            try
            {
                Log("RunValidation called");
                var startTime = DateTime.Now;
                _taskPane.SetProgress("extracting");

                PartMetadata part;
                try
                {
                    part = BuildPartMetadata();
                    Log(string.Format("BuildPartMetadata OK: {0} faces, faceRegistry={1}", part.face_count, _faceRegistry.Count));
                }
                catch (Exception ex)
                {
                    MessageBox.Show("EUREKA read error: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    _taskPane.SetProgress("error");
                    return;
                }

                _taskPane.SetProgress("rules");

                ValidationResult result = null;
                string errorMsg = null;
                try
                {
                    result = await _client.ValidatePart(part);
                }
                catch (Exception ex)
                {
                    errorMsg = ex.Message;
                }

                if (errorMsg != null)
                {
                    _taskPane.SetProgress("error");
                    MessageBox.Show("Validation request failed. Ensure server is running on http://localhost:8001\nDetails: " + errorMsg, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                _taskPane.SetProgress("gnn");

                if (result == null)
                {
                    _taskPane.SetProgress("error");
                    MessageBox.Show("Validation result was empty. Check server logs.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                _lastResult = result;
                Log(string.Format("ValidatePart OK: score={0}, violations={1}",
                    result?.overall_manufacturability_score, result?.violations?.Count));
                try { ApplyOverlay(result); } catch (Exception overlayEx) { Log("ApplyOverlay error: " + overlayEx.Message); }
                _taskPane.UpdateResults(result);
                _taskPane.SetProgress("done");
                
                var elapsed = DateTime.Now - startTime;
                _taskPane.SetValidationTime(elapsed.TotalSeconds);
            }
            catch (Exception ex)
            {
                MessageBox.Show("RunValidation error: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                _taskPane.SetProgress("error");
            }
        }

        public async System.Threading.Tasks.Task ExportPDF(ValidationResult result)
        {
            try
            {
                using (var sfd = new SaveFileDialog())
                {
                    sfd.Filter = "PDF Files (*.pdf)|*.pdf|HTML Files (*.html)|*.html|Markdown Files (*.md)|*.md";
                    sfd.FileName = string.Format("EUREKA_DFM_Report_{0}", result.part_id);
                    if (sfd.ShowDialog() == DialogResult.OK)
                    {
                        if (sfd.FileName.EndsWith(".pdf"))
                        {
                            var pdfBytes = await _client.GeneratePdfReport(result);
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
                        MessageBox.Show("Report exported successfully!", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Export failed: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private async System.Threading.Tasks.Task CheckAndStartBackend()
        {
            bool isAlive = false;
            try
            {
                string health = await _client.GetHealth();
                isAlive = true;
            }
            catch (Exception)
            {
            }

            if (!isAlive)
            {
                string exePath = System.Reflection.Assembly.GetExecutingAssembly().Location;
                string dir = System.IO.Path.GetDirectoryName(exePath);
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
                            await System.Threading.Tasks.Task.Delay(2000);
                            try
                            {
                                string health = await _client.GetHealth();
                                isAlive = true;
                                break;
                            }
                            catch {}
                        }
                    }
                    catch {}
                }
            }

            if (_taskPane != null)
            {
                _taskPane.SetBackendConnected(isAlive);
            }
        }
    }
}
