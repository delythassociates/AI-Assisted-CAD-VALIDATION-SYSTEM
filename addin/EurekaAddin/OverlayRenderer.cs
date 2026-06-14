using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

namespace EurekaAddin
{
    public class OverlayRenderer
    {
        private readonly SldWorks _swApp;
        private readonly Dictionary<string, IFace2> _faceCache = new Dictionary<string, IFace2>();
        private readonly List<INote> _activeNotes = new List<INote>();
        private List<Violation> _lastViolations = new List<Violation>();

        public OverlayRenderer(SldWorks swApp)
        {
            _swApp = swApp;
        }

        public void ApplyOverlay(List<Violation> violations, IModelDoc2 doc, Dictionary<string, IFace2> faceRegistry = null)
        {
            try
            {
                Log("ApplyOverlay starting.");
                _lastViolations = violations ?? new List<Violation>();

                var part = doc as IPartDoc;
                if (part == null)
                {
                    Log("Document is not a part document.");
                    return;
                }

                if (faceRegistry != null)
                {
                    _faceCache.Clear();
                    foreach (var kvp in faceRegistry)
                    {
                        _faceCache[kvp.Key] = kvp.Value;
                    }
                }
                else if (_faceCache.Count == 0)
                {
                    BuildFaceCache(part);
                }

                // Map face_id -> Violation
                var violationMap = new Dictionary<string, Violation>();
                foreach (var v in _lastViolations)
                {
                    if (v == null || string.IsNullOrEmpty(v.FaceId)) continue;

                    if (!violationMap.ContainsKey(v.FaceId))
                    {
                        violationMap[v.FaceId] = v;
                    }
                    else
                    {
                        var existing = violationMap[v.FaceId];
                        if (GetSeverityPriority(v.Severity) > GetSeverityPriority(existing.Severity))
                        {
                            violationMap[v.FaceId] = v;
                        }
                    }
                }

                // Colors
                Color passColor = Color.FromArgb(82, 196, 122); // green

                // Apply colors
                foreach (var kvp in _faceCache)
                {
                    string faceId = kvp.Key;
                    IFace2 face = kvp.Value;

                    Color colorToApply = passColor;
                    if (violationMap.TryGetValue(faceId, out Violation v))
                    {
                        colorToApply = GetSeverityColor(v.Severity);
                    }

                    ApplyFaceColor(face, colorToApply, doc);
                }

                doc.ClearSelection2(true);
                doc.GraphicsRedraw2();
                Log(string.Format("ApplyOverlay completed successfully. Colored {0} faces.", _faceCache.Count));
            }
            catch (Exception ex)
            {
                Log("ApplyOverlay error: " + ex.Message + "\n" + ex.StackTrace);
            }
        }

        public void ClearOverlay(IModelDoc2 doc)
        {
            try
            {
                Log("ClearOverlay starting.");
                var part = doc as IPartDoc;
                if (part == null) return;

                if (_faceCache.Count == 0)
                {
                    BuildFaceCache(part);
                }

                foreach (var face in _faceCache.Values)
                {
                    try
                    {
                        face.RemoveMaterialProperty2((int)swInConfigurationOpts_e.swThisConfiguration, null);
                    }
                    catch (Exception faceEx)
                    {
                        Log("RemoveMaterialProperty2 in ClearOverlay error: " + faceEx.Message);
                    }
                }

                doc.ClearSelection2(true);
                doc.GraphicsRedraw2();
                Log("ClearOverlay completed.");
            }
            catch (Exception ex)
            {
                Log("ClearOverlay error: " + ex.Message);
            }
        }

        public void HighlightViolation(Violation v, IModelDoc2 doc)
        {
            try
            {
                Log("HighlightViolation face: " + v.FaceId);
                // 1. Zoom to face and rotate
                ZoomToFace(v.FaceId, doc);

                // 2. Show annotation callout note
                ShowDimensionNote(v, doc);
            }
            catch (Exception ex)
            {
                Log("HighlightViolation error: " + ex.Message);
            }
        }

        public void ClearHighlight(IModelDoc2 doc)
        {
            try
            {
                Log("ClearHighlight starting.");
                ClearDimensionNotes(doc);

                // Re-apply overlay to restore colors
                ApplyOverlay(_lastViolations, doc);
            }
            catch (Exception ex)
            {
                Log("ClearHighlight error: " + ex.Message);
            }
        }

        public void InvalidateCache()
        {
            Log("Invalidating face cache.");
            _faceCache.Clear();
            _activeNotes.Clear();
        }

        // Private helpers
        private void BuildFaceCache(IPartDoc part)
        {
            try
            {
                _faceCache.Clear();
                object[] bodies = (object[])part.GetBodies2((int)swBodyType_e.swSolidBody, false);
                if (bodies == null) return;

                int faceIndex = 1;
                foreach (var bodyObj in bodies)
                {
                    var body = bodyObj as Body2;
                    if (body == null) continue;

                    object[] faces = (object[])body.GetFaces();
                    if (faces == null) continue;

                    foreach (var faceObj in faces)
                    {
                        var face = faceObj as IFace2;
                        if (face == null) continue;

                        string faceId = faceIndex.ToString();
                        _faceCache[faceId] = face;
                        faceIndex++;
                    }
                }
                Log(string.Format("Built face cache with {0} items.", _faceCache.Count));
            }
            catch (Exception ex)
            {
                Log("BuildFaceCache error: " + ex.Message);
            }
        }

        private void ApplyFaceColor(IFace2 face, Color color, IModelDoc2 doc)
        {
            try
            {
                double[] prop = new double[9];
                prop[0] = color.R / 255.0; // Red
                prop[1] = color.G / 255.0; // Green
                prop[2] = color.B / 255.0; // Blue
                prop[3] = 0.5; // Ambient
                prop[4] = 0.6; // Diffuse
                prop[5] = 0.4; // Specular
                prop[6] = 0.5; // Shininess
                prop[7] = 0.0; // Transparency
                prop[8] = 0.0; // Emission
                face.MaterialPropertyValues = prop;
            }
            catch (Exception ex)
            {
                Log("ApplyFaceColor error: " + ex.Message);
            }
        }

        private void ShowDimensionNote(Violation v, IModelDoc2 doc)
        {
            try
            {
                ClearDimensionNotes(doc);

                if (!_faceCache.TryGetValue(v.FaceId, out IFace2 face))
                {
                    Log("ShowDimensionNote: Face not found in cache: " + v.FaceId);
                    return;
                }

                double[] bbox = (double[])face.GetBox();
                if (bbox == null || bbox.Length < 6) return;

                double cx = (bbox[0] + bbox[3]) / 2.0;
                double cy = (bbox[1] + bbox[4]) / 2.0;
                double cz = (bbox[2] + bbox[5]) / 2.0;

                double delta = v.RequiredValue - v.MeasuredValue;
                string sign = delta >= 0 ? "+" : "";
                
                // Determine comparison sign: if measured > required, it's typically a max limit violation (so we need <= required)
                // if measured < required, it's typically a min limit violation (so we need >= required)
                string relation = "≥";
                if (v.MeasuredValue > v.RequiredValue)
                {
                    relation = "≤";
                }
                
                string noteText = string.Format("{0}\n{1:F2}{2} → {3}{4:F2}{5}\nFix: {6}{7:F2}{8}",
                    v.Id,
                    v.MeasuredValue, v.Unit,
                    relation, v.RequiredValue, v.Unit,
                    sign, delta, v.Unit);

                doc.ClearSelection2(true);
                var note = (INote)doc.InsertNote(noteText);
                if (note != null)
                {
                    dynamic dynNote = note;
                    dynNote.SetLeader2(true, 0, true, 0, 0, 0);
                    var ann = (IAnnotation)note.GetAnnotation();
                    if (ann != null)
                    {
                        // Offset note slightly from centroid
                        ann.SetPosition2(cx + 0.02, cy + 0.02, cz + 0.02);
                    }
                    _activeNotes.Add(note);
                    Log("Successfully inserted note for face: " + v.FaceId);
                }
                else
                {
                    Log("InsertNote returned null.");
                }

                doc.GraphicsRedraw2();
            }
            catch (Exception ex)
            {
                Log("ShowDimensionNote error: " + ex.Message);
            }
        }

        private void ClearDimensionNotes(IModelDoc2 doc)
        {
            try
            {
                if (_activeNotes.Count == 0) return;

                doc.ClearSelection2(true);
                foreach (var note in _activeNotes)
                {
                    try
                    {
                        var ann = (IAnnotation)note.GetAnnotation();
                        if (ann != null)
                        {
                            ann.Select3(true, null); // Append selection
                        }
                    }
                    catch (Exception selectEx)
                    {
                        Log("Select note error: " + selectEx.Message);
                    }
                }

                doc.Extension.DeleteSelection2((int)swDeleteSelectionOptions_e.swDelete_Absorbed);
                _activeNotes.Clear();
                doc.GraphicsRedraw2();
                Log("Cleared active dimension notes.");
            }
            catch (Exception ex)
            {
                Log("ClearDimensionNotes error: " + ex.Message);
            }
        }

        public Dictionary<string, IFace2> FaceCache => _faceCache;

        /// <summary>
        /// Zooms to a specific face, orients camera Normal-To, captures
        /// a cropped viewport screenshot, returns base64 PNG string.
        /// </summary>
        public string CaptureViolationSnapshot(IFace2 face, string faceId)
        {
            try
            {
                var doc = _swApp.IActiveDoc2 as ModelDoc2;
                if (doc == null) return null;

                // 1. Zoom camera to this face's bounding box
                ZoomToFace(faceId, doc);  // existing method — zooms + orients Normal-To

                // 2. Force SW to redraw and settle
                doc.GraphicsRedraw2();
                System.Threading.Thread.Sleep(150); // let viewport settle

                // 3. Capture the active viewport as bitmap
                ModelView activeView = doc.ActiveView as ModelView;
                if (activeView == null) return null;

                // Capture screen region using SolidWorks SaveBMP
                string tempPath = Path.Combine(
                    Path.GetTempPath(),
                    string.Format("dfm_face_{0}_{1:N}.png", faceId, Guid.NewGuid()));

                // Use SolidWorks built-in image capture
                bool saveResult = doc.SaveBMP(tempPath, 0, 0);

                if (!saveResult || !File.Exists(tempPath))
                    return null;

                // 4. Convert to base64
                byte[] imgBytes = File.ReadAllBytes(tempPath);
                File.Delete(tempPath);
                return Convert.ToBase64String(imgBytes);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(string.Format("[Snapshot] Failed for {0}: {1}", faceId, ex.Message));
                return null;
            }
        }

        private void ZoomToFace(string faceId, IModelDoc2 doc)
        {
            try
            {
                if (!_faceCache.TryGetValue(faceId, out IFace2 face))
                {
                    Log("ZoomToFace: face not in cache: " + faceId);
                    return;
                }

                doc.ClearSelection2(true);

                // Select the face using dynamic to bypass Select2 COM RCW issues
                try
                {
                    dynamic dyn = face;
                    dyn.Select2(false, 0);
                }
                catch (Exception selEx)
                {
                    Log("Dynamic Select2 failed: " + selEx.Message);
                }

                double[] bbox = (double[])face.GetBox();
                if (bbox != null && bbox.Length >= 6)
                {
                    // Add 30mm padding (0.03 meters)
                    double pad = 0.03;
                    doc.ViewZoomTo2(
                        bbox[0] - pad, bbox[1] - pad, bbox[2] - pad,
                        bbox[3] + pad, bbox[4] + pad, bbox[5] + pad
                    );
                }

                // Rotate viewport normal to selected face
                doc.ShowNamedView2("*Normal To", -1);

                // Deselect face
                doc.ClearSelection2(true);

                // Re-apply color overlay to this face specifically so it doesn't lose color
                Color faceColor = Color.FromArgb(82, 196, 122); // Good
                foreach (var v in _lastViolations)
                {
                    if (v != null && v.FaceId == faceId)
                    {
                        faceColor = GetSeverityColor(v.Severity);
                        break;
                    }
                }
                ApplyFaceColor(face, faceColor, doc);
                doc.ClearSelection2(true);
            }
            catch (Exception ex)
            {
                Log("ZoomToFace error: " + ex.Message);
            }
        }

        private Color GetSeverityColor(string severity)
        {
            switch ((severity ?? "").ToUpper())
            {
                case "CRITICAL": return Color.FromArgb(224, 82, 82);
                case "WARNING":  return Color.FromArgb(224, 154, 58);
                case "INFO":     return Color.FromArgb(74, 158, 224);
                default:         return Color.FromArgb(82, 196, 122);
            }
        }

        private int GetSeverityPriority(string severity)
        {
            if (string.IsNullOrEmpty(severity)) return 0;
            switch (severity.ToUpper())
            {
                case "CRITICAL": return 3;
                case "WARNING":  return 2;
                case "INFO":     return 1;
                default:         return 0;
            }
        }

        private int Rgb(int r, int g, int b)
        {
            return (b << 16) | (g << 8) | r;
        }

        private void Log(string message)
        {
            try
            {
                string logPath = Path.Combine(Path.GetTempPath(), "eureka_debug.log");
                File.AppendAllText(logPath, string.Format("[{0:HH:mm:ss}] {1}\r\n", DateTime.Now, message));
            }
            catch { }
        }
    }
}
