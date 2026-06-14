using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.Serialization;

[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

namespace EurekaAddin
{
    [DataContract]
    public class FaceGeometry
    {
        [DataMember] public string face_id { get; set; }
        [DataMember] public string face_type { get; set; } = "Plane";
        [DataMember] public double area_mm2 { get; set; }
        [DataMember] public double? thickness_mm { get; set; }
        [DataMember] public double? draft_angle_deg { get; set; }
        [DataMember] public double? radius_mm { get; set; }
        [DataMember] public double? depth_mm { get; set; }
        [DataMember] public double? width_mm { get; set; }
        [DataMember] public double? parent_wall_thickness_mm { get; set; }
        [DataMember] public string sw_feature_name { get; set; } = "";
        [DataMember] public string sw_feature_type { get; set; } = "";
        [DataMember] public double curvature_min { get; set; }
        [DataMember] public double curvature_max { get; set; }
        [DataMember] public double centroid_x { get; set; }
        [DataMember] public double centroid_y { get; set; }
        [DataMember] public double centroid_z { get; set; }
        [DataMember] public double normal_x { get; set; }
        [DataMember] public double normal_y { get; set; }
        [DataMember] public double normal_z { get; set; }
    }

    [DataContract]
    public class PartMetadata
    {
        [DataMember] public string filename { get; set; } = "";
        [DataMember] public string solidworks_part_number { get; set; } = "";
        [DataMember] public string process { get; set; } = "";
        [DataMember] public string material { get; set; } = "";
        [DataMember] public string bounding_box_mm { get; set; } = "";
        [DataMember] public int face_count { get; set; }
        [DataMember] public int feature_tree_depth { get; set; }
        [DataMember] public double? volume_mm3 { get; set; }
        [DataMember] public double? surface_area_mm2 { get; set; }
        [DataMember] public List<FaceGeometry> faces { get; set; } = new List<FaceGeometry>();
        [DataMember] public double nominal_wall_mm { get; set; } = 1.5;
        [DataMember] public string classification { get; set; } = "structural";
        [DataMember] public string pull_direction { get; set; } = "auto";
        [DataMember] public List<string> class_a_face_ids { get; set; } = new List<string>();
    }

    [DataContract]
    public class Violation
    {
        [DataMember] public string rule_id { get; set; }
        [DataMember] public string category { get; set; }
        [DataMember] public string severity { get; set; }
        [DataMember] public List<string> face_ids { get; set; } = new List<string>();
        [DataMember] public string solidworks_feature_name { get; set; } = "";
        [DataMember] public string measured_value { get; set; } = "";
        [DataMember] public string required_value { get; set; } = "";
        [DataMember] public string standard_reference { get; set; } = "";
        [DataMember] public string description { get; set; } = "";
        [DataMember] public string fix_suggestion { get; set; } = "";
        [DataMember] public string solidworks_fix_path { get; set; } = "";
        [DataMember] public int unaddressed_risk_score { get; set; } = 5;
        [DataMember] public string unaddressed_risk_reasoning { get; set; } = "";

        // Gemini-enriched fields
        [DataMember] public double? current_value_mm { get; set; }
        [DataMember] public double? minimum_required_mm { get; set; }
        [DataMember] public double? optimal_value_mm { get; set; }
        [DataMember] public double? fix_delta_mm { get; set; }
        [DataMember] public string plain_english { get; set; } = "";
        [DataMember] public string fix_instruction { get; set; } = "";
        [DataMember] public string highlight_color { get; set; } = "";
        [DataMember] public string status { get; set; } = "ACTIVE";

        // Compatibility properties
        public string Id { get { return rule_id; } set { rule_id = value; } }
        public string Severity { get { return severity; } set { severity = value; } }
        public string FaceId { get { return (face_ids != null && face_ids.Count > 0) ? face_ids[0] : ""; } set { if (face_ids == null) face_ids = new List<string>(); face_ids.Clear(); face_ids.Add(value); } }
        
        public double ParseNumericValue(string raw, double fallback)
        {
            if (string.IsNullOrEmpty(raw)) return fallback;
            try
            {
                var match = System.Text.RegularExpressions.Regex.Match(raw, @"([<>=!]*\s*)([-+]?\d+\.?\d*)\s*(deg|mm|°|deg from pull)?", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                if (match.Success)
                {
                    if (double.TryParse(match.Groups[2].Value, out double result))
                    {
                        return result;
                    }
                }
                
                var fallbackMatch = System.Text.RegularExpressions.Regex.Match(raw, @"[-+]?\d+\.?\d*");
                if (fallbackMatch.Success)
                {
                    if (double.TryParse(fallbackMatch.Value, out double result))
                    {
                        return result;
                    }
                }
            }
            catch { }
            return fallback;
        }

        public string Relation
        {
            get
            {
                if (string.IsNullOrEmpty(required_value)) return "≥";
                string r = required_value.ToLower();
                if (r.Contains("<=") || r.Contains("≤") || r.Contains("must not") || r.Contains("no undercut"))
                {
                    return "≤";
                }
                return "≥";
            }
        }


        public double MeasuredValue
        {
            get
            {
                if (current_value_mm.HasValue) return current_value_mm.Value;
                return ParseNumericValue(measured_value, 0.0);
            }
            set { current_value_mm = value; }
        }
        public double RequiredValue
        {
            get
            {
                if (minimum_required_mm.HasValue) return minimum_required_mm.Value;
                return ParseNumericValue(required_value, 0.0);
            }
            set { minimum_required_mm = value; }
        }
        public string Unit
        {
            get
            {
                string r = (rule_id ?? "").ToLower();
                if (r.Contains("draft") || r.Contains("angle") || r.Contains("undercut") || (measured_value ?? "").Contains("°") || (measured_value ?? "").Contains("deg") || (required_value ?? "").Contains("°") || (required_value ?? "").Contains("deg"))
                {
                    return "°";
                }
                return "mm";
            }
        }
        public string Message { get { return !string.IsNullOrEmpty(plain_english) ? plain_english : description; } }
        public string FixSuggestion { get { return !string.IsNullOrEmpty(fix_instruction) ? fix_instruction : (fix_suggestion ?? ""); } set { fix_suggestion = value; } }
        public string Source { get { return (rule_id ?? "").StartsWith("GNN-") ? "GNN" : "Rules"; } }
    }

    [DataContract]
    public class ValidationResult
    {
        [DataMember] public string part_id { get; set; } = "";
        [DataMember] public int overall_manufacturability_score { get; set; } = 100;
        [DataMember] public Dictionary<string, object> risk_summary { get; set; } = new Dictionary<string, object>();
        [DataMember] public List<Violation> violations { get; set; } = new List<Violation>();
        [DataMember] public List<string> passed_checks { get; set; } = new List<string>();
        [DataMember] public bool engineer_review_required { get; set; }
        [DataMember] public double confidence { get; set; } = 1.0;
        [DataMember] public double gnn_risk_score { get; set; }
        [DataMember] public Newtonsoft.Json.Linq.JObject gnn_anomaly { get; set; }
        [DataMember] public bool gemini_enriched { get; set; }
        [DataMember] public string process { get; set; } = "";
        [DataMember] public string material { get; set; } = "";
        [DataMember] public Dictionary<string, object> face_health { get; set; } = new Dictionary<string, object>();
        [DataMember] public string screenshot_png_base64 { get; set; } = "";
        [DataMember] public string gemini_narrative { get; set; } = "";

        // Compatibility properties
        public int Score { get { return overall_manufacturability_score; } set { overall_manufacturability_score = value; } }
        public string RiskLevel
        {
            get
            {
                if (Score < 50) return "HIGH RISK";
                if (Score < 80) return "MEDIUM RISK";
                return "LOW RISK";
            }
        }
        public ValidationSummary Summary
        {
            get
            {
                int crit = 0, warn = 0, inf = 0;
                if (violations != null)
                {
                    foreach (var v in violations)
                    {
                        if (v.severity == null) continue;
                        string s = v.severity.ToUpper();
                        if (s == "CRITICAL") crit++;
                        else if (s == "WARNING") warn++;
                        else inf++;
                    }
                }
                return new ValidationSummary { Critical = crit, Warning = warn, Info = inf };
            }
        }
        public AIAnalysis AiAnalysis
        {
            get
            {
                return new AIAnalysis
                {
                    RiskBar = (int)(gnn_risk_score * 100),
                    Mode = "GNN",
                    Confidence = (int)(confidence * 100)
                };
            }
        }
        public HeatmapData Heatmap
        {
            get
            {
                int GetVal(string key)
                {
                    if (face_health != null && face_health.TryGetValue(key, out object o))
                    {
                        try { return Convert.ToInt32(o); } catch { }
                    }
                    return 0;
                }
                return new HeatmapData
                {
                    Critical = GetVal("critical"),
                    AtRisk = GetVal("at_risk"),
                    Watch = GetVal("watch"),
                    Good = GetVal("clean")
                };
            }
        }
        public List<Violation> Violations { get { return violations; } set { violations = value; } }
        public string PullDirection { get { return pull_direction; } set { pull_direction = value; } }
        public bool EngineerReviewRequired { get { return engineer_review_required; } set { engineer_review_required = value; } }

        private string pull_direction = "auto";
    }

    public class ValidationSummary
    {
        public int Critical { get; set; }
        public int Warning { get; set; }
        public int Info { get; set; }
    }

    public class HeatmapData
    {
        public int Critical { get; set; }
        public int AtRisk { get; set; }
        public int Watch { get; set; }
        public int Good { get; set; }
    }

    public class AIAnalysis
    {
        public int RiskBar { get; set; }
        public string Mode { get; set; }
        public int Confidence { get; set; }
    }

    public enum Severity
    {
        CRITICAL, WARNING, INFO
    }

    public enum FaceType
    {
        Plane, Cylinder, Sphere, Cone, Torus, BSpline, Other
    }
}
