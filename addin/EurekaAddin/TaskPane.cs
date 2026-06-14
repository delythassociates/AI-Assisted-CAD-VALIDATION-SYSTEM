using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.Linq;
using System.Windows.Forms;

namespace EurekaAddin
{
    public partial class TaskPane : UserControl
    {
        // Static colors defined as fields (no magic hex literals)
        public static readonly Color BgMain        = Color.FromArgb(30, 34, 39);      // #1E2227 - main background
        public static readonly Color BgSurface     = Color.FromArgb(39, 43, 51);      // #272B33 - card/section bg
        public static readonly Color BgRaised      = Color.FromArgb(47, 53, 64);      // #2F3540 - elevated, hover
        public static readonly Color BorderColor   = Color.FromArgb(58, 63, 75);      // #3A3F4B - dividers
        public static readonly Color ColorCritical = Color.FromArgb(224, 82, 82);     // #E05252 - critical
        public static readonly Color ColorWarning  = Color.FromArgb(224, 154, 58);    // #E09A3A - warning
        public static readonly Color ColorInfo     = Color.FromArgb(74, 158, 224);    // #4A9EE0 - info
        public static readonly Color ColorGood     = Color.FromArgb(82, 196, 122);    // #52C47A - good
        public static readonly Color TextPrimary   = Color.FromArgb(232, 234, 240);   // #E8EAF0 - primary text
        public static readonly Color TextSecondary = Color.FromArgb(139, 145, 158);   // #8B919E - secondary text
        public static readonly Color AccentBlue    = Color.FromArgb(74, 128, 224);    // #4A80E0 - accent blue

        // Private control fields
        private FlowLayoutPanel flpRoot;

        // Section 1: Header
        private Panel pnlHeader;
        private Label lblHeaderTitle;
        private Panel pnlStatusDot;
        private Label lblValidationTime;

        // Section 2: Input Row
        private Panel pnlInput;
        private Label lblConfigHeader;
        private Label lblConfigSummary;
        private bool _configExpanded = true;
        private Label lblWall;
        private TextBox txtWallThickness;
        private Button btnValidate;
        private Button btnClear;
        private Label lblProcess;
        private ComboBox cmbProcess;
        private Label lblMaterial;
        private ComboBox cmbMaterial;

        // Section 3: Progress Stepper
        private Panel pnlStepper;

        // Section 4: Score Card
        private Panel pnlScore;
        private Panel pnlScoreArc;

        // Section 5: Engineer Review Banner
        private Panel pnlReviewBanner;
        private Label lblReviewBanner;
        private Panel pnlGeminiBadge;
        private Label lblGeminiBadge;

        // Section 6: AI Analysis Card
        private Panel pnlAiAnalysis;
        private Label lblAiTitle;
        private Label lblAiRiskLabel;
        private Label lblAiRiskVal;
        private Panel pnlAiProgressBar;
        private Label lblAiModeConfidence;

        // Flywheel Feedback Controls (embedded in pnlAiAnalysis)
        private Label lblFeedbackText;
        private Button btnFeedbackAccurate;
        private Button btnFeedbackOverride;
        private Label lblFeedbackStatus;

        // Section 7: Risk Heatmap Card (merged into Score Panel)
        private Panel[] pnlHeatmapCells;

        // Section 8: Post Validation Controls
        private Panel pnlPostValidation;
        private Label lblPostValTitle;
        private Label lblPullDirLabel;
        private ComboBox cmbPullDir;
        private Button btnMarkClassA;
        private Button btnExportReport;

        // Section 9: Violations List
        private Panel pnlViolations;
        private Panel pnlViolationsHeader;
        private Label lblFaceHighlight;
        private Panel pnlViolationsList;

        // Timers
        private System.Windows.Forms.Timer tmrFaceHighlight;

        // State variables
        private int _currentStep = -1; // -1 = idle, 0 = extract, 1 = rules, 2 = GNN, 3 = done, -2 = error
        private int _lastActiveStep = -1;
        private ValidationResult _lastResult;
        private bool _backendConnected = false;
        private bool _isInitialized = false;

        // Event Declarations
        public event EventHandler ValidateClicked;
        public event EventHandler ClearClicked;
        public event EventHandler<string> FaceSelected;
        public event EventHandler<string> PullDirectionChanged;
        public event EventHandler ExportReportClicked;
        public event EventHandler<bool> FeedbackClicked; // true = accurate, false = override

        public TaskPane()
        {
            this.DoubleBuffered = true;
            this.Width = 300;
            this.Height = 800;
            this.Dock = DockStyle.Fill;
            this.BackColor = BgMain;

            InitializeComponent();
            InitializeLayout();
            InitializeColors();
            _isInitialized = true;
        }

        private void InitializeComponent()
        {
            // Root panel
            flpRoot = new FlowLayoutPanel
            {
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoScroll = true,
                Dock = DockStyle.Fill,
                BackColor = BgMain,
                Padding = new Padding(10, 0, 10, 0)
            };
            this.Controls.Add(flpRoot);

            // SECTION 1: HEADER
            pnlHeader = new Panel { Size = new Size(280, 32), Margin = new Padding(0, 8, 0, 0), BackColor = BgMain };
            lblHeaderTitle = new Label
            {
                Text = "EUREKA DFM",
                Font = new Font("Segoe UI", 10f, FontStyle.Bold),
                ForeColor = AccentBlue,
                Location = new Point(0, 8),
                AutoSize = true
            };
            lblValidationTime = new Label
            {
                Text = "",
                Font = new Font("Segoe UI", 8.25f, FontStyle.Italic),
                ForeColor = TextSecondary,
                Location = new Point(110, 10),
                Size = new Size(150, 20),
                TextAlign = ContentAlignment.MiddleRight,
                Visible = false
            };
            pnlStatusDot = new Panel { Size = new Size(10, 10), Location = new Point(270, 11) };
            pnlStatusDot.Paint += PnlStatusDot_Paint;
            pnlHeader.Controls.AddRange(new Control[] { lblHeaderTitle, lblValidationTime, pnlStatusDot });
            flpRoot.Controls.Add(pnlHeader);

            // SECTION 2: INPUT ROW
            pnlInput = new Panel { Size = new Size(280, 150), Margin = new Padding(0, 4, 0, 0), BackColor = BgSurface };
            
            lblConfigHeader = new Label
            {
                Text = "CONFIG  ▲",
                Font = new Font("Segoe UI", 8f, FontStyle.Bold),
                ForeColor = TextSecondary,
                Location = new Point(10, 5),
                Size = new Size(80, 18),
                TextAlign = ContentAlignment.MiddleLeft,
                Cursor = Cursors.Hand
            };
            lblConfigSummary = new Label
            {
                Text = "",
                Font = new Font("Segoe UI", 8f, FontStyle.Italic),
                ForeColor = TextSecondary,
                Location = new Point(90, 5),
                Size = new Size(180, 18),
                TextAlign = ContentAlignment.MiddleLeft,
                Visible = false,
                Cursor = Cursors.Hand
            };

            lblConfigHeader.Click += (s, e) => ToggleConfigExpanded();
            lblConfigSummary.Click += (s, e) => ToggleConfigExpanded();
            pnlInput.Click += (s, e) => {
                if (e is MouseEventArgs me && me.Y < 24)
                {
                    ToggleConfigExpanded();
                }
            };

            lblWall = new Label
            {
                Text = "Wall (mm)",
                Font = new Font("Segoe UI", 9f),
                ForeColor = TextSecondary,
                Location = new Point(10, 28),
                Size = new Size(72, 20),
                TextAlign = ContentAlignment.MiddleLeft
            };
            txtWallThickness = new TextBox
            {
                Text = "3",
                Font = new Font("Segoe UI", 9f),
                TextAlign = HorizontalAlignment.Center,
                Location = new Point(86, 26),
                Size = new Size(32, 22),
                BackColor = BgRaised,
                ForeColor = TextPrimary,
                BorderStyle = BorderStyle.FixedSingle
            };
            
            lblPullDirLabel = new Label
            {
                Text = "Pull Dir",
                Font = new Font("Segoe UI", 9f),
                ForeColor = TextSecondary,
                Location = new Point(126, 28),
                Size = new Size(82, 20),
                TextAlign = ContentAlignment.MiddleLeft
            };
            cmbPullDir = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList,
                Font = new Font("Segoe UI", 9f),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(210, 26),
                Size = new Size(70, 22),
                BackColor = BgRaised,
                ForeColor = TextPrimary
            };
            cmbPullDir.Items.AddRange(new object[] { "+X", "-X", "+Y", "-Y", "+Z", "-Z" });
            cmbPullDir.SelectedIndex = 4; // default to +Z
            cmbPullDir.SelectedIndexChanged += (s, e) => {
                PullDirectionChanged?.Invoke(this, cmbPullDir.SelectedItem?.ToString());
            };

            string lastWallText = txtWallThickness.Text;
            txtWallThickness.Leave += (s, e) => {
                if (_isInitialized && txtWallThickness.Text != lastWallText)
                {
                    lastWallText = txtWallThickness.Text;
                }
            };
            txtWallThickness.KeyDown += (s, e) => {
                if (e.KeyCode == Keys.Enter)
                {
                    e.SuppressKeyPress = true;
                    if (_isInitialized && txtWallThickness.Text != lastWallText)
                    {
                        lastWallText = txtWallThickness.Text;
                    }
                }
            };

            btnValidate = new Button
            {
                Text = "Validate",
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(86, 118),
                Size = new Size(130, 24),
                BackColor = AccentBlue,
                ForeColor = Color.White,
                Cursor = Cursors.Hand
            };
            btnValidate.FlatAppearance.BorderSize = 0;
            btnValidate.Click += (s, e) => ValidateClicked?.Invoke(this, EventArgs.Empty);

            btnClear = new Button
            {
                Text = "Clear",
                Font = new Font("Segoe UI", 9f),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(224, 118),
                Size = new Size(56, 24),
                BackColor = BgRaised,
                ForeColor = TextSecondary,
                Cursor = Cursors.Hand
            };
            btnClear.FlatAppearance.BorderSize = 0;
            btnClear.Click += (s, e) => ClearClicked?.Invoke(this, EventArgs.Empty);

            lblProcess = new Label
            {
                Text = "Process",
                Font = new Font("Segoe UI", 9f),
                ForeColor = TextSecondary,
                Location = new Point(10, 58),
                Size = new Size(72, 20),
                TextAlign = ContentAlignment.MiddleLeft
            };
            cmbProcess = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList,
                Font = new Font("Segoe UI", 9f),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(86, 56),
                Size = new Size(194, 22),
                BackColor = BgRaised,
                ForeColor = TextPrimary
            };
            cmbProcess.Items.AddRange(new object[] {
                "Injection Moulding",
                "Die Casting (Aluminium)",
                "Die Casting (Zinc)",
                "Die Casting (Magnesium)"
            });
            cmbProcess.SelectedIndex = 0;
            cmbProcess.SelectedIndexChanged += (s, e) => {
                UpdateMaterialsDropdown();
            };

            lblMaterial = new Label
            {
                Text = "Material",
                Font = new Font("Segoe UI", 9f),
                ForeColor = TextSecondary,
                Location = new Point(10, 88),
                Size = new Size(72, 20),
                TextAlign = ContentAlignment.MiddleLeft
            };
            cmbMaterial = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList,
                Font = new Font("Segoe UI", 9f),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(86, 86),
                Size = new Size(194, 22),
                BackColor = BgRaised,
                ForeColor = TextPrimary
            };
            UpdateMaterialsDropdown();

            pnlInput.Controls.AddRange(new Control[] { 
                lblConfigHeader, lblConfigSummary,
                lblWall, txtWallThickness, lblPullDirLabel, cmbPullDir,
                btnValidate, btnClear, 
                lblProcess, cmbProcess, lblMaterial, cmbMaterial 
            });
            flpRoot.Controls.Add(pnlInput);

            // SECTION 3: PROGRESS STEPPER
            pnlStepper = new Panel { Size = new Size(280, 48), Margin = new Padding(0, 4, 0, 0), BackColor = BgSurface, Visible = false };
            pnlStepper.Paint += PnlStepper_Paint;
            flpRoot.Controls.Add(pnlStepper);

            // SECTION 4: SCORE CARD
            pnlScore = new Panel { Size = new Size(280, 76), Margin = new Padding(0, 4, 0, 0), BackColor = BgSurface };
            pnlScoreArc = new Panel { Size = new Size(80, 76), Location = new Point(0, 0), BackColor = BgSurface };
            pnlScoreArc.Paint += PnlScoreArc_Paint;
            
            pnlScore.Controls.AddRange(new Control[] { pnlScoreArc });
            CreateHeatmapCells();
            flpRoot.Controls.Add(pnlScore);

            // SECTION 5: ENGINEER REVIEW BANNER
            pnlReviewBanner = new Panel { Size = new Size(280, 30), Margin = new Padding(0, 4, 0, 0), BackColor = Color.FromArgb(61, 42, 21), Visible = false };
            pnlReviewBanner.Paint += PnlReviewBanner_Paint;
            lblReviewBanner = new Label
            {
                Text = "⚠  Engineer review required before tooling",
                Font = new Font("Segoe UI", 9f),
                ForeColor = ColorWarning,
                Location = new Point(12, 0),
                Size = new Size(268, 30),
                TextAlign = ContentAlignment.MiddleLeft
            };
            pnlReviewBanner.Controls.Add(lblReviewBanner);
            flpRoot.Controls.Add(pnlReviewBanner);

            // SECTION 5b: GEMINI ENRICHED BADGE
            pnlGeminiBadge = new Panel { Size = new Size(280, 28), Margin = new Padding(0, 4, 0, 0), BackColor = Color.FromArgb(42, 34, 64), Visible = false };
            pnlGeminiBadge.Paint += PnlGeminiBadge_Paint;
            lblGeminiBadge = new Label
            {
                Text = "✦ Gemini AI: enriched 0 findings",
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                ForeColor = Color.FromArgb(180, 160, 240),
                Location = new Point(12, 0),
                Size = new Size(268, 28),
                TextAlign = ContentAlignment.MiddleLeft
            };
            pnlGeminiBadge.Controls.Add(lblGeminiBadge);
            flpRoot.Controls.Add(pnlGeminiBadge);

            // SECTION 6: AI ANALYSIS CARD
            pnlAiAnalysis = new Panel { Size = new Size(280, 64), Margin = new Padding(0, 4, 0, 0), BackColor = BgSurface };
            lblAiTitle = new Label
            {
                Text = "AI ANALYSIS",
                Font = new Font("Segoe UI", 8f, FontStyle.Bold),
                ForeColor = TextSecondary,
                Location = new Point(10, 6),
                AutoSize = true
            };
            lblAiRiskLabel = new Label
            {
                Text = "Manufacturing Risk",
                Font = new Font("Segoe UI", 9f),
                ForeColor = TextPrimary,
                Location = new Point(10, 22),
                AutoSize = true
            };
            lblAiRiskVal = new Label
            {
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                Location = new Point(200, 22),
                Size = new Size(70, 16),
                TextAlign = ContentAlignment.MiddleRight
            };
            pnlAiProgressBar = new Panel { Size = new Size(260, 6), Location = new Point(10, 38) };
            pnlAiProgressBar.Paint += PnlAiProgressBar_Paint;
            lblAiModeConfidence = new Label
            {
                Font = new Font("Segoe UI", 8f),
                ForeColor = TextSecondary,
                Location = new Point(10, 48),
                Size = new Size(260, 14)
            };

            // Feedback controls inside pnlAiAnalysis
            lblFeedbackText = new Label
            {
                Text = "Is this AI risk rating accurate?",
                Font = new Font("Segoe UI", 8.5f),
                ForeColor = TextPrimary,
                Location = new Point(10, 66),
                AutoSize = true,
                Visible = false
            };
            
            btnFeedbackAccurate = new Button
            {
                Text = "✓ Accurate",
                Font = new Font("Segoe UI", 8f, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(10, 84),
                Size = new Size(125, 22),
                BackColor = BgRaised,
                ForeColor = ColorGood,
                Cursor = Cursors.Hand,
                Visible = false
            };
            btnFeedbackAccurate.FlatAppearance.BorderColor = ColorGood;
            btnFeedbackAccurate.FlatAppearance.BorderSize = 1;
            btnFeedbackAccurate.Click += (s, e) => {
                btnFeedbackAccurate.Enabled = false;
                btnFeedbackOverride.Enabled = false;
                FeedbackClicked?.Invoke(this, true);
            };

            btnFeedbackOverride = new Button
            {
                Text = "✗ Override",
                Font = new Font("Segoe UI", 8f, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(145, 84),
                Size = new Size(125, 22),
                BackColor = BgRaised,
                ForeColor = ColorWarning,
                Cursor = Cursors.Hand,
                Visible = false
            };
            btnFeedbackOverride.FlatAppearance.BorderColor = ColorWarning;
            btnFeedbackOverride.FlatAppearance.BorderSize = 1;
            btnFeedbackOverride.Click += (s, e) => {
                btnFeedbackAccurate.Enabled = false;
                btnFeedbackOverride.Enabled = false;
                FeedbackClicked?.Invoke(this, false);
            };
            
            lblFeedbackStatus = new Label
            {
                Text = "",
                Font = new Font("Segoe UI", 8f, FontStyle.Italic),
                ForeColor = TextSecondary,
                Location = new Point(10, 110),
                Size = new Size(260, 14),
                Visible = false
            };

            pnlAiAnalysis.Controls.AddRange(new Control[] { 
                lblAiTitle, lblAiRiskLabel, lblAiRiskVal, pnlAiProgressBar, lblAiModeConfidence,
                lblFeedbackText, btnFeedbackAccurate, btnFeedbackOverride, lblFeedbackStatus
            });
            flpRoot.Controls.Add(pnlAiAnalysis);

            // SECTION 8: POST-VALIDATION CONTROLS
            pnlPostValidation = new Panel { Size = new Size(280, 56), Margin = new Padding(0, 4, 0, 0), BackColor = BgSurface, Visible = false };
            lblPostValTitle = new Label
            {
                Text = "POST-VALIDATION",
                Font = new Font("Segoe UI", 8f, FontStyle.Bold),
                ForeColor = TextSecondary,
                Location = new Point(10, 4),
                AutoSize = true
            };

            btnMarkClassA = new Button
            {
                Text = "Mark Class A",
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(10, 20),
                Size = new Size(125, 26),
                BackColor = BgRaised,
                ForeColor = AccentBlue,
                Cursor = Cursors.Hand
            };
            btnMarkClassA.FlatAppearance.BorderColor = AccentBlue;
            btnMarkClassA.FlatAppearance.BorderSize = 1;

            btnExportReport = new Button
            {
                Text = "Export PDF",
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(145, 20),
                Size = new Size(125, 26),
                BackColor = AccentBlue,
                ForeColor = Color.White,
                Cursor = Cursors.Hand
            };
            btnExportReport.FlatAppearance.BorderSize = 0;
            btnExportReport.Click += (s, e) => ExportReportClicked?.Invoke(this, EventArgs.Empty);

            pnlPostValidation.Controls.AddRange(new Control[] { lblPostValTitle, btnMarkClassA, btnExportReport });
            flpRoot.Controls.Add(pnlPostValidation);

            // SECTION 9: VIOLATIONS LIST
            pnlViolations = new Panel { Size = new Size(280, 300), Margin = new Padding(0, 4, 0, 0), BackColor = BgMain };
            pnlViolationsHeader = new Panel { Size = new Size(280, 24), Location = new Point(0, 0), BackColor = BgMain };
            pnlViolationsHeader.Paint += PnlViolationsHeader_Paint;
            
            lblFaceHighlight = new Label
            {
                Font = new Font("Segoe UI", 8.5f),
                ForeColor = AccentBlue,
                Location = new Point(0, 24),
                Size = new Size(280, 16),
                Visible = false
            };

            pnlViolationsList = new Panel
            {
                Location = new Point(0, 42),
                Size = new Size(280, 258),
                AutoScroll = true,
                BackColor = BgMain
            };

            pnlViolations.Controls.AddRange(new Control[] { pnlViolationsHeader, lblFaceHighlight, pnlViolationsList });
            flpRoot.Controls.Add(pnlViolations);

            // Timers
            tmrFaceHighlight = new System.Windows.Forms.Timer { Interval = 3000 };
            tmrFaceHighlight.Tick += (s, e) => { tmrFaceHighlight.Stop(); lblFaceHighlight.Visible = false; };
        }

        private void UpdateMaterialsDropdown()
        {
            if (cmbMaterial == null || cmbProcess == null) return;
            cmbMaterial.Items.Clear();
            string selectedProc = GetSelectedProcess();
            if (selectedProc == "injection_moulding")
            {
                cmbMaterial.Items.AddRange(new object[] { "ABS", "PA6", "PA66-GF30", "PP", "PC", "POM" });
            }
            else if (selectedProc == "die_cast_al")
            {
                cmbMaterial.Items.AddRange(new object[] { "Al-ADC12" });
            }
            else if (selectedProc == "die_cast_zn")
            {
                cmbMaterial.Items.AddRange(new object[] { "Zn-ZA8" });
            }
            else if (selectedProc == "die_cast_mg")
            {
                cmbMaterial.Items.AddRange(new object[] { "Mg-AZ91D" });
            }
            else
            {
                cmbMaterial.Items.AddRange(new object[] { "ABS" });
            }
            if (cmbMaterial.Items.Count > 0)
            {
                cmbMaterial.SelectedIndex = 0;
            }
        }

        private void ToggleConfigExpanded()
        {
            _configExpanded = !_configExpanded;
            UpdateConfigLayout();
        }

        private void UpdateConfigLayout()
        {
            pnlInput.SuspendLayout();
            if (_configExpanded)
            {
                lblConfigHeader.Text = "CONFIG  ▲";
                lblConfigSummary.Visible = false;
                
                lblWall.Visible = true;
                txtWallThickness.Visible = true;
                lblPullDirLabel.Visible = true;
                cmbPullDir.Visible = true;
                lblProcess.Visible = true;
                cmbProcess.Visible = true;
                lblMaterial.Visible = true;
                cmbMaterial.Visible = true;
                btnValidate.Visible = true;
                btnClear.Visible = true;
                
                pnlInput.Height = 150;
            }
            else
            {
                lblConfigHeader.Text = "CONFIG  ▼";
                
                string procName = cmbProcess.SelectedItem?.ToString() ?? "";
                if (procName.Contains("(")) procName = procName.Split('(')[0].Trim();
                string matName = cmbMaterial.SelectedItem?.ToString() ?? "";
                string wallText = txtWallThickness.Text;
                string pullDir = cmbPullDir.SelectedItem?.ToString() ?? "";
                
                lblConfigSummary.Text = $"{wallText}mm · {pullDir} · {procName} · {matName}";
                lblConfigSummary.Visible = true;
                
                lblWall.Visible = false;
                txtWallThickness.Visible = false;
                lblPullDirLabel.Visible = false;
                cmbPullDir.Visible = false;
                lblProcess.Visible = false;
                cmbProcess.Visible = false;
                lblMaterial.Visible = false;
                cmbMaterial.Visible = false;
                btnValidate.Visible = false;
                btnClear.Visible = false;
                
                pnlInput.Height = 24;
            }
            pnlInput.ResumeLayout(true);
            
            OnResize(EventArgs.Empty);
        }

        private void PnlGeminiBadge_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using (var pen = new Pen(Color.FromArgb(90, 75, 150), 1))
            using (var path = RoundedRect(new Rectangle(0, 0, pnlGeminiBadge.Width - 1, pnlGeminiBadge.Height - 1), 4))
            {
                e.Graphics.DrawPath(pen, path);
            }
        }

        private void InitializeLayout()
        {
            // Explicit layout rules
        }

        private void InitializeColors()
        {
            // Set any standard Control color properties if necessary
        }

        // Public API
        public void UpdateResults(ValidationResult result)
        {
            SafeInvoke(() =>
            {
                _lastResult = result;

                // Update score card and merged cells
                pnlScoreArc.Invalidate();
                pnlScore.Invalidate(true);

                // Show/hide review banner
                pnlReviewBanner.Visible = result.EngineerReviewRequired;

                // Show/hide Gemini badge
                int enrichedCount = 0;
                if (result.Violations != null)
                {
                    foreach (var v in result.Violations)
                    {
                        if (!string.IsNullOrEmpty(v.plain_english))
                        {
                            enrichedCount++;
                        }
                    }
                }

                if (result.gemini_enriched && enrichedCount > 0)
                {
                    lblGeminiBadge.Text = $"✦ Gemini AI: enriched {enrichedCount} findings";
                    pnlGeminiBadge.Visible = true;
                }
                else
                {
                    pnlGeminiBadge.Visible = false;
                }

                // AI analysis bar
                int riskVal = result.AiAnalysis.RiskBar;
                lblAiRiskVal.Text = riskVal.ToString();
                lblAiRiskVal.ForeColor = GetScoreColor(100 - riskVal);
                pnlAiProgressBar.Invalidate();
                lblAiModeConfidence.Text = $"{result.AiAnalysis.Mode} · {result.AiAnalysis.Confidence}% confidence";

                // Setup post-validation visibility
                pnlPostValidation.Visible = true;
                pnlAiAnalysis.Height = 130;
                lblFeedbackText.Visible = true;
                btnFeedbackAccurate.Visible = true;
                btnFeedbackOverride.Visible = true;
                btnFeedbackAccurate.Enabled = true;
                btnFeedbackOverride.Enabled = true;
                lblFeedbackStatus.Visible = false;
                lblFeedbackStatus.Location = new Point(10, 110);

                // Load violations
                pnlViolationsList.Controls.Clear();
                if (result.Violations != null && result.Violations.Count > 0)
                {
                    int index = 0;
                    foreach (var v in result.Violations)
                    {
                        var row = new ViolationRowPanel(v, this)
                        {
                            Width = pnlViolationsList.ClientSize.Width,
                            Location = new Point(0, index * 28) // height is now 28px!
                        };
                        pnlViolationsList.Controls.Add(row);
                        index++;
                    }
                }

                pnlViolations.Invalidate(); // Repaint header badge
                ReflowViolationRows();
                SetProgress("done");
            });
        }

        public void SetProgress(string step)
        {
            SafeInvoke(() =>
            {
                switch (step.ToLower())
                {
                    case "idle":
                        _currentStep = -1;
                        pnlStepper.Visible = false;
                        btnValidate.Enabled = true;
                        btnClear.Enabled = false;
                        pnlPostValidation.Visible = false;
                        _configExpanded = true;
                        UpdateConfigLayout();
                        break;
                    case "extracting":
                        _currentStep = 0;
                        _lastActiveStep = 0;
                        pnlStepper.Visible = true;
                        btnValidate.Enabled = false;
                        btnClear.Enabled = false;
                        pnlPostValidation.Visible = false;
                        _configExpanded = false;
                        UpdateConfigLayout();
                        break;
                    case "rules":
                        _currentStep = 1;
                        _lastActiveStep = 1;
                        pnlStepper.Visible = true;
                        btnValidate.Enabled = false;
                        btnClear.Enabled = false;
                        pnlPostValidation.Visible = false;
                        break;
                    case "gnn":
                        _currentStep = 2;
                        _lastActiveStep = 2;
                        pnlStepper.Visible = true;
                        btnValidate.Enabled = false;
                        btnClear.Enabled = false;
                        pnlPostValidation.Visible = false;
                        break;
                    case "done":
                        _currentStep = 3;
                        _lastActiveStep = 3;
                        pnlStepper.Visible = false;
                        btnValidate.Enabled = true;
                        btnClear.Enabled = true;
                        pnlPostValidation.Visible = true;
                        break;
                    case "error":
                        _currentStep = -2;
                        pnlStepper.Visible = false;
                        btnValidate.Enabled = true;
                        btnClear.Enabled = false;
                        pnlPostValidation.Visible = false;
                        _configExpanded = true;
                        UpdateConfigLayout();
                        break;
                }
                pnlStepper.Invalidate();
            });
        }

        public void ClearResults()
        {
            SafeInvoke(() =>
            {
                _lastResult = null;
                pnlViolationsList.Controls.Clear();
                SetProgress("idle");
                pnlReviewBanner.Visible = false;
                pnlGeminiBadge.Visible = false;
                lblValidationTime.Visible = false;
                pnlScoreArc.Invalidate();
                pnlAiProgressBar.Invalidate();
                pnlScore.Invalidate(true);
                pnlViolations.Invalidate();
                pnlAiAnalysis.Height = 64;
                lblFeedbackText.Visible = false;
                btnFeedbackAccurate.Visible = false;
                btnFeedbackOverride.Visible = false;
                lblFeedbackStatus.Visible = false;
                btnFeedbackAccurate.Enabled = true;
                btnFeedbackOverride.Enabled = true;
                lblFeedbackStatus.Location = new Point(10, 110);
                ReflowViolationRows();
            });
        }

        // Compatibility getters
        public string GetSelectedProcess()
        {
            if (cmbProcess == null || cmbProcess.SelectedItem == null) return "injection_moulding";
            switch (cmbProcess.SelectedItem.ToString())
            {
                case "Die Casting (Aluminium)": return "die_cast_al";
                case "Die Casting (Zinc)": return "die_cast_zn";
                case "Die Casting (Magnesium)": return "die_cast_mg";
                default: return "injection_moulding";
            }
        }
        public string GetSelectedMaterial() => cmbMaterial?.SelectedItem?.ToString() ?? "ABS";

        public void SetValidationTime(double seconds)
        {
            SafeInvoke(() =>
            {
                lblValidationTime.Text = string.Format("Validated in {0:F1}s", seconds);
                lblValidationTime.Visible = true;
            });
        }

        public void SetBackendConnected(bool connected)
        {
            SafeInvoke(() =>
            {
                _backendConnected = connected;
                pnlStatusDot.Invalidate();
            });
        }

        public void ShowFeedbackRecorded(string message)
        {
            SafeInvoke(() =>
            {
                lblFeedbackStatus.Text = message;
                lblFeedbackStatus.ForeColor = ColorGood;
                lblFeedbackStatus.Visible = true;

                // Hide feedback prompt and buttons
                lblFeedbackText.Visible = false;
                btnFeedbackAccurate.Visible = false;
                btnFeedbackOverride.Visible = false;

                // Position feedback status higher up and shrink card to fit
                lblFeedbackStatus.Location = new Point(10, 66);
                pnlAiAnalysis.Height = 88;

                // Reflow the panels to avoid blank spaces
                OnResize(EventArgs.Empty);
            });
        }

        public void ShowFeedbackError(string message)
        {
            SafeInvoke(() =>
            {
                lblFeedbackStatus.Text = message;
                lblFeedbackStatus.ForeColor = ColorCritical;
                lblFeedbackStatus.Visible = true;
                btnFeedbackAccurate.Enabled = true;
                btnFeedbackOverride.Enabled = true;
            });
        }

        public string GetClassification() => "structural";
        public double GetNominalWall()
        {
            if (double.TryParse(txtWallThickness.Text, out double val)) return val;
            return 3.0;
        }
        public string GetPullDirection() => cmbPullDir.SelectedItem?.ToString() ?? "auto";

        // Layout sizing adjustments
        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            if (flpRoot == null) return;
            
            int W = this.Width;
            flpRoot.Size = new Size(W, this.Height);
            
            int panelWidth = W - 20; // 10 padding on left and right
            
            // Resize panels
            if (pnlHeader != null) pnlHeader.Width = panelWidth;
            if (pnlInput != null)
            {
                pnlInput.Width = panelWidth;
                // Update controls inside pnlInput
                if (lblPullDirLabel != null) lblPullDirLabel.Left = Math.Min(126, panelWidth - 154);
                if (cmbPullDir != null)
                {
                    cmbPullDir.Left = Math.Min(210, panelWidth - 70);
                    cmbPullDir.Width = panelWidth - cmbPullDir.Left - 10;
                }
                if (cmbProcess != null) cmbProcess.Width = panelWidth - 86 - 10;
                if (cmbMaterial != null) cmbMaterial.Width = panelWidth - 86 - 10;
                if (btnClear != null) btnClear.Left = panelWidth - btnClear.Width - 10;
                if (btnValidate != null && btnClear != null) btnValidate.Width = btnClear.Left - btnValidate.Left - 8;
            }
            if (pnlStepper != null) pnlStepper.Width = panelWidth;
            if (pnlScore != null)
            {
                pnlScore.Width = panelWidth;
                if (pnlHeatmapCells != null)
                {
                    int scoreW = 80;
                    int availableW = panelWidth - scoreW - 12; // 6px padding on left & right
                    int cellW = availableW / 4;
                    for (int i = 0; i < pnlHeatmapCells.Length; i++)
                    {
                        if (pnlHeatmapCells[i] != null)
                        {
                            pnlHeatmapCells[i].Left = scoreW + 6 + i * cellW;
                            pnlHeatmapCells[i].Width = cellW - 4;
                            pnlHeatmapCells[i].Height = 52;
                            pnlHeatmapCells[i].Top = 12;
                        }
                    }
                }
            }
            if (pnlAiAnalysis != null)
            {
                pnlAiAnalysis.Width = panelWidth;
                if (lblAiRiskVal != null) lblAiRiskVal.Left = panelWidth - lblAiRiskVal.Width - 10;
                if (pnlAiProgressBar != null) pnlAiProgressBar.Width = panelWidth - 20;
                if (lblAiModeConfidence != null) lblAiModeConfidence.Width = panelWidth - 20;
                if (lblFeedbackText != null) lblFeedbackText.Width = panelWidth - 20;
                if (btnFeedbackAccurate != null && btnFeedbackOverride != null)
                {
                    int btnW = (panelWidth - 30) / 2;
                    btnFeedbackAccurate.Width = btnW;
                    btnFeedbackOverride.Left = 10 + btnW + 10;
                    btnFeedbackOverride.Width = btnW;
                }
                if (lblFeedbackStatus != null)
                {
                    lblFeedbackStatus.Width = panelWidth - 20;
                }
            }
            if (pnlPostValidation != null)
            {
                pnlPostValidation.Width = panelWidth;
                if (btnMarkClassA != null && btnExportReport != null)
                {
                    int btnW = (panelWidth - 30) / 2;
                    btnMarkClassA.Width = btnW;
                    btnExportReport.Left = 10 + btnW + 10;
                    btnExportReport.Width = btnW;
                }
            }
            if (pnlViolations != null) pnlViolations.Width = panelWidth;
            if (pnlViolationsHeader != null) pnlViolationsHeader.Width = panelWidth;
            if (lblFaceHighlight != null) lblFaceHighlight.Width = panelWidth;
            if (pnlViolationsList != null) pnlViolationsList.Width = panelWidth;

            // Recalculate remaining height for violations
            int usedHeight = 0;
            foreach (Control c in flpRoot.Controls)
            {
                if (c != pnlViolations && c.Visible)
                {
                    usedHeight += c.Height + c.Margin.Top + c.Margin.Bottom;
                }
            }
            int remaining = this.Height - usedHeight - pnlViolations.Margin.Top - pnlViolations.Margin.Bottom;
            pnlViolations.Height = Math.Max(150, remaining);
            pnlViolationsList.Height = pnlViolations.Height - 42;

            ReflowViolationRows();
        }

        // Reflow rows in scroll container
        public void ReflowViolationRows()
        {
            if (this.InvokeRequired)
            {
                this.Invoke((Action)(() => ReflowViolationRows()));
                return;
            }

            pnlViolationsList.SuspendLayout();
            int y = 0;
            foreach (Control c in pnlViolationsList.Controls)
            {
                if (c.Visible)
                {
                    c.Top = y;
                    c.Width = pnlViolationsList.ClientSize.Width;
                    y += c.Height;
                }
            }
            pnlViolationsList.AutoScrollMinSize = new Size(0, y);
            pnlViolationsList.ResumeLayout(true);
        }

        // Row interaction callbacks
        public void OnViolationRowClicked(ViolationRowPanel clickedRow)
        {
            pnlViolationsList.SuspendLayout();
            foreach (Control c in pnlViolationsList.Controls)
            {
                if (c is ViolationRowPanel row && row != clickedRow && row.IsExpanded)
                {
                    row.Collapse();
                }
            }

            clickedRow.ToggleExpand();
            if (clickedRow.IsExpanded)
            {
                FaceSelected?.Invoke(this, clickedRow.Data.FaceId);
                // Show viewport highlight confirmation label at top of violations section
                lblFaceHighlight.Text = $"↗ Face {clickedRow.Data.FaceId} highlighted in viewport";
                lblFaceHighlight.Visible = true;
                tmrFaceHighlight.Stop();
                tmrFaceHighlight.Start();
            }
            else
            {
                FaceSelected?.Invoke(this, "");
                lblFaceHighlight.Visible = false;
                tmrFaceHighlight.Stop();
            }
            pnlViolationsList.ResumeLayout(true);
        }

        // Private custom paint events
        private void PnlStatusDot_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            Color dotColor = _backendConnected ? ColorGood : ColorCritical;

            if (_currentStep >= 0 && _currentStep < 3)
            {
                dotColor = AccentBlue;
            }
            else if (_currentStep == -2)
            {
                dotColor = ColorCritical;
            }

            using (var brush = new SolidBrush(dotColor))
            {
                e.Graphics.FillEllipse(brush, 0, 0, 10, 10);
            }
        }

        private void PnlStepper_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            e.Graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

            string[] steps = { "Extract", "Rules", "GNN", "Done" };
            int stepCount = steps.Length;
            int padding = 16;
            int stepWidth = (pnlStepper.Width - padding * 2) / (stepCount - 1);
            int y = 16;
            int r = 7;

            // Draw line segment connectors
            for (int i = 0; i < stepCount - 1; i++)
            {
                int x1 = padding + i * stepWidth;
                int x2 = padding + (i + 1) * stepWidth;
                bool completed = (_currentStep != -2) && (i < _currentStep);

                using (var pen = new Pen(completed ? ColorGood : BorderColor, 2))
                {
                    e.Graphics.DrawLine(pen, x1 + r, y, x2 - r, y);
                }
            }

            // Draw steps
            for (int i = 0; i < stepCount; i++)
            {
                int cx = padding + i * stepWidth;
                int state = 0; // 0 = pending, 1 = active, 2 = complete, 3 = error

                if (_currentStep == -2)
                {
                    state = (i <= _lastActiveStep) ? 3 : 0;
                }
                else
                {
                    if (i < _currentStep) state = 2;
                    else if (i == _currentStep) state = 1;
                }

                // Fill circle background
                if (state == 0) // pending
                {
                    using (var brush = new SolidBrush(BgRaised))
                    using (var pen = new Pen(BorderColor, 1.5f))
                    {
                        e.Graphics.FillEllipse(brush, cx - r, y - r, r * 2, r * 2);
                        e.Graphics.DrawEllipse(pen, cx - r, y - r, r * 2, r * 2);
                    }
                }
                else if (state == 1) // active
                {
                    using (var brush = new SolidBrush(AccentBlue))
                    {
                        e.Graphics.FillEllipse(brush, cx - r, y - r, r * 2, r * 2);
                    }
                }
                else if (state == 2) // complete
                {
                    using (var brush = new SolidBrush(ColorGood))
                    {
                        e.Graphics.FillEllipse(brush, cx - r, y - r, r * 2, r * 2);
                    }
                    using (var font = new Font("Segoe UI", 7.5f, FontStyle.Bold))
                    using (var brush = new SolidBrush(Color.White))
                    using (var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
                    {
                        e.Graphics.DrawString("✓", font, brush, new RectangleF(cx - r, y - r + 0.5f, r * 2, r * 2), sf);
                    }
                }
                else if (state == 3) // error
                {
                    using (var brush = new SolidBrush(ColorCritical))
                    {
                        e.Graphics.FillEllipse(brush, cx - r, y - r, r * 2, r * 2);
                    }
                    using (var font = new Font("Segoe UI", 7.5f, FontStyle.Bold))
                    using (var brush = new SolidBrush(Color.White))
                    using (var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
                    {
                        e.Graphics.DrawString("✗", font, brush, new RectangleF(cx - r, y - r, r * 2, r * 2), sf);
                    }
                }

                // Labels
                using (var font = new Font("Segoe UI", 8.5f, state == 1 ? FontStyle.Bold : FontStyle.Regular))
                using (var brush = new SolidBrush(state == 1 ? TextPrimary : TextSecondary))
                using (var sf = new StringFormat { Alignment = StringAlignment.Center })
                {
                    e.Graphics.DrawString(steps[i], font, brush, cx, y + r + 2, sf);
                }
            }
        }

        private void PnlScoreArc_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            e.Graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

            int score = _lastResult != null ? _lastResult.Score : 0;
            string riskLevel = _lastResult != null ? _lastResult.RiskLevel : "NO DATA";
            Color scoreCol = GetScoreColor(score);

            // Centered Score number
            using (var fontScore = new Font("Segoe UI", 24f, FontStyle.Bold))
            using (var brushScore = new SolidBrush(scoreCol))
            using (var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
            {
                e.Graphics.DrawString(score.ToString(), fontScore, brushScore, new RectangleF(0, 10, 80, 32), sf);
            }

            // Risk Level tier label below the score
            using (var fontRisk = new Font("Segoe UI", 8f, FontStyle.Bold))
            using (var brushRisk = new SolidBrush(TextSecondary))
            using (var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
            {
                e.Graphics.DrawString(riskLevel, fontRisk, brushRisk, new RectangleF(0, 42, 80, 20), sf);
            }
        }

        private void PnlReviewBanner_Paint(object sender, PaintEventArgs e)
        {
            using (var brush = new SolidBrush(ColorWarning))
            {
                e.Graphics.FillRectangle(brush, 0, 0, 3, pnlReviewBanner.Height);
            }
        }

        private void PnlAiProgressBar_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using (var brushBg = new SolidBrush(BgRaised))
            {
                e.Graphics.FillRectangle(brushBg, pnlAiProgressBar.ClientRectangle);
            }

            int riskBar = _lastResult != null ? _lastResult.AiAnalysis.RiskBar : 0;
            if (riskBar > 0)
            {
                // Colors: high risk = red, low risk = green
                Color riskCol = GetScoreColor(100 - riskBar);
                float fillW = (riskBar / 100f) * pnlAiProgressBar.Width;
                using (var brushFill = new SolidBrush(riskCol))
                {
                    e.Graphics.FillRectangle(brushFill, 0, 0, fillW, pnlAiProgressBar.Height);
                }
            }
        }

        private void PnlViolationsHeader_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            e.Graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

            // Header title
            using (var font = new Font("Segoe UI", 9f, FontStyle.Bold))
            using (var brush = new SolidBrush(TextPrimary))
            {
                e.Graphics.DrawString("VIOLATIONS", font, brush, 0, 4);
            }

            // Count badge
            int vCount = _lastResult != null && _lastResult.Violations != null ? _lastResult.Violations.Count : 0;
            if (vCount > 0)
            {
                Rectangle badgeRect = new Rectangle(pnlViolationsHeader.Width - 30, 3, 24, 16);
                using (var path = RoundedRect(badgeRect, 4))
                using (var brushBadge = new SolidBrush(AccentBlue))
                {
                    e.Graphics.FillPath(brushBadge, path);
                }

                using (var fontBadge = new Font("Segoe UI", 8f, FontStyle.Bold))
                using (var brushText = new SolidBrush(Color.White))
                using (var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
                {
                    e.Graphics.DrawString(vCount.ToString(), fontBadge, brushText, badgeRect, sf);
                }
            }
        }

        private void CreateHeatmapCells()
        {
            pnlHeatmapCells = new Panel[4];
            string[] labels = { "Critical", "At Risk", "Watch", "Good" };
            Color[] colors = { ColorCritical, ColorWarning, ColorInfo, ColorGood };

            for (int i = 0; i < 4; i++)
            {
                int idx = i;
                var cell = new Panel
                {
                    Location = new Point(80 + 6 + i * 44, 12),
                    Size = new Size(43, 52),
                    BackColor = BgRaised
                };
                cell.Paint += (s, e) =>
                {
                    e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
                    e.Graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

                    Color cellColor = colors[idx];
                    int count = 0;

                    if (_lastResult != null)
                    {
                        if (idx == 0) count = _lastResult.Heatmap.Critical;
                        else if (idx == 1) count = _lastResult.Heatmap.AtRisk;
                        else if (idx == 2) count = _lastResult.Heatmap.Watch;
                        else if (idx == 3) count = _lastResult.Heatmap.Good;
                    }

                    // Top border segment line
                    using (var brushBorder = new SolidBrush(cellColor))
                    {
                        e.Graphics.FillRectangle(brushBorder, 0, 0, cell.Width, 3);
                    }

                    // Count
                    Color textCol = (count == 0) ? TextSecondary : cellColor;
                    using (var fontCount = new Font("Segoe UI", 12f, FontStyle.Bold))
                    using (var brushCount = new SolidBrush(textCol))
                    using (var sf = new StringFormat { Alignment = StringAlignment.Center })
                    {
                        e.Graphics.DrawString(count.ToString(), fontCount, brushCount, cell.Width / 2f, 8, sf);
                    }

                    // Sub label
                    using (var fontLabel = new Font("Segoe UI", 7f))
                    using (var brushLabel = new SolidBrush(TextSecondary))
                    using (var sf = new StringFormat { Alignment = StringAlignment.Center })
                    {
                        e.Graphics.DrawString(labels[idx], fontLabel, brushLabel, cell.Width / 2f, 28, sf);
                    }
                };
                pnlScore.Controls.Add(cell);
                pnlHeatmapCells[i] = cell;
            }
        }

        // Helper/utility methods
        public static GraphicsPath RoundedRect(Rectangle bounds, int radius)
        {
            var path = new GraphicsPath();
            int diameter = radius * 2;
            path.AddArc(bounds.X, bounds.Y, diameter, diameter, 180, 90);
            path.AddArc(bounds.Right - diameter, bounds.Y, diameter, diameter, 270, 90);
            path.AddArc(bounds.Right - diameter, bounds.Bottom - diameter, diameter, diameter, 0, 90);
            path.AddArc(bounds.X, bounds.Bottom - diameter, diameter, diameter, 90, 90);
            path.CloseFigure();
            return path;
        }

        public static Color GetSeverityColor(string severity)
        {
            switch ((severity ?? "").ToUpper())
            {
                case "CRITICAL": return ColorCritical;
                case "WARNING":  return ColorWarning;
                case "INFO":     return ColorInfo;
                default:         return TextSecondary;
            }
        }

        public static string GetSeverityBadge(string severity)
        {
            switch ((severity ?? "").ToUpper())
            {
                case "CRITICAL": return "CRIT";
                case "WARNING":  return "WARN";
                case "INFO":     return "INFO";
                default:         return "???";
            }
        }

        public static Color GetScoreColor(int score)
        {
            if (score >= 75) return ColorGood;
            if (score >= 50) return ColorWarning;
            return ColorCritical;
        }

        private void SafeInvoke(Action action)
        {
            if (this.InvokeRequired) this.Invoke(action);
            else action();
        }
    }

    // Violation Row Custom Control
    public class ViolationRowPanel : Panel
    {
        public Violation Data { get; set; }
        public bool IsExpanded { get; set; }

        private readonly TaskPane _owner;

        // Sub components for expansion
        public Panel pnlCallout;
        public Panel pnlFix;
        public Label lblFixText;
        public Button btnFix;

        private System.Windows.Forms.Timer tmrFix;
        private bool isFixExpanded = false;

        public int FixPanelHeight => pnlFix != null ? pnlFix.Height : 0;

        public ViolationRowPanel(Violation data, TaskPane owner)
        {
            this.Data = data;
            this._owner = owner;
            this.IsExpanded = false;
            this.DoubleBuffered = true;
            this.Height = 28;
            this.BackColor = TaskPane.BgSurface;

            InitializeRowLayout();
            WireEvents(this);
        }

        private void InitializeRowLayout()
        {
            // Callout Card (deep blue, border 1px AccentBlue rounded rect)
            pnlCallout = new Panel
            {
                Location = new Point(8, 32),
                Size = new Size(264, 110),
                BackColor = Color.FromArgb(26, 35, 50),
                Visible = false
            };
            pnlCallout.Paint += PnlCallout_Paint;

            // Details inside Callout
            var lblFace = new Label
            {
                Text = "Face " + Data.FaceId,
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                ForeColor = TaskPane.TextPrimary,
                Location = new Point(10, 8),
                AutoSize = true
            };

            btnFix = new Button
            {
                Text = "How to Fix ↗",
                Font = new Font("Segoe UI", 9f),
                FlatStyle = FlatStyle.Flat,
                Location = new Point(154, 78),
                Size = new Size(100, 24),
                BackColor = Color.FromArgb(26, 35, 50),
                ForeColor = TaskPane.AccentBlue,
                Cursor = Cursors.Hand
            };
            btnFix.FlatAppearance.BorderSize = 0;
            btnFix.Click += BtnFix_Click;

            var pnlDivider = new Panel
            {
                Location = new Point(10, 74),
                Size = new Size(244, 1),
                BackColor = TaskPane.BorderColor
            };

            if (Data.Source == "GNN")
            {
                var lblMessage = new Label
                {
                    Text = Data.Message,
                    Font = new Font("Segoe UI", 8.5f),
                    ForeColor = TaskPane.TextPrimary,
                    Location = new Point(10, 26),
                    Size = new Size(244, 46),
                    MaximumSize = new Size(244, 46),
                    AutoEllipsis = true
                };
                var lblSource = new Label
                {
                    Text = "Source: GNN",
                    Font = new Font("Segoe UI", 8f, FontStyle.Italic),
                    ForeColor = TaskPane.TextSecondary,
                    Location = new Point(10, 80),
                    AutoSize = true
                };

                pnlCallout.Controls.AddRange(new Control[] { lblFace, lblMessage, pnlDivider, lblSource, btnFix });
            }
            else
            {
                double reqNum = Data.ParseNumericValue(Data.required_value, double.NaN);
                double measNum = Data.ParseNumericValue(Data.measured_value, double.NaN);
                
                string measStr = double.IsNaN(measNum) ? Data.measured_value : string.Format("{0:F2}{1}", Data.MeasuredValue, Data.Unit);
                string reqStr = double.IsNaN(reqNum) ? Data.required_value : string.Format("{0} {1:F2}{2}", Data.Relation, Data.RequiredValue, Data.Unit);

                var lblMeasured = new Label
                {
                    Text = "Measured     " + measStr,
                    Font = new Font("Segoe UI", 9f),
                    ForeColor = TaskPane.TextPrimary,
                    Location = new Point(10, 24),
                    AutoSize = true
                };
                var lblRequired = new Label
                {
                    Text = "Required      " + reqStr,
                    Font = new Font("Segoe UI", 9f),
                    ForeColor = TaskPane.TextPrimary,
                    Location = new Point(10, 40),
                    AutoSize = true
                };

                double delta = Data.Relation == "≤" ? Data.MeasuredValue - Data.RequiredValue : Data.RequiredValue - Data.MeasuredValue;
                bool hasDelta = !double.IsNaN(reqNum) && !double.IsNaN(measNum);

                string deltaText;
                if (delta > 0)
                {
                    deltaText = string.Format("Fix delta      {0}{1:F2}{2}", Data.Relation == "≤" ? "-" : "+", delta, Data.Unit);
                }
                else
                {
                    deltaText = "Fix delta      (OK)";
                }

                var lblDelta = new Label
                {
                    Text = deltaText,
                    Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                    ForeColor = delta > 0 ? TaskPane.ColorWarning : TaskPane.ColorGood,
                    Location = new Point(10, 56),
                    AutoSize = true,
                    Visible = hasDelta
                };
                var lblSource = new Label
                {
                    Text = "Source: " + Data.Source,
                    Font = new Font("Segoe UI", 8f, FontStyle.Italic),
                    ForeColor = TaskPane.TextSecondary,
                    Location = new Point(10, 80),
                    AutoSize = true
                };

                pnlCallout.Controls.AddRange(new Control[] { lblFace, lblMeasured, lblRequired, lblDelta, pnlDivider, lblSource, btnFix });
            }
            this.Controls.Add(pnlCallout);

            // Suggestion Panel (below callout card)
            pnlFix = new Panel
            {
                Location = new Point(8, 146),
                Size = new Size(264, 0),
                BackColor = Color.FromArgb(26, 35, 50),
                Visible = false
            };
            pnlFix.Paint += PnlFix_Paint;

            lblFixText = new Label
            {
                Text = Data.FixSuggestion,
                Font = new Font("Segoe UI", 9f, FontStyle.Italic),
                ForeColor = TaskPane.TextSecondary,
                Location = new Point(10, 8),
                MaximumSize = new Size(244, 1000),
                AutoSize = true
            };
            pnlFix.Controls.Add(lblFixText);
            this.Controls.Add(pnlFix);

            // Timer
            tmrFix = new System.Windows.Forms.Timer { Interval = 15 };
            tmrFix.Tick += TmrFix_Tick;
        }

        private void WireEvents(Control parent)
        {
            foreach (Control c in parent.Controls)
            {
                if (c == btnFix || c == pnlCallout || c == pnlFix || c.Parent == pnlCallout || c.Parent == pnlFix)
                    continue; // Skip interactive or deep inner parts from triggering expand click
                c.MouseEnter += (s, e) => Row_MouseEnter();
                c.MouseLeave += (s, e) => Row_MouseLeave();
                c.Click += (s, e) => Row_Click();
                WireEvents(c);
            }
            parent.MouseEnter += (s, e) => Row_MouseEnter();
            parent.MouseLeave += (s, e) => Row_MouseLeave();
            parent.Click += (s, e) => Row_Click();
        }

        private void Row_MouseEnter()
        {
            this.BackColor = TaskPane.BgRaised;
            this.Cursor = Cursors.Hand;
            this.Invalidate();
        }

        private void Row_MouseLeave()
        {
            this.BackColor = TaskPane.BgSurface;
            this.Invalidate();
        }

        private void Row_Click()
        {
            _owner.OnViolationRowClicked(this);
        }

        public void Expand()
        {
            IsExpanded = true;
            this.Height = 146 + pnlFix.Height;
            pnlCallout.Visible = true;
            this.Invalidate();
        }

        public void Collapse()
        {
            IsExpanded = false;
            isFixExpanded = false;
            pnlFix.Height = 0;
            pnlFix.Visible = false;
            pnlCallout.Visible = false;
            btnFix.Text = "How to Fix ↗";
            this.Height = 28;
            this.Invalidate();
        }

        public void ToggleExpand()
        {
            if (IsExpanded) Collapse();
            else Expand();
        }

        private void BtnFix_Click(object sender, EventArgs e)
        {
            isFixExpanded = !isFixExpanded;
            tmrFix.Start();
        }

        private void TmrFix_Tick(object sender, EventArgs e)
        {
            _owner.SuspendLayout();
            this.SuspendLayout();
            if (isFixExpanded)
            {
                pnlFix.Visible = true;
                int target = lblFixText.PreferredHeight + 16;
                if (pnlFix.Height < target)
                {
                    pnlFix.Height = Math.Min(target, pnlFix.Height + 12);
                    this.Height = 146 + pnlFix.Height;
                    _owner.ReflowViolationRows();
                }
                else
                {
                    pnlFix.Height = target;
                    this.Height = 146 + pnlFix.Height;
                    tmrFix.Stop();
                    btnFix.Text = "How to Fix ↙";
                    _owner.ReflowViolationRows();
                }
            }
            else
            {
                if (pnlFix.Height > 0)
                {
                    pnlFix.Height = Math.Max(0, pnlFix.Height - 12);
                    this.Height = 146 + pnlFix.Height;
                    _owner.ReflowViolationRows();
                }
                else
                {
                    pnlFix.Height = 0;
                    pnlFix.Visible = false;
                    this.Height = 146;
                    tmrFix.Stop();
                    btnFix.Text = "How to Fix ↗";
                    _owner.ReflowViolationRows();
                }
            }
            this.ResumeLayout(true);
            _owner.ResumeLayout(true);
        }

        private void PnlCallout_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using (var pen = new Pen(TaskPane.AccentBlue, 1))
            using (var path = TaskPane.RoundedRect(new Rectangle(0, 0, pnlCallout.Width - 1, pnlCallout.Height - 1), 6))
            {
                e.Graphics.DrawPath(pen, path);
            }
        }

        private void PnlFix_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using (var pen = new Pen(TaskPane.AccentBlue, 1))
            using (var path = TaskPane.RoundedRect(new Rectangle(0, 0, pnlFix.Width - 1, pnlFix.Height - 1), 6))
            {
                e.Graphics.DrawPath(pen, path);
            }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            e.Graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

            Color sevCol = TaskPane.GetSeverityColor(Data.Severity);

            // 1. Left severity border bar
            using (var brushBar = new SolidBrush(sevCol))
            {
                e.Graphics.FillRectangle(brushBar, 0, 0, 3, this.Height);
            }

            // 2. Severity Badge Pill at X=10, Y=7
            Rectangle badgeRect = new Rectangle(10, 7, 28, 14);
            using (var pathBadge = TaskPane.RoundedRect(badgeRect, 3))
            using (var brushBadge = new SolidBrush(sevCol))
            {
                e.Graphics.FillPath(brushBadge, pathBadge);
            }

            using (var fontBadge = new Font("Segoe UI", 7f, FontStyle.Bold))
            using (var brushWhite = new SolidBrush(Color.White))
            using (var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
            {
                e.Graphics.DrawString(TaskPane.GetSeverityBadge(Data.Severity), fontBadge, brushWhite, badgeRect, sf);
            }

            // 3. Rule ID at X=44, Y=5
            float ruleWidth = 0;
            using (var fontRule = new Font("Segoe UI", 9.5f, FontStyle.Bold))
            using (var brushRule = new SolidBrush(sevCol))
            {
                e.Graphics.DrawString(Data.Id, fontRule, brushRule, 44, 5);
                ruleWidth = e.Graphics.MeasureString(Data.Id, fontRule).Width;
            }

            // 4. Face ID and Values drawing logic
            if (!IsExpanded)
            {
                // Collapsed single line format
                using (var fontFace = new Font("Segoe UI", 8.5f))
                using (var brushSec = new SolidBrush(TaskPane.TextSecondary))
                {
                    float faceX = 44 + ruleWidth + 8;
                    e.Graphics.DrawString("F" + Data.FaceId, fontFace, brushSec, faceX, 6);
                }

                if (Data.Source == "GNN")
                {
                    // Draw risk score for GNN checks (e.g., "63% Risk")
                    string riskPercent = "";
                    string desc = Data.description ?? "";
                    int riskIdx = desc.IndexOf("risk ");
                    if (riskIdx >= 0)
                    {
                        int pctIdx = desc.IndexOf("%", riskIdx);
                        if (pctIdx > riskIdx)
                        {
                            riskPercent = desc.Substring(riskIdx + 5, pctIdx - riskIdx - 5 + 1);
                        }
                    }
                    if (string.IsNullOrEmpty(riskPercent))
                    {
                        riskPercent = "Risk";
                    }
                    else
                    {
                        riskPercent = riskPercent + " Risk";
                    }

                    Font fontBold = new Font("Segoe UI", 8.5f, FontStyle.Bold);
                    float wRisk = e.Graphics.MeasureString(riskPercent, fontBold).Width;
                    float startX = this.Width - 8 - wRisk;
                    using (var brushMeas = new SolidBrush(sevCol))
                    {
                        e.Graphics.DrawString(riskPercent, fontBold, brushMeas, startX, 6);
                    }
                }
                else
                {
                    double reqNum = Data.ParseNumericValue(Data.required_value, double.NaN);
                    double measNum = Data.ParseNumericValue(Data.measured_value, double.NaN);
                    
                    string measText = double.IsNaN(measNum) ? Data.measured_value : Data.MeasuredValue.ToString("F2") + Data.Unit;
                    string arrowText = " " + Data.Relation + " ";
                    string reqText = double.IsNaN(reqNum) ? Data.required_value : Data.RequiredValue.ToString("F2") + Data.Unit;
                    
                    if (double.IsNaN(reqNum))
                    {
                        arrowText = " ";
                        reqText = Data.required_value;
                    }

                    Font fontBold = new Font("Segoe UI", 8.5f, FontStyle.Bold);
                    Font fontRegular = new Font("Segoe UI", 8.5f);

                    float wMeas = e.Graphics.MeasureString(measText, fontBold).Width;
                    float wArrow = e.Graphics.MeasureString(arrowText, fontRegular).Width;
                    float wReq = e.Graphics.MeasureString(reqText, fontRegular).Width;
                    float totalW = wMeas + wArrow + wReq;

                    float startX = this.Width - 8 - totalW;

                    using (var brushMeas = new SolidBrush(sevCol))
                    {
                        e.Graphics.DrawString(measText, fontBold, brushMeas, startX, 6);
                    }
                    using (var brushSec = new SolidBrush(TaskPane.TextSecondary))
                    {
                        e.Graphics.DrawString(arrowText, fontRegular, brushSec, startX + wMeas, 6);
                    }
                    using (var brushPrim = new SolidBrush(TaskPane.TextPrimary))
                    {
                        e.Graphics.DrawString(reqText, fontRegular, brushPrim, startX + wMeas + wArrow, 6);
                    }
                }
            }
            else
            {
                // Expanded format
                using (var fontFace = new Font("Segoe UI", 8.5f))
                using (var brushSec = new SolidBrush(TaskPane.TextSecondary))
                using (var sf = new StringFormat { Alignment = StringAlignment.Far })
                {
                    e.Graphics.DrawString("Face " + Data.FaceId, fontFace, brushSec, this.Width - 8, 8, sf);
                }

                if (Data.Source == "GNN")
                {
                    string riskPercent = "";
                    string desc = Data.description ?? "";
                    int riskIdx = desc.IndexOf("risk ");
                    if (riskIdx >= 0)
                    {
                        int pctIdx = desc.IndexOf("%", riskIdx);
                        if (pctIdx > riskIdx)
                        {
                            riskPercent = desc.Substring(riskIdx + 5, pctIdx - riskIdx - 5 + 1) + " Risk Score";
                        }
                    }
                    if (string.IsNullOrEmpty(riskPercent))
                    {
                        riskPercent = "Elevated Risk";
                    }

                    using (var fontMeas = new Font("Segoe UI", 10.5f, FontStyle.Bold))
                    using (var brushMeas = new SolidBrush(sevCol))
                    {
                        e.Graphics.DrawString(riskPercent, fontMeas, brushMeas, 12, 26);
                    }
                }
                else
                {
                    double reqNum = Data.ParseNumericValue(Data.required_value, double.NaN);
                    double measNum = Data.ParseNumericValue(Data.measured_value, double.NaN);
                    
                    string measText = double.IsNaN(measNum) ? Data.measured_value : Data.MeasuredValue.ToString("F2") + Data.Unit;
                    string arrowText = " " + Data.Relation + " ";
                    string reqText = double.IsNaN(reqNum) ? Data.required_value : Data.RequiredValue.ToString("F2") + Data.Unit;

                    if (double.IsNaN(reqNum))
                    {
                        arrowText = " ";
                        reqText = Data.required_value;
                    }

                    float currentX = 12;
                    using (var fontMeas = new Font("Segoe UI", 10.5f, FontStyle.Bold))
                    using (var brushMeas = new SolidBrush(sevCol))
                    {
                        e.Graphics.DrawString(measText, fontMeas, brushMeas, currentX, 26);
                        currentX += e.Graphics.MeasureString(measText, fontMeas).Width + 2;
                    }

                    using (var fontArrow = new Font("Segoe UI", 9.5f))
                    using (var brushSec = new SolidBrush(TaskPane.TextSecondary))
                    {
                        e.Graphics.DrawString(arrowText, fontArrow, brushSec, currentX, 27);
                        currentX += e.Graphics.MeasureString(arrowText, fontArrow).Width + 2;
                    }

                    float reqWidth = 0;
                    using (var fontReq = new Font("Segoe UI", 9.5f))
                    using (var brushPrim = new SolidBrush(TaskPane.TextPrimary))
                    {
                        e.Graphics.DrawString(reqText, fontReq, brushPrim, currentX, 27);
                        reqWidth = e.Graphics.MeasureString(reqText, fontReq).Width;
                    }

                    if (!double.IsNaN(reqNum) && !double.IsNaN(measNum))
                    {
                        double delta = Data.Relation == "≤" ? Data.MeasuredValue - Data.RequiredValue : Data.RequiredValue - Data.MeasuredValue;
                        string deltaText = delta > 0 ? "(" + (Data.Relation == "≤" ? "-" : "+") + delta.ToString("F2") + Data.Unit + " needed)" : "(OK)";
                        Color deltaColor = delta > 0 ? TaskPane.ColorWarning : TaskPane.ColorGood;

                        using (var fontDelta = new Font("Segoe UI", 8.5f))
                        using (var brushDelta = new SolidBrush(deltaColor))
                        using (var sf = new StringFormat { Alignment = StringAlignment.Far })
                        {
                            float deltaWidth = e.Graphics.MeasureString(deltaText, fontDelta).Width;

                            float reqRightEdge = currentX + reqWidth;
                            float deltaLeftEdge = (this.Width - 8) - deltaWidth;

                            float drawY = 28;
                            if (deltaLeftEdge < reqRightEdge + 6)
                            {
                                drawY = 38;
                            }

                            e.Graphics.DrawString(deltaText, fontDelta, brushDelta, this.Width - 8, drawY, sf);
                        }
                    }
                }
            }

            // 7. If expanded, draw the AccentBlue triangle pointer pointing up to Callout at Y=32
            if (IsExpanded)
            {
                Point[] points = {
                    new Point(this.Width / 2 - 6, 32),
                    new Point(this.Width / 2 + 6, 32),
                    new Point(this.Width / 2, 25)
                };
                using (var brushTriangle = new SolidBrush(TaskPane.AccentBlue))
                {
                    e.Graphics.FillPolygon(brushTriangle, points);
                }
            }
        }
    }
}
