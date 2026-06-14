from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..core.models import ValidationResult
from pydantic import BaseModel
from typing import Optional, Dict, List, Any

class ReportPayload(BaseModel):
    result:          ValidationResult
    face_snapshots:  Optional[Dict[str, str]] = {}
    faces_geometry:  Optional[List[dict]]     = []

from ..rules.engine import PROCESS_INFO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
import html
import requests
import json
import base64
import os
import logging
import time

import asyncio

logger = logging.getLogger(__name__)

from ..utils.gemini_utils import call_gemini_with_timeout

def generate_gemini_report_narrative(result: ValidationResult) -> str:
    # Check cache first
    cached_narrative = getattr(result, "gemini_narrative", "")
    if cached_narrative:
        logger.info("Using cached Gemini report narrative.")
        return cached_narrative

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — skipping Gemini report narrative.")
        return ""
        
    violations_summary = []
    for v in (result.violations or []):
        violations_summary.append({
            "rule_id": v.rule_id,
            "severity": getattr(v.severity, "value", str(v.severity)),
            "description": v.description,
            "measured": v.measured_value,
            "required": v.required_value,
            "fix": v.fix_suggestion
        })
        
    prompt = (
        "You are a Principal Manufacturing Engineer (DFM Specialist) at Varroc. "
        "Review the following DFM validation results and write a highly professional, "
        "technical Executive Summary for an engineering report.\n\n"
        f"Part ID: {result.part_id}\n"
        f"Process: {result.process}\n"
        f"Material: {result.material}\n"
        f"DFM Score: {result.overall_manufacturability_score}/100\n"
        f"GNN Anomaly Score: {result.gnn_risk_score:.2f}\n"
        f"Violations:\n{json.dumps(violations_summary, indent=2)}\n\n"
        "Instructions:\n"
        "- Write a 3-paragraph, highly technical engineering narrative.\n"
        "- Paragraph 1: Executive Summary. State the overall manufacturability, DFM score, process/material appropriateness, and tooling readiness.\n"
        "- Paragraph 2: Technical Breakdown of Major Risks. Analyze the critical and warning violations (e.g., thickness deviations, low draft, sharp corners), explaining the physical consequences (e.g., cold shuts, core sticking, stress concentration) in automotive application contexts.\n"
        "- Paragraph 3: Recommended Corrective Action Plan. Detail the exact design modifications required to clear the tool for production.\n"
        "- Do not use generic filler. Be specific to the process and material (e.g. Al-ADC12 die casting or ABS moulding).\n"
        "- Use professional engineering tone (ASME/NADCA reference style).\n"
        "Output ONLY the text. No markdown, no HTML, no introductory text."
    )
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
        },
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=8.0)
        response.raise_for_status()
        resp_json = response.json()
        raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        return raw_text
    except Exception as e:
        logger.error(f"Gemini narrative generation failed: {e}")
        return ""

router = APIRouter()

def get_score_label(score: int) -> str:
    if score >= 80: return "Good — minor recommendations"
    if score >= 60: return "Fair — design changes needed"
    if score >= 40: return "Poor — significant redesign required"
    return "Unacceptable — must redesign before tooling"

@router.post("")
async def generate_report(result: ValidationResult):
    c = result.risk_summary
    score = result.overall_manufacturability_score

    lines = []
    lines.append("## Executive Summary")
    lines.append(f"Eureka DFM validation completed for part **{result.part_id}**. Overall manufacturability score: **{score}/100** — {get_score_label(score)}.")
    lines.append(f"Found {c.get('critical_count',0)} critical, {c.get('warning_count',0)} warning, and {c.get('info_count',0)} informational issues.")
    if result.gnn_risk_score > 0.7:
        lines.append(f"**GNN anomaly detected:** risk score {result.gnn_risk_score:.2f} — multi-feature interaction identified.")
    if result.engineer_review_required:
        lines.append("**Engineer review required** before proceeding to tooling.")
    lines.append("")

    # Engineering Narrative from Gemini
    narrative = await call_gemini_with_timeout(
        generate_gemini_report_narrative,
        result,
        timeout_seconds=8.0,
        fallback="Executive summary unavailable."
    )
    if narrative:
        lines.append("## Principal Engineering DFM Assessment")
        lines.append(narrative)
        lines.append("")

    gnn_anom = getattr(result, "gnn_anomaly", None)
    if isinstance(gnn_anom, dict) and gnn_anom.get("gemini_explanation"):
        lines.append("### AI Anomaly Diagnosis")
        lines.append(gnn_anom["gemini_explanation"])
        lines.append("")

    if result.violations:
        critical_v = [v for v in result.violations if getattr(v.severity, "value", str(v.severity)).upper() == "CRITICAL"]
        if critical_v:
            lines.append("## Critical Findings (Pinpointed Features)")
            for i, v in enumerate(critical_v, 1):
                faces_str = f"on Face {', '.join(v.face_ids)}" if v.face_ids else ""
                lines.append(f"{i}. **{v.rule_id}** {faces_str}: {v.description}")
                lines.append(f"   - Severity: {getattr(v.severity, 'value', str(v.severity)).upper()} | Risk: {v.unaddressed_risk_score}/10")
                lines.append(f"   - Measured: {v.measured_value} | Required: {v.required_value}")
                lines.append(f"   - Fix: {v.fix_suggestion}")
                lines.append("")

        warning_v = [v for v in result.violations if getattr(v.severity, "value", str(v.severity)).upper() == "WARNING"]
        if warning_v:
            lines.append("## Warning Findings (Pinpointed Features)")
            for i, v in enumerate(warning_v, 1):
                faces_str = f"on Face {', '.join(v.face_ids)}" if v.face_ids else ""
                lines.append(f"{i}. **{v.rule_id}** {faces_str}: {v.description}")
                lines.append(f"   - Risk: {v.unaddressed_risk_score}/10 | Fix: {v.fix_suggestion}")
                lines.append("")

    lines.append("## Recommended Actions")
    for i, v in enumerate(result.violations[:5], 1):
        if getattr(v.severity, "value", str(v.severity)).upper() in ("CRITICAL", "WARNING"):
            lines.append(f"{i}. **{v.rule_id}**: {v.fix_suggestion} ({v.solidworks_fix_path})")
    lines.append("")

    lines.append("## Sign-Off Checklist")
    checklist = [
        "All critical violations resolved (0 remaining)",
        "All warning violations reviewed and accepted or resolved",
        "GNN risk score below 0.5 (if applicable)",
        "Part weight within specification",
        "Material compatibility confirmed",
        "Assembly clearance verified"
    ]
    for item in checklist:
        lines.append(f"- [ ] {item}")

    return {"report": "\n".join(lines)}


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (on page 2 and later)
        if self._pageNumber > 1:
            self.drawString(54, 750, "EUREKA DFM — Design for Manufacturability Validation Report")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer (on all pages)
        self.drawString(54, 40, "Confidential — Generated by Eureka DFM 3.0")
        self.drawRightString(558, 40, f"Page {self._pageNumber} of {page_count}")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)
        
        self.restoreState()


def make_safe_text(text: str) -> str:
    if not text:
        return ""
    escaped = html.escape(str(text))
    # Restore allowed ReportLab tags
    escaped = escaped.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    escaped = escaped.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    escaped = escaped.replace("&lt;br/&gt;", "<br/>").replace("&lt;br&gt;", "<br/>")
    escaped = escaped.replace("&lt;font", "<font").replace("&lt;/font&gt;", "</font>")
    escaped = escaped.replace("&quot;", "\"").replace("&#x27;", "'")
    return escaped


def build_violation_card(violation: dict,
                          face_snapshots: dict,
                          faces_geometry: dict,
                          styles) -> list:
    """
    Builds a structured violation card with:
    - Left column: violation metadata
    - Right column: face snapshot image with annotation
    """

    severity = violation.get("severity", "INFO")
    rule_id  = violation.get("rule_id", "—")
    face_ids = violation.get("face_ids", [])

    # Color coding
    severity_colors = {
        "CRITICAL": colors.HexColor("#C0392B"),
        "WARNING":  colors.HexColor("#E67E22"),
        "INFO":     colors.HexColor("#2980B9"),
    }
    accent = severity_colors.get(severity, colors.grey)

    # ── Left column: text metadata ──────────────────────────────
    left_content = [
        Paragraph(f'<font color="{accent.hexval()}"><b>'
                  f'[{severity}] {make_safe_text(rule_id)}</b></font>',
                  styles["Heading3"]),
        Spacer(1, 4),
        Paragraph(f'<b>Description:</b> '
                  f'{make_safe_text(violation.get("description", ""))}',
                  styles["Normal"]),
        Spacer(1, 4),
        Paragraph(f'<b>Measured:</b> '
                  f'{make_safe_text(violation.get("measured_value", "—"))}',
                  styles["Normal"]),
        Paragraph(f'<b>Required:</b> '
                  f'{make_safe_text(violation.get("required_value", "—"))}',
                  styles["Normal"]),
        Spacer(1, 6),
        Paragraph(f'<b>Fix:</b> '
                  f'{make_safe_text(violation.get("fix_suggestion", ""))}',
                  styles["Normal"]),
    ]

    # Add Gemini plain-english if present
    if violation.get("plain_english"):
        left_content += [
            Spacer(1, 4),
            Paragraph(f'<i>{make_safe_text(violation["plain_english"])}</i>',
                      styles["Normal"]),
        ]

    # ── Right column: face snapshot ─────────────────────────────
    right_content = []

    # Find first face_id that has a snapshot
    snapshot_b64 = None
    snapshot_face_id = None
    for fid in face_ids:
        if fid in face_snapshots:
            snapshot_b64 = face_snapshots[fid]
            snapshot_face_id = fid
            break

    if snapshot_b64:
        try:
            img_bytes = base64.b64decode(snapshot_b64)
            img_buf   = BytesIO(img_bytes)

            # Render at fixed width (200pt) and height (150pt) to fit 210pt column
            img = RLImage(img_buf, width=200, height=150)
            img.hAlign = "CENTER"

            # Face annotation below image
            face_geo = faces_geometry.get(snapshot_face_id, {})
            area     = face_geo.get("area_mm2", 0)
            ftype    = face_geo.get("face_type", "")

            caption_parts = [f"Face: {snapshot_face_id}"]
            if ftype:
                caption_parts.append(ftype)
            if area > 0:
                caption_parts.append(f"{area:.1f} mm²")

            right_content = [
                img,
                Spacer(1, 3),
                Paragraph(
                    make_safe_text(" · ".join(caption_parts)),
                    styles["Caption"]   # small grey italic
                ),
            ]
        except Exception:
            right_content = [
                Paragraph("(snapshot unavailable)", styles["Normal"])
            ]
    else:
        # Fallback: show face ID list if no snapshot
        if face_ids:
            right_content = [
                Paragraph(
                    f"Affected faces:<br/>" +
                    "<br/>".join(make_safe_text(fid) for fid in face_ids),
                    styles["Normal"]
                )
            ]

    # ── Assemble two-column table ───────────────────────────────
    card_table = Table(
        [[left_content, right_content]],
        colWidths=[294, 210],
        style=TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING",   (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0), (-1,-1), 8),
            ("BOX",          (0,0), (-1,-1), 0.5,
                             colors.HexColor("#DDDDDD")),
            ("LINEABOVE",    (0,0), (-1,0),  2, accent),
            ("BACKGROUND",   (0,0), (-1,-1),
                             colors.HexColor("#FAFAFA")),
        ])
    )

    return [KeepTogether([card_table, Spacer(1, 8)])]


@router.post("/pdf")
async def generate_pdf_report(payload: ReportPayload):
    result          = payload.result
    face_snapshots  = payload.face_snapshots or {}
    faces_geometry  = {
        f["face_id"]: f
        for f in (payload.faces_geometry or [])
    }
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54
        )
        
        styles = getSampleStyleSheet()
        
        c_primary = colors.HexColor("#1A365D")
        c_text = colors.HexColor("#2D3748")
        c_sec = colors.HexColor("#718096")
        
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=c_primary
        )
        
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=c_sec
        )
        
        h1_style = ParagraphStyle(
            'Heading1_Custom',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            textColor=c_primary,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True
        )
        
        h2_style = ParagraphStyle(
            'Heading2_Custom',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=13,
            textColor=c_primary,
            spaceBefore=6,
            spaceAfter=4,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            'Body_Custom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=c_text
        )
        
        story = []
        
        # --- HEADER SECTION ---
        header_data = [
            [
                Paragraph("<b>EUREKA DFM</b><br/><font size=8.5 color='#718096'>Design for Manufacturability Analysis</font>", title_style),
                Paragraph(f"<b>Part ID:</b> {make_safe_text(result.part_id or 'Unknown')}<br/><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style)
            ]
        ]
        header_table = Table(header_data, colWidths=[320, 184])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(header_table)
        
        divider = Table([[""]], colWidths=[504])
        divider.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.5, c_primary),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(divider)
        story.append(Spacer(1, 10))
        
        # --- EXECUTIVE SUMMARY ---
        score = result.overall_manufacturability_score
        score_color = "#E05252"
        if score >= 80:
            score_color = "#52C47A"
        elif score >= 60:
            score_color = "#E09A3A"
        
        score_label = get_score_label(score)
        
        score_val_style = ParagraphStyle(
            'ScoreValCustom',
            parent=body_style,
            fontName='Helvetica-Bold',
            fontSize=32,
            leading=36,
            alignment=1
        )
        score_lbl_style = ParagraphStyle(
            'ScoreLblCustom',
            parent=body_style,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor("#718096"),
            fontSize=9,
            leading=12,
            alignment=1
        )
        score_label_style = ParagraphStyle(
            'ScoreLabelCustom',
            parent=body_style,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor(score_color),
            fontSize=8,
            leading=11,
            alignment=1
        )

        score_cell_data = [
            [Paragraph("<b>DFM SCORE</b>", score_lbl_style)],
            [Paragraph(f"<b><font color='{score_color}'>{score}</font></b><font size=16 color='#718096'>/100</font>", score_val_style)],
            [Paragraph(f"<b>{score_label.upper()}</b>", score_label_style)]
        ]
        score_widget_table = Table(score_cell_data, colWidths=[150])
        score_widget_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        
        process_name = PROCESS_INFO.get(result.process, {}).get("name", result.process) if result.process else "Unknown"
        material_name = result.material if result.material else "Unknown"
        
        c = result.risk_summary
        crit_count = c.get('critical_count', 0)
        warn_count = c.get('warning_count', 0)
        info_count = c.get('info_count', 0)
        
        meta_html = f"""
        <b>Manufacturing Process:</b> {make_safe_text(process_name)}<br/>
        <b>Selected Material:</b> {make_safe_text(material_name)}<br/>
        <b>GNN Interaction Risk:</b> {result.gnn_risk_score:.2f}<br/>
        <b>AI Analysis Confidence:</b> {int(result.confidence * 100)}%
        """
        
        summary_html = f"""
        <b>DFM Findings Summary:</b><br/>
        <font color='#E05252'>● <b>{crit_count} Critical Issues</b></font><br/>
        <font color='#E09A3A'>● <b>{warn_count} Warning Issues</b></font><br/>
        <font color='#4A9EE0'>● <b>{info_count} Info Points</b></font>
        """
        
        summary_table_data = [
            [Paragraph(meta_html, body_style), Paragraph(summary_html, body_style)]
        ]
        summary_table = Table(summary_table_data, colWidths=[174, 160])
        summary_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        
        exec_data = [
            [score_widget_table, summary_table]
        ]
        exec_table = Table(exec_data, colWidths=[150, 354])
        exec_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (1,0), (1,0), 10),
        ]))
        story.append(exec_table)
        story.append(Spacer(1, 15))

        # --- SCREENSHOT IMAGE (IF PROVIDED) ---
        img_temp_path = None
        if getattr(result, "screenshot_png_base64", None):
            try:
                img_data = base64.b64decode(result.screenshot_png_base64)
                img_temp_path = f"temp_screenshot_{result.part_id}.png"
                with open(img_temp_path, "wb") as f:
                    f.write(img_data)
                
                cad_img = RLImage(img_temp_path, width=300, height=180)
                
                img_desc = """
                <b>Visual DFM Heatmap & Face Highlights</b><br/>
                The CAD viewport screenshot on the left shows the model with real-time face highlights indicating DFM risk areas:<br/>
                ● <font color='#E05252'><b>RED</b></font>: Critical violations (wall thickness or corner fillet radius below limit).<br/>
                ● <font color='#E09A3A'><b>ORANGE</b></font>: GNN high geometric anomaly risk zones.<br/>
                ● <font color='#4A9EE0'><b>YELLOW</b></font>: Warnings / Informational regions (insufficient draft, GDT control).<br/><br/>
                <i>Note: These colors are projected live onto the CAD geometry inside the SolidWorks active window.</i>
                """
                
                img_table_data = [
                    [cad_img, Paragraph(img_desc, body_style)]
                ]
                img_table = Table(img_table_data, colWidths=[300, 204])
                img_table.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
                    ('PADDING', (0,0), (-1,-1), 8),
                ]))
                
                story.append(Paragraph("SolidWorks Viewport & Face Highlights", h1_style))
                story.append(img_table)
                story.append(Spacer(1, 12))
            except Exception as img_ex:
                logger.error(f"Failed to embed screenshot: {img_ex}")

        # --- GEMINI NARRATIVE SUMMARY ---
        narrative = await call_gemini_with_timeout(
            generate_gemini_report_narrative,
            result,
            timeout_seconds=8.0,
            fallback="Executive summary unavailable."
        )
        if narrative:
            story.append(Paragraph("Principal Engineering DFM Narrative", h1_style))
            for p_text in narrative.split("\n\n"):
                if p_text.strip():
                    story.append(Paragraph(make_safe_text(p_text.strip()), body_style))
                    story.append(Spacer(1, 6))
            story.append(Spacer(1, 8))
        
        # --- AI ANOMALY INSIGHTS ---
        gnn_anom = getattr(result, "gnn_anomaly", None)
        explanation = None
        if isinstance(gnn_anom, dict) and gnn_anom.get("gemini_explanation"):
            explanation = gnn_anom["gemini_explanation"]
        
        if explanation:
            story.append(Paragraph("AI Anomaly Diagnosis & Risk Analysis", h1_style))
            ai_text = f"<b>✦ Gemini AI Insights:</b><br/>{make_safe_text(explanation)}"
            ai_table_data = [[Paragraph(ai_text, body_style)]]
            ai_table = Table(ai_table_data, colWidths=[504])
            ai_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FAF5FF")),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D1C4E9")),
                ('PADDING', (0,0), (-1,-1), 10),
                ('LINELEFT', (0,0), (0,-1), 3.0, colors.HexColor("#805AD5")),
            ]))
            story.append(ai_table)
            story.append(Spacer(1, 12))
            
        # --- DETAILED FINDINGS ---
        if not face_snapshots and getattr(result, "screenshot_png_base64", None):
            story.append(Paragraph(
                "<i>See colored face overlay in viewport screenshot above — face IDs annotated.</i>",
                body_style
            ))
            story.append(Spacer(1, 8))

        violations = result.violations or []
        critical_violations = [v for v in violations if getattr(v.severity, "value", str(v.severity)).upper() == "CRITICAL"]
        warning_violations = [v for v in violations if getattr(v.severity, "value", str(v.severity)).upper() == "WARNING"]
        info_violations = [v for v in violations if getattr(v.severity, "value", str(v.severity)).upper() not in ("CRITICAL", "WARNING")]

        card_styles = {
            "Heading3": ParagraphStyle(
                'CardHeading3',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=9.5,
                leading=13,
                textColor=colors.HexColor("#1A365D")
            ),
            "Normal": ParagraphStyle(
                'CardNormal',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=8.5,
                leading=11.5,
                textColor=colors.HexColor("#2D3748")
            ),
            "Caption": ParagraphStyle(
                'CardCaption',
                parent=styles['Normal'],
                fontName='Helvetica-Oblique',
                fontSize=7.5,
                leading=10,
                textColor=colors.HexColor("#718096"),
                alignment=1
            ),
            "Heading2": ParagraphStyle(
                'CardHeading2',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=10,
                leading=13,
                textColor=colors.HexColor("#1A365D")
            )
        }

        def draw_violations_group(group_list, title, color_hex):
            if not group_list:
                return
            story.append(Paragraph(title, h1_style))
            for v in group_list:
                v_dict = v.model_dump() if hasattr(v, "model_dump") else (v.dict() if hasattr(v, "dict") else v)
                cards = build_violation_card(
                    v_dict, face_snapshots, faces_geometry, card_styles
                )
                story.extend(cards)
            story.append(Spacer(1, 6))

        draw_violations_group(critical_violations, "Critical Findings", "#E05252")
        draw_violations_group(warning_violations, "Warning Findings", "#E09A3A")
        draw_violations_group(info_violations, "Informational Findings", "#4A9EE0")
        
        # --- SIGN-OFF CHECKLIST ---
        story.append(Paragraph("Quality Check & Tooling Sign-Off Checklist", h1_style))
        checklist_data = []
        checklist = [
            f"All critical DFM violations resolved ({len(critical_violations)} remaining)",
            f"All warning violations reviewed and signed off ({len(warning_violations)} remaining)",
            "Material compatibility and flow characteristics confirmed in simulation",
            "Nominal wall thickness variation is within acceptable tolerance range",
            "Pull direction and draft angles verified to prevent tool drag",
            "GNN anomaly interaction risk level below acceptable threshold (0.50)"
        ]
        for item in checklist:
            checklist_data.append([
                Paragraph("[ &nbsp; ]", ParagraphStyle('Check', parent=body_style, fontName='Helvetica-Bold')),
                Paragraph(item, body_style)
            ])
        chk_table = Table(checklist_data, colWidths=[24, 480])
        chk_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(chk_table)
        
        # Build PDF Document
        try:
            doc.build(story, canvasmaker=NumberedCanvas)
        finally:
            if img_temp_path and os.path.exists(img_temp_path):
                try:
                    os.remove(img_temp_path)
                except:
                    pass
        
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf")
        
    except Exception as e:
        import logging
        logging.getLogger("uvicorn").error(f"Error in PDF generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF report generation failed: {str(e)}")
