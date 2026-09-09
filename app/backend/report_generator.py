"""Structured Quality Audit Report Generator in JSON and PDF formats."""

import io
import json
from datetime import datetime
from typing import Dict, Any
from fpdf import FPDF


class RetinaGuardPDFReport(FPDF):
    """Clean clinical quality assurance PDF report template."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 35, 60)
        self.cell(0, 10, "RetinaGuard-QA | Technical Quality Audit Report", border=False, align="L")
        self.ln(6)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 110, 125)
        self.cell(0, 8, "Automated Pre-Diagnostic Fundus Acquisition Quality Gate", border=False, align="L")
        self.ln(10)
        self.set_draw_color(220, 225, 235)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 130, 140)
        self.cell(0, 5, "RetinaGuard-QA assesses technical acquisition quality only. Not a diagnostic medical device.", align="C")
        self.ln(4)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")


def generate_json_report(assessment_dict: Dict[str, Any], filename: str = "image.jpg") -> str:
    """Generate standardized JSON quality audit payload."""
    report = {
        "report_metadata": {
            "system_name": "RetinaGuard-QA",
            "version": assessment_dict.get("model_version", "1.0.0"),
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "evaluated_file": filename
        },
        "quality_assessment": assessment_dict
    }
    return json.dumps(report, indent=2)


def generate_pdf_report(assessment_dict: Dict[str, Any], filename: str = "image.jpg") -> bytes:
    """Generate a clean PDF report summarizing quality score, defects, and operator guidance."""
    pdf = RetinaGuardPDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # File & Time Info
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(40, 50, 70)
    pdf.cell(40, 6, "Evaluated File:", 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, str(filename), 0)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 6, "Audit Timestamp:", 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), 0)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 6, "Inference Latency:", 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"{assessment_dict.get('latency_ms', 0):.1f} ms (CPU Execution)", 0)
    pdf.ln(10)

    # Primary Decision Banner
    decision = assessment_dict.get("decision", "MANUAL_REVIEW")
    pdf.set_font("Helvetica", "B", 13)
    if decision == "ACCEPT":
        pdf.set_fill_color(230, 245, 230)
        pdf.set_text_color(25, 120, 45)
    elif decision == "USABLE_WITH_WARNING":
        pdf.set_fill_color(255, 248, 225)
        pdf.set_text_color(180, 110, 10)
    elif decision == "RECAPTURE_WITH_GUIDANCE":
        pdf.set_fill_color(255, 235, 235)
        pdf.set_text_color(180, 30, 30)
    else:
        pdf.set_fill_color(240, 240, 250)
        pdf.set_text_color(70, 60, 140)

    pdf.cell(0, 12, f"  ACTION: {assessment_dict.get('action_title', decision)}", fill=True, ln=True)
    pdf.ln(4)

    # Quality Grade & Score
    pdf.set_text_color(30, 40, 50)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(60, 7, "Quality Grade:", 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"{assessment_dict.get('quality_grade', 'N/A')} (Confidence: {assessment_dict.get('quality_grade_confidence', 0)*100:.1f}%)", 0)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(60, 7, "Quality Index (0-100):", 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"{assessment_dict.get('quality_score', 0):.1f} / 100", 0)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(60, 7, "Predictive Uncertainty:", 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Entropy = {assessment_dict.get('predictive_entropy', 0):.3f} bits", 0)
    pdf.ln(10)

    # Detected Defects
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 35, 60)
    pdf.cell(0, 8, "Acquisition Defects Analysis", 0, ln=True)
    pdf.set_font("Helvetica", "", 10)

    defects = assessment_dict.get("detected_defects", [])
    if defects:
        for d in defects:
            pdf.set_text_color(160, 30, 30)
            pdf.cell(5, 6, "-", 0)
            pdf.cell(0, 6, f"{d['name']} (Probability: {d['probability']*100:.1f}%)", 0, ln=True)
    else:
        pdf.set_text_color(30, 130, 50)
        pdf.cell(0, 6, "No severe optical defects detected.", 0, ln=True)

    pdf.ln(6)

    # Operator Actionable Instructions
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 35, 60)
    pdf.cell(0, 8, "Actionable Operator Instructions", 0, ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 50, 60)

    instructions = assessment_dict.get("actionable_instructions", [])
    for idx, inst in enumerate(instructions, 1):
        pdf.cell(6, 6, f"{idx}.", 0)
        pdf.multi_cell(0, 6, inst)
        pdf.ln(1)

    pdf.ln(8)
    pdf.set_draw_color(230, 230, 230)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Ethical boundary statement
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 130, 140)
    pdf.multi_cell(0, 4, assessment_dict.get("clinical_disclaimer", ""))

    return bytes(pdf.output())
