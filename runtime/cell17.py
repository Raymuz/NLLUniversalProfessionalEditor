#@title 9. PROFESSIONAL GRADIO UI v5.1 — FULL 30-RULE DIAGNOSTICS
# Complete replacement for Cell 9.
# Compatible with Cell 5 v4.1, Cell 5A v3.4, Cell 6 v5.1,
# Cell 7 v4, Cell 7A v2, Cell 7B v2, Cell 8 v4, and Cell 8A v2.

import json
import html
import re
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from dataclasses import asdict, is_dataclass

import pandas as pd
import gradio as gr


# =====================================================================
# 1. PREFLIGHT
# =====================================================================

EXPECTED_RULE_COUNT = 30

_REQUIRED_CELL9 = [
    "ENGINE", "Engine", "Analyzer", "Generator", "Validator", "Ranker",
    "RULES", "PROFILES", "realization_issues",
]

_missing_cell9 = [name for name in _REQUIRED_CELL9 if name not in globals()]
if _missing_cell9:
    raise RuntimeError(
        "Cell 9 cannot run. Missing: " + ", ".join(_missing_cell9)
        + ". Run Cells 1–8A in order first."
    )

if len(RULES) != EXPECTED_RULE_COUNT:
    raise RuntimeError(
        f"Cell 9 expected {EXPECTED_RULE_COUNT} registered transformation "
        f"rules, but found {len(RULES)}. Rerun Cells 5, 5A, 6, 7, 7A, "
        "and 7B in order."
    )

MUST_PARAPHRASE_ACTIVE = (
    getattr(Engine.one, "__name__", "") == "_one_with_must_paraphrase_v2"
)

if not MUST_PARAPHRASE_ACTIVE:
    raise RuntimeError(
        "Cell 9 requires the Cell 7B v2 must-paraphrase orchestrator. "
        "Run Cell 7B v2 before Cell 9."
    )


# =====================================================================
# 2. DETERMINISTIC SETTINGS
# =====================================================================

def _set_cfg_flag(name, value):
    if "CFG" in globals():
        try:
            setattr(CFG, name, bool(value))
        except Exception:
            pass


def _apply_ui_settings(use_stanza, use_language_tool):
    # No generative/model-assisted candidate production in this UI.
    _set_cfg_flag("mlm", False)
    _set_cfg_flag("sts", False)
    _set_cfg_flag("nli", False)
    _set_cfg_flag("online", False)

    # Optional read-only analysis resources.
    _set_cfg_flag("stanza", use_stanza)
    _set_cfg_flag("lt", use_language_tool)


# =====================================================================
# 3. GENERAL HELPERS
# =====================================================================

def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _file_path(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    return getattr(uploaded_file, "name", None) or str(uploaded_file)


def _read_input_text(typed_text, uploaded_file):
    typed_text = str(typed_text or "")
    if typed_text.strip():
        return typed_text.strip(), "typed_text"

    path_value = _file_path(uploaded_file)
    if not path_value:
        raise gr.Error("Enter text or upload a TXT, DOCX, or text-based PDF.")

    path = Path(path_value)
    if not path.exists():
        raise gr.Error("The uploaded file could not be found.")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace").strip(), path.name

    if suffix == ".docx":
        from docx import Document
        document = Document(str(path))
        text = "\n\n".join(p.text for p in document.paragraphs).strip()
        return text, path.name

    if suffix == ".pdf":
        try:
            import fitz
        except Exception as exc:
            raise gr.Error("PyMuPDF is unavailable: " + str(exc))
        pdf = fitz.open(str(path))
        try:
            text = "\n\n".join(page.get_text("text") for page in pdf).strip()
        finally:
            pdf.close()
        if not text:
            raise gr.Error(
                "The PDF contains no extractable text. Scanned PDFs require OCR."
            )
        return text, path.name

    raise gr.Error("Unsupported file type. Use TXT, DOCX, or text-based PDF.")


def _trace(item):
    value = getattr(item, "diagnostics", None)
    if value is None:
        value = getattr(item, "trace", None)
    return dict(value or {})


def _checks(item):
    return dict(getattr(item, "checks", None) or {})


def _changed(item):
    return str(getattr(item, "original", "")) != str(getattr(item, "revised", ""))


# =====================================================================
# 4. DIAGNOSTIC TABLES
# =====================================================================

def _sentence_rows(result):
    rows = []
    for number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        trace = _trace(item)
        selected = trace.get("selected") or {}
        generated = trace.get("generated_by_lane", {}) or {}
        must = trace.get("must_paraphrase", {}) or {}
        attempts = must.get("attempts", []) or []
        rows.append({
            "Sentence": number,
            "Status": getattr(item, "status", ""),
            "Changed": _changed(item),
            "Requested mode": must.get("requested_mode", getattr(result, "mode", "")),
            "Accepted mode": must.get("accepted_mode", ""),
            "Attempts": must.get("attempt_count", len(attempts)),
            "Must-paraphrase outcome": must.get("outcome", ""),
            "Detected families": " | ".join(list(getattr(item, "families", []) or [])),
            "Selected family": getattr(item, "family", None) or selected.get("family") or "",
            "Selected lane": selected.get("lane", ""),
            "Selected score": selected.get("score"),
            "Minimal candidates": generated.get("Minimal", 0),
            "Balanced candidates": generated.get("Balanced", 0),
            "Structural candidates": generated.get("Structural", 0),
            "Before final filter": trace.get("generated_before_final_filter"),
            "After final filter": trace.get("generated_after_final_filter"),
            "Original": getattr(item, "original", ""),
            "Revised": getattr(item, "revised", ""),
            "Reason": getattr(item, "reason", ""),
        })
    return rows


def _routing_rows(result):
    rows = []
    for number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        trace = _trace(item)
        applicable = trace.get("applicable", {}) or {}
        generated = trace.get("generated_by_lane", {}) or {}
        rows.append({
            "Sentence": number,
            "Status": getattr(item, "status", ""),
            "Detected families": " | ".join(
                trace.get("detected_families", [])
                or list(getattr(item, "families", []) or [])
            ),
            "Minimal rules": " | ".join(applicable.get("Minimal", []) or []),
            "Balanced rules": " | ".join(applicable.get("Balanced", []) or []),
            "Structural rules": " | ".join(applicable.get("Structural", []) or []),
            "Minimal generated": generated.get("Minimal", 0),
            "Balanced generated": generated.get("Balanced", 0),
            "Structural generated": generated.get("Structural", 0),
            "Before final filter": trace.get("generated_before_final_filter"),
            "After final filter": trace.get("generated_after_final_filter"),
            "Rule errors": json.dumps(_json_safe(trace.get("rule_errors", [])), ensure_ascii=False),
            "Terminal rejections": json.dumps(_json_safe(trace.get("rejections", {})), ensure_ascii=False),
        })
    return rows


def _realization_rows(result):
    rows = []
    for sentence_number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        for rejection_number, rejection in enumerate(
            list(_trace(item).get("realization_rejections", []) or []), 1
        ):
            rows.append({
                "Sentence": sentence_number,
                "Rejection": rejection_number,
                "Stage": rejection.get("stage", ""),
                "Lane": rejection.get("lane", ""),
                "Family/rule": rejection.get("rule", ""),
                "Issues": " | ".join(rejection.get("issues", []) or []),
                "Rejected text": rejection.get("text", ""),
            })
    if not rows:
        rows.append({
            "Sentence": 0, "Rejection": 0, "Stage": "", "Lane": "",
            "Family/rule": "", "Issues": "None", "Rejected text": "",
        })
    return rows


def _candidate_rows(result):
    rows = []
    for sentence_number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        candidates = list(_trace(item).get("candidate_traces", []) or [])
        if not candidates:
            rows.append({
                "Sentence": sentence_number,
                "Candidate": 0,
                "Lane": "",
                "Family": "",
                "Accepted": False,
                "Score": None,
                "Change": None,
                "Failures": "",
                "Rejection reasons": "No candidate trace available",
                "Candidate text": "",
            })
            continue
        for candidate_number, candidate in enumerate(candidates, 1):
            rows.append({
                "Sentence": sentence_number,
                "Candidate": candidate_number,
                "Lane": candidate.get("lane", ""),
                "Family": candidate.get("family", ""),
                "Accepted": candidate.get("accepted", False),
                "Score": candidate.get("score"),
                "Change": candidate.get("change"),
                "Failures": " | ".join(candidate.get("failures", []) or []),
                "Rejection reasons": json.dumps(
                    _json_safe(candidate.get("rejection_reasons", [])),
                    ensure_ascii=False,
                ),
                "Candidate text": candidate.get("text", ""),
            })
    return rows


def _attempt_rows(result):
    rows = []
    for sentence_number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        must = _trace(item).get("must_paraphrase", {}) or {}
        attempts = must.get("attempts", []) or []
        if not attempts:
            rows.append({
                "Sentence": sentence_number, "Attempt": 0, "Mode": "",
                "Status": "", "Changed": False, "Selected family": "",
                "Failures": "", "Realization issues": "",
                "Generated after filter": None, "Revised": "",
            })
            continue
        for attempt_number, attempt in enumerate(attempts, 1):
            rows.append({
                "Sentence": sentence_number,
                "Attempt": attempt_number,
                "Mode": attempt.get("attempted_mode", ""),
                "Status": attempt.get("status", ""),
                "Changed": attempt.get("changed", False),
                "Selected family": attempt.get("selected_family", ""),
                "Failures": " | ".join(attempt.get("failures", []) or []),
                "Realization issues": " | ".join(attempt.get("realization_issues", []) or []),
                "Generated after filter": attempt.get("generated_after_final_filter"),
                "Revised": attempt.get("revised", ""),
            })
    return rows


def _finding_rows(result):
    rows = []
    for finding in list(getattr(result, "findings", []) or []):
        safe = _json_safe(finding)
        rows.append(safe if isinstance(safe, dict) else {"Finding": safe})
    return rows


# =====================================================================
# 5. SUMMARY AND FULL PAYLOAD
# =====================================================================

def _summary_markdown(result):
    sentences = list(getattr(result, "sentences", []) or [])
    total = len(sentences)
    successes = sum(getattr(item, "status", "") == "SUCCESS" for item in sentences)
    changed = sum(_changed(item) for item in sentences)
    statuses = Counter(getattr(item, "status", "UNKNOWN") for item in sentences)
    attempts = sum(
        int((_trace(item).get("must_paraphrase", {}) or {}).get("attempt_count", 0) or 0)
        for item in sentences
    )
    preserved = sum(
        (_trace(item).get("must_paraphrase", {}) or {}).get("outcome")
        == "ORIGINAL_PRESERVED_AFTER_EXHAUSTIVE_SAFE_SEARCH"
        for item in sentences
    )
    status_text = ", ".join(f"{name}: {count}" for name, count in sorted(statuses.items()))
    warnings = list(getattr(result, "warnings", []) or [])
    warning_text = "<br>".join(html.escape(str(x)) for x in warnings) if warnings else "None"

    return f"""
### Processing summary

- **Discipline:** {getattr(result, 'discipline', '')}
- **Routing evidence:** {', '.join(map(str, getattr(result, 'evidence', []) or [])) or 'default profile'}
- **Writing Pattern Score:** {getattr(result, 'pattern_score', '')} — **{getattr(result, 'band', '')}**
- **Registered transformation rules:** {len(RULES)}
- **Must-paraphrase orchestrator:** {'Active' if MUST_PARAPHRASE_ACTIVE else 'Inactive'}
- **Sentences processed:** {total}
- **Successful selections:** {successes}
- **Changed sentences:** {changed}
- **Unchanged sentences:** {total - changed}
- **Total bounded attempts:** {attempts}
- **Preserved after exhaustive safe search:** {preserved}
- **Terminal statuses:** {status_text or 'None'}
- **Warnings:** {warning_text}

> The Writing Pattern Score describes observable structural and stylistic characteristics only. It does not identify authorship.
"""


def _full_payload(result, source_name):
    sentences = []
    for number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        sentences.append({
            "sentence": number,
            "status": getattr(item, "status", ""),
            "original": getattr(item, "original", ""),
            "revised": getattr(item, "revised", ""),
            "families": _json_safe(getattr(item, "families", [])),
            "selected_family": getattr(item, "family", None),
            "reason": getattr(item, "reason", ""),
            "checks": _json_safe(_checks(item)),
            "trace": _json_safe(_trace(item)),
        })

    return {
        "source_name": source_name,
        "workflow": getattr(result, "workflow", ""),
        "mode": getattr(result, "mode", ""),
        "discipline": getattr(result, "discipline", ""),
        "routing_evidence": _json_safe(getattr(result, "evidence", [])),
        "writing_pattern_score": getattr(result, "pattern_score", None),
        "writing_pattern_band": getattr(result, "band", ""),
        "warnings": _json_safe(getattr(result, "warnings", [])),
        "stats": _json_safe(getattr(result, "stats", {})),
        "registered_rule_count": len(RULES),
        "must_paraphrase_active": MUST_PARAPHRASE_ACTIVE,
        "must_paraphrase_wrapper": getattr(Engine.one, "__name__", ""),
        "rewrite_configuration": {
            "deterministic_rules": True,
            "masked_language_model": False,
            "sts_model": False,
            "nli_model": False,
            "online_lexicon": False,
            "language_tool_optional": bool(getattr(globals().get("CFG", object()), "lt", False)),
        },
        "registered_rules": [
            {
                "name": registered.name,
                "lanes": sorted(registered.lanes),
                "families": sorted(registered.families),
                "priority": registered.priority,
            }
            for registered in RULES
        ],
        "sentences": sentences,
    }


# =====================================================================
# 6. REPORTS
# =====================================================================

def _write_reports(result, source_name, payload):
    folder = Path(tempfile.mkdtemp(prefix="nll_review_"))
    stem = Path(source_name or "nll_input").stem
    token = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    txt_path = folder / f"{stem}_{token}_revised.txt"
    json_path = folder / f"{stem}_{token}_diagnostics.json"
    docx_path = folder / f"{stem}_{token}_review.docx"
    html_path = folder / f"{stem}_{token}_review.html"

    revised = str(getattr(result, "revised", ""))
    txt_path.write_text(revised, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    from docx import Document
    document = Document()
    document.add_heading("NLL Universal Professional Editor — Review", level=0)
    document.add_paragraph("Discipline: " + str(getattr(result, "discipline", "")))
    document.add_paragraph(
        "Writing Pattern Score: " + str(getattr(result, "pattern_score", ""))
        + " — " + str(getattr(result, "band", ""))
    )
    document.add_paragraph(
        "The Writing Pattern Score describes observable structural and stylistic "
        "characteristics only. It does not identify authorship."
    )
    document.add_heading("Revised text", level=1)
    for paragraph in re.split(r"\n\s*\n", revised):
        document.add_paragraph(paragraph)

    document.add_heading("Sentence review", level=1)
    for number, item in enumerate(list(getattr(result, "sentences", []) or []), 1):
        trace = _trace(item)
        document.add_heading(f"Sentence {number}: {getattr(item, 'status', '')}", level=2)
        document.add_paragraph("Original: " + str(getattr(item, "original", "")))
        document.add_paragraph("Revised: " + str(getattr(item, "revised", "")))
        document.add_paragraph("Reason: " + str(getattr(item, "reason", "")))
        document.add_paragraph(
            "Must-paraphrase: "
            + json.dumps(_json_safe(trace.get("must_paraphrase", {})), ensure_ascii=False)
        )
    document.save(str(docx_path))

    rows = _sentence_rows(result)
    body = "".join(
        "<tr>"
        f"<td>{row['Sentence']}</td>"
        f"<td>{html.escape(str(row['Status']))}</td>"
        f"<td>{html.escape(str(row['Accepted mode']))}</td>"
        f"<td>{html.escape(str(row['Detected families']))}</td>"
        f"<td>{html.escape(str(row['Original']))}</td>"
        f"<td>{html.escape(str(row['Revised']))}</td>"
        f"<td>{html.escape(str(row['Reason']))}</td>"
        "</tr>"
        for row in rows
    )
    html_path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>NLL Review</title><style>
body{{font-family:Arial,sans-serif;max-width:1500px;margin:30px auto;color:#172033}}
h1,h2{{color:#123b66}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #cbd5e1;padding:8px;vertical-align:top;text-align:left}}
th{{background:#eaf2fb}}pre{{white-space:pre-wrap;background:#f8fafc;padding:12px}}
</style></head><body>
<h1>NLL Universal Professional Editor — Review</h1>
<p><strong>Rules:</strong> {len(RULES)} | <strong>Must paraphrase:</strong> {MUST_PARAPHRASE_ACTIVE}</p>
<h2>Revised text</h2><pre>{html.escape(revised)}</pre>
<h2>Sentence review</h2><table><thead><tr><th>#</th><th>Status</th><th>Mode</th>
<th>Families</th><th>Original</th><th>Revised</th><th>Reason</th></tr></thead>
<tbody>{body}</tbody></table>
<h2>Full diagnostics</h2><pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>
</body></html>""",
        encoding="utf-8",
    )

    return [str(txt_path), str(docx_path), str(json_path), str(html_path)]


# =====================================================================
# 7. MAIN CALLBACK
# =====================================================================

def run_ui_with_full_diagnostics(
    typed_text,
    uploaded_file,
    workflow,
    mode,
    discipline,
    protected_terms,
    use_stanza,
    use_language_tool,
    progress=gr.Progress(),
):
    source_text, source_name = _read_input_text(typed_text, uploaded_file)
    if not source_text.strip():
        raise gr.Error("No readable text was provided.")

    _apply_ui_settings(use_stanza, use_language_tool)
    progress(0.02, desc="Preparing request")

    try:
        result = ENGINE.process(
            source_text, workflow, mode, discipline, protected_terms, progress
        )
    except TypeError:
        result = ENGINE.process(
            text=source_text,
            workflow=workflow,
            mode=mode,
            discipline=discipline,
            terms_text=protected_terms,
            progress=progress,
        )

    progress(0.86, desc="Building diagnostics")
    payload = _full_payload(result, source_name)
    files = _write_reports(result, source_name, payload)
    progress(1.0, desc="Complete")

    return (
        getattr(result, "revised", source_text),
        _summary_markdown(result),
        pd.DataFrame(_sentence_rows(result)),
        pd.DataFrame(_routing_rows(result)),
        pd.DataFrame(_attempt_rows(result)),
        pd.DataFrame(_realization_rows(result)),
        pd.DataFrame(_candidate_rows(result)),
        pd.DataFrame(_finding_rows(result)),
        payload,
        files,
    )


# =====================================================================
# 8. PROFESSIONAL GRADIO UI
# =====================================================================

CELL9_CSS = """
.gradio-container { max-width: 1650px !important; }
#nll-title { text-align:center; margin-bottom:4px; color:#123b66; }
#nll-subtitle { text-align:center; color:#596779; margin-bottom:16px; }
.nll-panel { border:1px solid #d5dee8; border-radius:14px; padding:12px; background:#fbfdff; }
.nll-note { background:#fff8dc; border-left:4px solid #d7a100; padding:8px 12px; }
.nll-ok { background:#e8f6ee; border-left:4px solid #238636; padding:8px 12px; }
"""


with gr.Blocks(css=CELL9_CSS, title="NLL Universal Professional Editor") as NLL_APP:
    gr.Markdown("# NLL Universal Professional Editor", elem_id="nll-title")
    gr.Markdown(
        "Deterministic, non-generative rewriting with must-paraphrase orchestration, "
        "meaning safeguards, and complete sentence/candidate diagnostics.",
        elem_id="nll-subtitle",
    )

    gr.Markdown(
        f"**System ready:** {len(RULES)} rules | Must-paraphrase active | "
        "Cell 5A realization gate active | Cell 7 judge active",
        elem_classes="nll-ok",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="nll-panel"):
            gr.Markdown("## Input and controls")
            input_text = gr.Textbox(
                label="Text",
                lines=20,
                placeholder="Paste academic, contract, policy, technical, business, or professional text.",
            )
            input_file = gr.File(
                label="Optional file upload",
                file_types=[".txt", ".docx", ".pdf"],
                type="filepath",
            )
            workflow_control = gr.Radio(
                choices=["Scan Only", "Scan & Neutralize", "Edit / Paraphrase Only"],
                value="Edit / Paraphrase Only",
                label="Workflow",
            )
            mode_control = gr.Radio(
                choices=["Minimal", "Balanced", "Structural"],
                value="Balanced",
                label="Preferred rewrite mode",
            )
            discipline_control = gr.Dropdown(
                choices=["Auto Detect"] + list(PROFILES.keys()),
                value="Auto Detect",
                label="Discipline",
            )
            protected_terms_control = gr.Textbox(
                label="Immutable/user-protected terms",
                placeholder="Comma, semicolon, or line-separated terms, e.g., WSC, TxDOT, ALCEF-RSF",
                lines=3,
            )
            with gr.Accordion("Optional non-generative analysis resources", open=False):
                stanza_control = gr.Checkbox(
                    value=bool(getattr(globals().get("CFG", object()), "stanza", False)),
                    label="Stanza parsing support",
                )
                language_tool_control = gr.Checkbox(
                    value=bool(getattr(globals().get("CFG", object()), "lt", False)),
                    label="Local LanguageTool comparison",
                )
            analyze_button = gr.Button("Analyze / Revise", variant="primary", size="lg")
            gr.Markdown(
                "The selected mode is attempted first. If no safe change survives, "
                "the other modes are searched. The original is retained only after "
                "all safe deterministic options are exhausted.",
                elem_classes="nll-note",
            )

        with gr.Column(scale=1, elem_classes="nll-panel"):
            gr.Markdown("## Revised output and review")
            revised_output = gr.Textbox(label="Revised output", lines=20)
            summary_output = gr.Markdown()

            with gr.Tabs():
                with gr.Tab("Sentence comparison"):
                    sentence_table = gr.Dataframe(
                        label="Original, revised, mode, family, and judge outcome",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Routing diagnostics"):
                    routing_table = gr.Dataframe(
                        label="Detected families, applicable rules, and generated candidates",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Must-paraphrase attempts"):
                    attempt_table = gr.Dataframe(
                        label="Requested-mode-first judge and retry attempts",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Realization rejections"):
                    realization_table = gr.Dataframe(
                        label="Candidates rejected by Cell 5A",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Candidate diagnostics"):
                    candidate_table = gr.Dataframe(
                        label="Validated candidates, scores, and rejection reasons",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Writing-pattern findings"):
                    findings_table = gr.Dataframe(
                        label="Observable structural and stylistic findings",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Full JSON diagnostics"):
                    diagnostic_json = gr.JSON(label="Complete diagnostic payload")
                with gr.Tab("Downloads"):
                    download_output = gr.File(
                        label="Download revised text and review reports",
                        file_count="multiple",
                    )

    inputs = [
        input_text, input_file, workflow_control, mode_control,
        discipline_control, protected_terms_control,
        stanza_control, language_tool_control,
    ]
    outputs = [
        revised_output, summary_output, sentence_table, routing_table,
        attempt_table, realization_table, candidate_table, findings_table,
        diagnostic_json, download_output,
    ]

    analyze_button.click(
        fn=run_ui_with_full_diagnostics,
        inputs=inputs,
        outputs=outputs,
        show_progress="full",
        concurrency_limit=1,
    )

NLL_APP.queue(default_concurrency_limit=1, max_size=8)

print("=" * 76)
print("CELL 9 PROFESSIONAL GRADIO UI v5.1 LOADED")
print("=" * 76)
print("Aligned pipeline: 5 v4.1 -> 5A v3.4 -> 6 v5.1 -> 7 v4 -> 7A v2 -> 7B v2 -> 8 v4 -> 8A v2")
print("Registered transformation rules:", len(RULES))
print("Expected rule count:", EXPECTED_RULE_COUNT)
print("Must-paraphrase orchestrator: active")
print("Cell 5A realization diagnostics: active")
print("Sentence comparison: active")
print("Routing diagnostics: active")
print("Must-paraphrase attempt diagnostics: active")
print("Candidate diagnostics: active")
print("Writing Pattern Score terminology: active")
print("LLM/MLM candidate generation: disabled")
print("STS/NLI scoring: disabled")
print("Online lexical fallback: disabled")
print("TXT/DOCX/JSON/HTML downloads: active")
print("=" * 76)

# Gradio is mounted by the production ASGI app; do not launch here.