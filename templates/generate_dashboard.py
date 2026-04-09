"""
Generate an HTML status dashboard for the PATHS EEG pipeline.

Usage:
    python templates/generate_dashboard.py

Output:
    outputs/dashboard.html  — open this file in any web browser.

The dashboard shows, for each subject:
  - Which pipeline steps completed / failed / not yet run
  - Key QC metrics: ICA components removed, specparam R², exponent
  - Links to per-subject MNE HTML reports
"""
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Locate project
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "src" / "spectral"))
from spectral.utils import load_config, find_project_root

PROJECT_ROOT = find_project_root(_HERE)
cfg = load_config(PROJECT_ROOT / "settings.toml")
_pipe = cfg.get("pipeline", {})

BIDS_ROOT    = PROJECT_ROOT / cfg["paths"].get("bids_root", "data/bids")
OUTPUT_ROOT  = PROJECT_ROOT / _pipe.get("output_root",  "outputs/pipeline")
STATUS_FILE  = PROJECT_ROOT / _pipe.get("status_file",  "outputs/pipeline_status.json")
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
DASHBOARD    = OUTPUTS_DIR / "dashboard.html"

STEP_NAMES = {1: "Preprocessing", 2: "Epochs", 3: "ICA", 4: "Specparam"}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_status() -> dict:
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}


def load_ica_log() -> dict:
    """Returns {subject_id: {n_auto, n_manual, bad_channels, status}}."""
    log_path = OUTPUTS_DIR / "ica_cleaning_master_log.csv"
    result = {}
    if not log_path.exists():
        return result
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = str(row.get("subject_id", "")).strip()
            if not sub:
                continue
            auto_raw = row.get("auto_detected", "").strip()
            manual_raw = row.get("manual_exclude", "").strip()
            n_auto   = len([x for x in auto_raw.split(",")   if x.strip()]) if auto_raw   else 0
            n_manual = len([x for x in manual_raw.split(",") if x.strip()]) if manual_raw else 0
            result[sub] = {
                "n_auto":   n_auto,
                "n_manual": n_manual,
                "n_removed": n_manual if n_manual else n_auto,
                "bad_channels": row.get("bad_channels", "").strip(),
                "ica_status": row.get("status", "").strip(),
            }
    return result


def load_specparam_metrics(subject_id: str) -> dict:
    """Returns mean R² and median exponent from the specparam CSV."""
    # Try both naming variants
    candidates = [
        OUTPUTS_DIR / "specparam" / f"sub-{subject_id}" / f"sub-{subject_id}-specparam.csv",
        OUTPUTS_DIR / "specparam" / f"sub-{subject_id}" / f"sub-{subject_id}-specparam (1).csv",
    ]
    for path in candidates:
        if path.exists():
            r2_vals, exp_vals = [], []
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        r2_vals.append(float(row["gof_rsquared"]))
                        exp_vals.append(float(row["exponent"]))
                    except (KeyError, ValueError):
                        pass
            if r2_vals:
                return {
                    "mean_r2":      round(sum(r2_vals) / len(r2_vals), 3),
                    "median_exp":   round(sorted(exp_vals)[len(exp_vals) // 2], 2),
                    "n_rows":       len(r2_vals),
                }
    return {}


def find_report(subject_id: str) -> str | None:
    """Return a relative path to the best available HTML report for this subject."""
    report_dir = OUTPUTS_DIR / "reports" / f"sub-{subject_id}"
    if not report_dir.exists():
        return None
    # Prefer the manual (post-ICA) report, fall back to auto report
    for suffix in ["_report_manual_specparam.html", "_report_manual.html", "_report.html"]:
        candidate = report_dir / f"sub-{subject_id}{suffix}"
        if candidate.exists():
            return str(candidate.relative_to(OUTPUTS_DIR))
    # If none of the known names match, just take any html
    htmls = sorted(report_dir.glob("*.html"))
    if htmls:
        return str(htmls[-1].relative_to(OUTPUTS_DIR))
    return None


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_STATUS_BADGE = {
    "success": ('<span class="badge success">done</span>', "success"),
    "failed":  ('<span class="badge failed">FAILED</span>', "failed"),
    "not_run": ('<span class="badge not-run">—</span>', "not-run"),
}


def status_cell(status_entry: dict) -> tuple[str, str]:
    """Return (html_badge, css_class) for a step entry."""
    s = status_entry.get("status", "not_run") if status_entry else "not_run"
    badge, css = _STATUS_BADGE.get(s, ('<span class="badge not-run">—</span>', "not-run"))
    # Add timestamp tooltip if available
    ts = status_entry.get("completed_at", "") if status_entry else ""
    dur = status_entry.get("duration_s", "") if status_entry else ""
    title = ""
    if ts:
        title = f' title="{ts}  ({dur}s)"'
    return f'<td class="step-cell {css}"{title}>{badge}</td>', css


def fmt_error(entry: dict) -> str:
    err = entry.get("error", "")
    if not err:
        return ""
    short = err[:120] + ("…" if len(err) > 120 else "")
    return f'<div class="error-msg">{short}</div>'


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------

def build_dashboard():
    status      = load_status()
    ica_log     = load_ica_log()
    all_subjects = sorted(p.name.replace("sub-", "") for p in BIDS_ROOT.glob("sub-*"))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Count overall stats
    total = len(all_subjects)
    completed = sum(
        1 for s in all_subjects
        if all(
            status.get(s, {}).get(str(k), {}).get("status") == "success"
            for k in STEP_NAMES
        )
    )
    failed = sum(
        1 for s in all_subjects
        if any(
            status.get(s, {}).get(str(k), {}).get("status") == "failed"
            for k in STEP_NAMES
        )
    )
    not_started = sum(1 for s in all_subjects if s not in status)

    # Build table rows
    rows_html = []
    for sub in all_subjects:
        sub_status = status.get(sub, {})
        ica_info = ica_log.get(sub, {})
        spec = load_specparam_metrics(sub)
        report_path = find_report(sub)

        # Subject label + report link
        label = f"sub-{sub}"
        if report_path:
            label = f'<a href="{report_path}" target="_blank">sub-{sub}</a>'

        # Step cells
        cells = []
        any_failed = False
        for step_num in sorted(STEP_NAMES):
            entry = sub_status.get(str(step_num))
            cell_html, css = status_cell(entry)
            if css == "failed":
                any_failed = True
                err = fmt_error(entry or {})
                cell_html = f'<td class="step-cell {css}"><span class="badge failed">FAILED</span>{err}</td>'
            cells.append(cell_html)

        # ICA QC cell
        if ica_info:
            removed = ica_info["n_removed"]
            ica_status_label = ica_info.get("ica_status", "")
            manual_note = " (manual)" if ica_info.get("n_manual") else ""
            ica_cell = f'<td class="qc-cell">{removed} removed{manual_note}<br><small>{ica_status_label}</small></td>'
        elif sub_status.get("3", {}).get("status") == "success":
            ica_cell = '<td class="qc-cell"><small>done (no log)</small></td>'
        else:
            ica_cell = '<td class="qc-cell">—</td>'

        # Specparam QC cell
        if spec:
            r2_class = "good" if spec["mean_r2"] >= 0.95 else ("warn" if spec["mean_r2"] >= 0.85 else "poor")
            spec_cell = (
                f'<td class="qc-cell">'
                f'R²=<span class="metric {r2_class}">{spec["mean_r2"]}</span>&nbsp; '
                f'exp={spec["median_exp"]}'
                f'</td>'
            )
        elif sub_status.get("4", {}).get("status") == "success":
            spec_cell = '<td class="qc-cell"><small>done (no CSV)</small></td>'
        else:
            spec_cell = '<td class="qc-cell">—</td>'

        row_class = "row-failed" if any_failed else ""
        row = (
            f'<tr class="{row_class}">'
            f'<td class="sub-label">{label}</td>'
            + "".join(cells)
            + ica_cell
            + spec_cell
            + "</tr>"
        )
        rows_html.append(row)

    rows = "\n".join(rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PATHS Pipeline Dashboard</title>
<style>
  :root {{
    --success: #2e7d32; --success-bg: #e8f5e9;
    --failed:  #c62828; --failed-bg:  #ffebee;
    --warn:    #e65100; --warn-bg:    #fff3e0;
    --notrun:  #9e9e9e; --notrun-bg:  #f5f5f5;
    --good:    #2e7d32;
    --poor:    #c62828;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px; color: #212121; background: #fafafa; padding: 24px;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 20px; font-size: 0.85rem; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .stat-box {{
    padding: 10px 18px; border-radius: 6px; background: #fff;
    border: 1px solid #e0e0e0; text-align: center;
  }}
  .stat-box .num {{ font-size: 1.6rem; font-weight: 700; }}
  .stat-box .lbl {{ font-size: 0.75rem; color: #666; text-transform: uppercase; }}
  .stat-box.done   .num {{ color: var(--success); }}
  .stat-box.fail   .num {{ color: var(--failed); }}
  .stat-box.pend   .num {{ color: var(--notrun); }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  thead {{ background: #37474f; color: #fff; }}
  th {{ padding: 10px 12px; text-align: center; font-weight: 600;
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eeeeee; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover {{ background: #f5f5f5; }}
  tr.row-failed {{ background: var(--failed-bg); }}
  tr.row-failed:hover {{ background: #ffcdd2; }}
  .sub-label {{ font-weight: 600; white-space: nowrap; }}
  .sub-label a {{ color: #1565c0; text-decoration: none; }}
  .sub-label a:hover {{ text-decoration: underline; }}
  .step-cell {{ text-align: center; }}
  .step-cell.success {{ background: var(--success-bg); }}
  .step-cell.failed  {{ background: var(--failed-bg); }}
  .step-cell.not-run {{ color: var(--notrun); }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600;
  }}
  .badge.success {{ background: var(--success); color: #fff; }}
  .badge.failed  {{ background: var(--failed);  color: #fff; }}
  .badge.not-run {{ background: #e0e0e0;         color: #757575; }}
  .qc-cell {{ font-size: 0.82rem; color: #444; }}
  .error-msg {{
    font-size: 0.72rem; color: var(--failed); margin-top: 3px;
    max-width: 220px; word-break: break-word;
  }}
  .metric.good {{ color: var(--good); font-weight: 700; }}
  .metric.warn {{ color: var(--warn); font-weight: 700; }}
  .metric.poor {{ color: var(--poor); font-weight: 700; }}
  .footer {{ margin-top: 16px; font-size: 0.78rem; color: #9e9e9e; }}
</style>
</head>
<body>
<h1>PATHS Pipeline &mdash; Status Dashboard</h1>
<p class="subtitle">Last generated: {now} &nbsp;|&nbsp; Status file: {STATUS_FILE.relative_to(PROJECT_ROOT)}</p>

<div class="stats">
  <div class="stat-box"><div class="num">{total}</div><div class="lbl">Subjects</div></div>
  <div class="stat-box done"><div class="num">{completed}</div><div class="lbl">All steps done</div></div>
  <div class="stat-box fail"><div class="num">{failed}</div><div class="lbl">Has failed step</div></div>
  <div class="stat-box pend"><div class="num">{not_started}</div><div class="lbl">Not started</div></div>
</div>

<table>
  <thead>
    <tr>
      <th style="text-align:left">Subject</th>
      <th>Preprocessing</th>
      <th>Epochs</th>
      <th>ICA</th>
      <th>Specparam</th>
      <th>ICA&nbsp;QC</th>
      <th>Specparam&nbsp;QC</th>
    </tr>
  </thead>
  <tbody>
{rows}
  </tbody>
</table>

<p class="footer">
  Regenerate: <code>python templates/generate_dashboard.py</code> &nbsp;|&nbsp;
  R&sup2; colour: <span style="color:var(--good)">&#9632;</span>&ge;0.95 good &nbsp;
  <span style="color:var(--warn)">&#9632;</span>&ge;0.85 warn &nbsp;
  <span style="color:var(--poor)">&#9632;</span>&lt;0.85 poor
</p>
</body>
</html>
"""

    DASHBOARD.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {DASHBOARD}")
    print(f"Open in browser:      file://{DASHBOARD}")


if __name__ == "__main__":
    build_dashboard()
