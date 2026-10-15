
import json
import html
import webbrowser
import threading
import http.server
import socketserver
from pathlib import Path
from functools import partial

import pandas as pd


# ============================================================
# PATHS — KEEP THIS REVIEW SEPARATE
# ============================================================

PROJECT_DIR = Path(r"C:\Users\divya\whisper-india-project")
BASE_DIR = PROJECT_DIR / "results" / "baseline"

SAMPLE_CSV = BASE_DIR / "unflagged_random_sample_100.csv"
AUDIO_DIR = BASE_DIR / "unflagged_review_audio"
MANIFEST_CSV = AUDIO_DIR / "review_manifest.csv"

HTML_PATH = BASE_DIR / "unflagged_review_reviewer.html"
OUTPUT_CSV = BASE_DIR / "unflagged_random_sample_100_reviewed.csv"

PORT = 8766
STORAGE_KEY = "whisper_india_unflagged_review_v1"

LABELS = [
    ("HALLUCINATION", "HALLUCINATION — substantial content unsupported by the audio"),
    ("ORDINARY_ASR_ERROR", "ORDINARY ASR ERROR — incorrect, but broadly follows the speech"),
    ("WRONG_LANGUAGE", "WRONG LANGUAGE — transcription is in an inappropriate language"),
    ("VALID", "VALID — prediction is supported by the audio"),
    ("UNCERTAIN", "UNCERTAIN — cannot confidently decide"),
]


def main():
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(f"Missing sample CSV: {SAMPLE_CSV}")
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Missing audio manifest: {MANIFEST_CSV}")

    sample = pd.read_csv(SAMPLE_CSV).fillna("")
    manifest = pd.read_csv(MANIFEST_CSV).fillna("")

    # Match the sample and manifest by stable IDs, not row order.
    sample["eval_id"] = sample["eval_id"].astype(int)
    manifest["eval_id"] = manifest["eval_id"].astype(int)

    df = sample.merge(
        manifest[["eval_id", "dataset", "audio_file", "sample_rate", "audio_duration_seconds"]],
        on=["eval_id", "dataset"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )

    if df["audio_file"].isna().any() or (df["audio_file"] == "").any():
        raise ValueError("Some sample rows have no matching extracted audio.")

    missing_audio = [
        str(audio_file)
        for audio_file in df["audio_file"]
        if not (AUDIO_DIR / str(audio_file)).exists()
    ]
    if missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} audio files are missing. First: {missing_audio[0]}"
        )

    records = []
    case_blocks = []

    for index, row in df.reset_index(drop=True).iterrows():
        audio_filename = Path(str(row["audio_file"])).name

        record = {
            "index": int(index),
            "eval_id": int(row["eval_id"]),
            "dataset": str(row["dataset"]),
            "state": str(row.get("state", "")),
            "district": str(row.get("district", "")),
            "detected_language": str(row.get("detected_language", "")),
            "audio_duration": str(row.get("audio_duration", row.get("audio_duration_seconds", ""))),
            "reference": str(row.get("reference", "")),
            "prediction": str(row.get("prediction", "")),
            "audio_filename": audio_filename,
        }
        records.append(record)

        label_options = "\n".join(
            f"""
            <label class="option">
                <input type="radio" name="label-{index}"
                       value="{code}" onchange="saveCase({index})">
                {html.escape(label_text)}
            </label>
            """
            for code, label_text in LABELS
        )

        case_blocks.append(f"""
        <section class="case" id="case-{index}">
            <h2>Case {index + 1} of {len(df)}</h2>
            <div class="metadata">
                <b>Dataset:</b> {html.escape(record["dataset"])}
                &nbsp; | &nbsp; <b>Eval ID:</b> {record["eval_id"]}
                &nbsp; | &nbsp; <b>State/background:</b> {html.escape(record["state"])}
                &nbsp; | &nbsp; <b>District:</b> {html.escape(record["district"])}
                <br><b>Automatically detected language:</b>
                {html.escape(record["detected_language"])}
                &nbsp; | &nbsp; <b>Duration:</b>
                {html.escape(record["audio_duration"])} seconds
            </div>

            <audio controls preload="none">
                <source src="/audio/{html.escape(audio_filename)}" type="audio/wav">
                Your browser does not support audio playback.
            </audio>

            <h3>Reference transcript</h3>
            <div class="reference">{html.escape(record["reference"])}</div>

            <h3>Whisper prediction</h3>
            <div class="prediction">{html.escape(record["prediction"])}</div>

            <div class="label-box">
                <h3>Human assessment</h3>
                <p>Listen to the audio before deciding. These recordings were randomly sampled
                from outputs the automatic detector did not flag. Do not assume they are valid
                just because they were not flagged.</p>
                {label_options}
                <label class="notes-label">Reviewer notes (optional)
                    <textarea id="notes-{index}" rows="2"
                    oninput="saveCase({index})"
                    placeholder="What did you hear? Why is this label appropriate?"></textarea>
                </label>
                <div id="status-{index}" class="status">Not labeled</div>
            </div>
        </section>
        """)

    records_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")

    labels_json = json.dumps(
        {code: text for code, text in LABELS},
        ensure_ascii=False,
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unflagged Whisper Output Review</title>
<style>
body {{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#f5f6f8;color:#222;}}
header,.case {{background:white;padding:22px;margin-bottom:22px;border-radius:10px;box-shadow:0 2px 7px #00000012;}}
.controls {{position:sticky;top:0;background:#fff;padding:14px;border-radius:8px;z-index:2;box-shadow:0 2px 7px #0002;margin:16px 0;}}
button {{padding:10px 14px;margin:5px 8px 5px 0;cursor:pointer;}}
.metadata {{background:#f0f1f3;padding:12px;border-radius:6px;line-height:1.8;margin:12px 0;}}
audio {{width:100%;margin:10px 0;}}
.reference,.prediction {{padding:13px;border-radius:6px;line-height:1.6;overflow-wrap:anywhere;}}
.reference {{background:#eaf6ea;}}
.prediction {{background:#fff0f0;}}
.label-box {{background:#f0f4ff;padding:15px;border-radius:8px;margin-top:18px;}}
.option {{display:block;margin:12px 0;line-height:1.5;}}
.notes-label {{display:block;margin-top:14px;}}
textarea {{display:block;width:98%;box-sizing:border-box;margin-top:6px;padding:8px;}}
.status {{margin-top:10px;font-weight:bold;color:#225522;}}
small {{color:#555;}}
</style>
</head>
<body>
<header>
<h1>Unflagged Whisper Output — Human Audit</h1>
<p><b>Purpose:</b> estimate whether the automatic detector missed hallucination-like failures
in a random sample of recordings it did not flag.</p>
<p>Listen to each clip, compare the reference and prediction, and select one label.
A strange-looking transcript alone is not sufficient evidence of hallucination.</p>
<p><b>Important:</b> these are human judgments, not automatically verified ground truth.
If the audio or reference is unclear, select UNCERTAIN.</p>
</header>

<div class="controls">
<div id="progress">Labeled: 0 / {len(records)}</div>
<button onclick="downloadCSV()">Download reviewed CSV</button>
<button onclick="downloadProgress()">Download progress backup</button>
<small>Labels autosave in this browser. Download the reviewed CSV before closing.</small>
</div>

{"".join(case_blocks)}

<script>
const records = {records_json};
const LABEL_TEXT = {labels_json};
const STORAGE_KEY = "{STORAGE_KEY}";
let reviewData = {{}};
try {{ reviewData = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
catch (e) {{ reviewData = {{}}; }}

function saveCase(index) {{
    const selected = document.querySelector('input[name="label-' + index + '"]:checked');
    const notes = document.getElementById("notes-" + index).value;
    reviewData[index] = {{label: selected ? selected.value : "", notes: notes}};
    localStorage.setItem(STORAGE_KEY, JSON.stringify(reviewData));
    updateStatus(index);
    updateProgress();
}}

function updateStatus(index) {{
    const data = reviewData[index] || {{}};
    document.getElementById("status-" + index).textContent =
        data.label ? "Saved: " + data.label : "Not labeled";
}}

function updateProgress() {{
    let count = 0;
    for (const record of records) {{
        if (reviewData[record.index] && reviewData[record.index].label) count++;
    }}
    document.getElementById("progress").textContent =
        "Labeled: " + count + " / " + records.length;
}}

function csvEscape(value) {{
    const s = value === null || value === undefined ? "" : String(value);
    return '"' + s.replace(/"/g, '""') + '"';
}}

function makeCSV() {{
    const headers = [
        "eval_id", "dataset", "state", "district", "detected_language",
        "audio_duration", "reference", "prediction", "audio_filename",
        "human_label", "reviewer_notes"
    ];
    const rows = [headers];
    for (const r of records) {{
        const saved = reviewData[r.index] || {{}};
        rows.push([
            r.eval_id, r.dataset, r.state, r.district, r.detected_language,
            r.audio_duration, r.reference, r.prediction, r.audio_filename,
            saved.label || "", saved.notes || ""
        ]);
    }}
    return rows.map(row => row.map(csvEscape).join(",")).join("\\r\\n");
}}

function downloadCSV() {{
    const blob = new Blob(["\\uFEFF" + makeCSV()], {{type:"text/csv;charset=utf-8;"}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "unflagged_random_sample_100_reviewed.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}}

function downloadProgress() {{
    const blob = new Blob([JSON.stringify(reviewData, null, 2)], {{type:"application/json"}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "unflagged_review_progress_backup.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}}

function loadSavedReview() {{
    for (const [index, data] of Object.entries(reviewData)) {{
        if (data.label) {{
            const radio = document.querySelector(
                'input[name="label-' + index + '"][value="' + data.label + '"]'
            );
            if (radio) radio.checked = true;
        }}
        const notes = document.getElementById("notes-" + index);
        if (notes) notes.value = data.notes || "";
        updateStatus(Number(index));
    }}
    updateProgress();
}}
loadSavedReview();
</script>
</body>
</html>
"""

    HTML_PATH.write_text(page, encoding="utf-8")
    print(f"Reviewer HTML created: {HTML_PATH}")
    print(f"Cases: {len(records)}")
    print("Labels are stored in browser localStorage with a dedicated key.")
    print(f"CSV export filename: {OUTPUT_CSV.name}")
    print("Do not clear browser storage before downloading your review CSV.")

    class ReviewHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(AUDIO_DIR), **kwargs)

        def do_GET(self):
            if self.path.startswith("/audio/"):
                self.path = "/" + self.path[len("/audio/"):]
                return super().do_GET()
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page.encode("utf-8"))
                return
            self.send_error(404)

        def log_message(self, format, *args):
            pass

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        server = ReusableTCPServer(("127.0.0.1", PORT), ReviewHandler)
    except OSError as exc:
        raise RuntimeError(
            f"Could not start local reviewer on port {PORT}. "
            "Close any old reviewer using that port and rerun. Details: " + str(exc)
        ) from exc

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Opening reviewer: {url}")
    webbrowser.open(url)
    print("Keep this terminal open while reviewing. Press Ctrl+C to stop the server.")
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()
        print("Reviewer server stopped.")


if __name__ == "__main__":
    main()
