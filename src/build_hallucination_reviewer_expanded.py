import os
import io
import pandas as pd
import webbrowser
import http.server
import socketserver
import threading
import json
import html
import soundfile as sf
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = r"C:\Users\divya\whisper-india-project"

CANDIDATES_CSV = os.path.join(
    PROJECT_DIR,
    "results",
    "baseline",
    "hallucination_candidates_expanded.csv"
)

REVIEW_CSV = os.path.join(
    PROJECT_DIR,
    "results",
    "baseline",
    "hallucination_review_expanded.csv"
)

AUDIO_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "baseline",
    "review_audio_expanded"
)

HTML_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "baseline",
    "hallucination_reviewer_expanded.html"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    AUDIO_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CANDIDATES
# ============================================================

df = pd.read_csv(
    CANDIDATES_CSV
)

print("=" * 60)
print("BUILDING EXPANDED HALLUCINATION HUMAN REVIEWER")
print("=" * 60)

print(
    f"Candidates loaded: {len(df)}"
)


# ============================================================
# CREATE REVIEW CSV
# ============================================================

if os.path.exists(REVIEW_CSV):

    existing_review = pd.read_csv(
        REVIEW_CSV
    )

    print(
        f"Existing review file found: "
        f"{len(existing_review)} records"
    )

    # Preserve any labels already entered.
    if (
        "human_label" in existing_review.columns
        and "reviewer_notes" in existing_review.columns
    ):

        label_map = dict(
            zip(
                existing_review["eval_id"],
                existing_review["human_label"]
            )
        )

        notes_map = dict(
            zip(
                existing_review["eval_id"],
                existing_review["reviewer_notes"]
            )
        )

        df["human_label"] = (
            df["eval_id"]
            .map(label_map)
            .fillna("")
        )

        df["reviewer_notes"] = (
            df["eval_id"]
            .map(notes_map)
            .fillna("")
        )

    else:

        df["human_label"] = ""
        df["reviewer_notes"] = ""

else:

    df["human_label"] = ""
    df["reviewer_notes"] = ""


# ============================================================
# AUDIO EXTRACTION
# ============================================================

print()
print("=" * 60)
print("PREPARING AUDIO REVIEW CLIPS")
print("=" * 60)


def get_parquet_path(dataset, source_shard):

    if dataset == "svarah":

        return (
            Path(PROJECT_DIR)
            / "data"
            / "raw"
            / "svarah"
            / "data"
            / source_shard
        )

    elif dataset == "lahaja":

        return (
            Path(PROJECT_DIR)
            / "data"
            / "raw"
            / "lahaja"
            / "data"
            / source_shard
        )

    else:

        raise ValueError(
            f"Unknown dataset: {dataset}"
        )


def extract_audio_from_parquet(
    dataset,
    source_shard,
    source_row
):

    parquet_file = get_parquet_path(
        dataset,
        source_shard
    )

    source_df = pd.read_parquet(
        parquet_file
    )

    audio_object = source_df.iloc[
        int(source_row)
    ]["audio_filepath"]

    if not isinstance(
        audio_object,
        dict
    ):

        raise TypeError(
            "Expected embedded audio dictionary, "
            f"got {type(audio_object)}"
        )

    audio_bytes = audio_object["bytes"]

    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    return audio, sample_rate


# ============================================================
# CREATE AUDIO FILES
# ============================================================

audio_failures = []

for i, row in df.iterrows():

    dataset = str(
        row["dataset"]
    )

    eval_id = int(
        row["eval_id"]
    )

    audio_filename = (
        f"{i:03d}_{dataset}_{eval_id}.wav"
    )

    audio_path = os.path.join(
        AUDIO_DIR,
        audio_filename
    )

    # --------------------------------------------------------
    # Do not recreate existing clips
    # --------------------------------------------------------

    if os.path.exists(audio_path):

        print(
            f"[{i + 1}/{len(df)}] "
            f"EXISTS | {audio_filename}"
        )

        continue

    try:

        audio, sample_rate = (
            extract_audio_from_parquet(
                dataset,
                row["source_shard"],
                row["source_row"]
            )
        )

        sf.write(
            audio_path,
            audio,
            sample_rate
        )

        print(
            f"[{i + 1}/{len(df)}] "
            f"CREATED | {audio_filename}"
        )

    except Exception as e:

        print(
            f"[{i + 1}/{len(df)}] "
            f"ERROR | {audio_filename} | "
            f"{repr(e)}"
        )

        audio_failures.append(
            {
                "eval_id": eval_id,
                "dataset": dataset,
                "error": repr(e)
            }
        )


print()
print(
    f"Audio clips ready: "
    f"{len(df) - len(audio_failures)}"
)

print(
    f"Audio extraction failures: "
    f"{len(audio_failures)}"
)


# ============================================================
# SAVE INITIAL REVIEW CSV
# ============================================================

df.to_csv(
    REVIEW_CSV,
    index=False,
    encoding="utf-8"
)

print()
print(
    f"Review CSV saved: {REVIEW_CSV}"
)


# ============================================================
# PREPARE DATA FOR JAVASCRIPT
# ============================================================

records = []

for i, row in df.iterrows():

    dataset = str(
        row["dataset"]
    )

    eval_id = int(
        row["eval_id"]
    )

    audio_filename = (
        f"{i:03d}_{dataset}_{eval_id}.wav"
    )

    records.append(
        {
            "index": int(i),
            "eval_id": eval_id,
            "dataset": dataset,
            "state": str(row["state"]),
            "district": str(row["district"]),
            "detected_language": str(
                row["detected_language"]
            ),
            "audio_duration": str(
                row["audio_duration"]
            ),
            "reference": str(
                row["reference"]
            ),
            "prediction": str(
                row["prediction"]
            ),
            "candidate_reason": str(
                row["candidate_reason"]
            ),
            "audio_filename": audio_filename
        }
    )


records_json = json.dumps(
    records,
    ensure_ascii=False
)


# ============================================================
# CREATE CASE HTML
# ============================================================

case_blocks = []

for item in records:

    i = item["index"]

    case_blocks.append(
        f"""
        <div class="case" id="case-{i}">

            <h2>
                Case {i + 1} / {len(records)}
            </h2>

            <div class="metadata">

                <b>Dataset:</b>
                {html.escape(item["dataset"])}

                &nbsp;&nbsp;

                <b>Eval ID:</b>
                {item["eval_id"]}

                &nbsp;&nbsp;

                <b>State:</b>
                {html.escape(item["state"])}

                &nbsp;&nbsp;

                <b>District:</b>
                {html.escape(item["district"])}

                &nbsp;&nbsp;

                <b>Detected language:</b>
                {html.escape(item["detected_language"])}

                &nbsp;&nbsp;

                <b>Duration:</b>
                {html.escape(item["audio_duration"])}s

            </div>


            <div class="audio">

                <audio controls preload="none">

                    <source
                        src="/audio/{html.escape(item["audio_filename"])}"
                        type="audio/wav">

                </audio>

            </div>


            <div class="text-box">

                <h3>
                    REFERENCE TRANSCRIPT
                </h3>

                <div class="reference">

                    {html.escape(item["reference"])}

                </div>

            </div>


            <div class="text-box">

                <h3>
                    WHISPER PREDICTION
                </h3>

                <div class="prediction">

                    {html.escape(item["prediction"])}

                </div>

            </div>


            <div class="text-box">

                <h3>
                    WHY THIS WAS FLAGGED
                </h3>

                <div>

                    {html.escape(item["candidate_reason"])}

                </div>

            </div>


            <div class="label-box">

                <h3>
                    HUMAN LABEL
                </h3>


                <label>

                    <input
                        type="radio"
                        name="label-{i}"
                        value="HALLUCINATION"
                        onchange="saveCase({i})">

                    HALLUCINATION

                </label>


                <label>

                    <input
                        type="radio"
                        name="label-{i}"
                        value="ORDINARY_ASR_ERROR"
                        onchange="saveCase({i})">

                    ORDINARY_ASR_ERROR

                </label>


                <label>

                    <input
                        type="radio"
                        name="label-{i}"
                        value="WRONG_LANGUAGE"
                        onchange="saveCase({i})">

                    WRONG_LANGUAGE

                </label>


                <label>

                    <input
                        type="radio"
                        name="label-{i}"
                        value="VALID"
                        onchange="saveCase({i})">

                    VALID

                </label>


                <label>

                    <input
                        type="radio"
                        name="label-{i}"
                        value="UNCERTAIN"
                        onchange="saveCase({i})">

                    UNCERTAIN

                </label>


                <br><br>


                <label>

                    Reviewer notes:

                    <br>

                    <textarea
                        id="notes-{i}"
                        rows="3"
                        placeholder="Optional notes..."
                        oninput="saveCase({i})"></textarea>

                </label>


                <div
                    class="saved-status"
                    id="status-{i}">

                    Not labeled

                </div>

            </div>

        </div>
        """
    )


# ============================================================
# HTML
# ============================================================

html_content = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
Whisper Expanded Hallucination Human Review
</title>


<style>

body {{

    font-family: Arial, sans-serif;

    margin: 30px;

    background: #f5f5f5;

}}

h1 {{

    margin-bottom: 5px;

}}


.instructions {{

    background: white;

    padding: 20px;

    border-radius: 10px;

    margin-bottom: 25px;

}}


.controls {{

    position: sticky;

    top: 0;

    z-index: 100;

    background: white;

    padding: 15px;

    margin-bottom: 20px;

    border-radius: 10px;

    box-shadow: 0 2px 8px rgba(0,0,0,0.15);

}}


.case {{

    background: white;

    padding: 25px;

    margin-bottom: 30px;

    border-radius: 10px;

    box-shadow: 0 2px 8px rgba(0,0,0,0.1);

}}


.metadata {{

    background: #f0f0f0;

    padding: 10px;

    border-radius: 5px;

    margin-bottom: 20px;

}}


.audio {{

    margin: 20px 0;

}}


.text-box {{

    margin-top: 15px;

}}


.text-box h3 {{

    margin-bottom: 5px;

}}


.reference {{

    background: #eaf6ea;

    padding: 12px;

    border-radius: 5px;

    font-size: 17px;

}}


.prediction {{

    background: #fff0f0;

    padding: 12px;

    border-radius: 5px;

    font-size: 17px;

    word-break: break-word;

}}


.label-box {{

    margin-top: 20px;

    padding: 15px;

    background: #f0f4ff;

    border-radius: 8px;

}}


.label-box label {{

    margin-right: 18px;

    display: inline-block;

    margin-bottom: 10px;

}}


textarea {{

    width: 95%;

    padding: 8px;

}}


button {{

    padding: 12px 20px;

    font-size: 16px;

    cursor: pointer;

    margin-right: 10px;

}}


.saved-status {{

    margin-top: 10px;

    font-weight: bold;

}}


.progress {{

    font-size: 17px;

    font-weight: bold;

}}

</style>

</head>


<body>


<h1>
Whisper Expanded Hallucination Human Review
</h1>


<div class="instructions">

<b>Instructions</b>

<p>

Listen to the audio first.

Then compare the audio with the
reference transcript and Whisper prediction.

</p>


<ul>

<li>

<b>HALLUCINATION</b> —
substantial generated content is unsupported
by the speech.

</li>


<li>

<b>ORDINARY_ASR_ERROR</b> —
incorrect transcription, but output broadly
follows the speech.

</li>


<li>

<b>WRONG_LANGUAGE</b> —
output is in an inappropriate language,
without sufficient evidence to classify it
as hallucination.

</li>


<li>

<b>VALID</b> —
the unusual or repetitive output is actually
supported by the audio.

</li>


<li>

<b>UNCERTAIN</b> —
you cannot confidently determine the category.

</li>

</ul>


<p>

<b>Important:</b>

Do not classify something as hallucination
merely because the prediction looks strange.

Listen to the audio.

</p>


</div>


<div class="controls">

    <div
        class="progress"
        id="progress">

        Labeled: 0 / {len(records)}

    </div>


    <br>


    <button onclick="downloadCSV()">

        Download Review CSV

    </button>


    <button onclick="clearSavedReview()">

        Clear Saved Review

    </button>

</div>


{"".join(case_blocks)}


<script>

const records = {records_json};


const STORAGE_KEY =
    "whisper_india_hallucination_review_v3";


let reviewData =
    JSON.parse(
        localStorage.getItem(STORAGE_KEY) || "{{}}"
    );


// ============================================================
// SAVE ONE CASE
// ============================================================

function saveCase(index) {{

    const selected =
        document.querySelector(
            'input[name="label-' +
            index +
            '"]:checked'
        );


    const notes =
        document.getElementById(
            "notes-" + index
        ).value;


    reviewData[index] = {{

        label:
            selected
            ? selected.value
            : "",

        notes:
            notes

    }};


    localStorage.setItem(

        STORAGE_KEY,

        JSON.stringify(reviewData)

    );


    updateStatus(index);

    updateProgress();

}}


// ============================================================
// LOAD SAVED REVIEW
// ============================================================

function loadSavedReview(){{

    for (
        const [index, data]
        of Object.entries(reviewData)
    ){{

        if (data.label){{

            const radio =
                document.querySelector(
                    'input[name="label-' +
                    index +
                    '"][value="' +
                    data.label +
                    '"]'
                );


            if (radio){{

                radio.checked = true;

            }}

        }}


        const notes =
            document.getElementById(
                "notes-" + index
            );


        if (notes){{

            notes.value =
                data.notes || "";

        }}


        updateStatus(index);

    }}


    updateProgress();

}}


// ============================================================
// STATUS
// ============================================================

function updateStatus(index){{

    const status =
        document.getElementById(
            "status-" + index
        );


    const data =
        reviewData[index];


    if (
        data &&
        data.label
    ){{

        status.textContent =
            "Saved: " +
            data.label;

    }}
    else{{

        status.textContent =
            "Not labeled";

    }}

}}


// ============================================================
// PROGRESS
// ============================================================

function updateProgress(){{

    let count = 0;


    for (
        const record
        of records
    ){{

        const data =
            reviewData[
                record.index
            ];


        if (
            data &&
            data.label
        ){{

            count++;

        }}

    }}


    document.getElementById(
        "progress"
    ).textContent =
        "Labeled: " +
        count +
        " / " +
        records.length;

}}


// ============================================================
// CSV ESCAPING
// ============================================================

function csvEscape(value){{

    value =
        value === null ||
        value === undefined

        ? ""

        : String(value);


    return '"' +
        value.replace(
            /"/g,
            '""'
        ) +
        '"';

}}


// ============================================================
// DOWNLOAD CSV
// ============================================================

function downloadCSV(){{

    const headers = [

        "eval_id",
        "dataset",
        "state",
        "district",
        "detected_language",
        "audio_duration",
        "reference",
        "prediction",
        "candidate_reason",
        "human_label",
        "reviewer_notes"

    ];


    let csv =
        headers
        .map(csvEscape)
        .join(",") +
        "\\n";


    for (
        const record
        of records
    ){{

        const saved =
            reviewData[
                record.index
            ] || {{}};


        const row = [

            record.eval_id,

            record.dataset,

            record.state,

            record.district,

            record.detected_language,

            record.audio_duration,

            record.reference,

            record.prediction,

            record.candidate_reason,

            saved.label || "",

            saved.notes || ""

        ];


        csv +=

            row
            .map(csvEscape)
            .join(",") +

            "\\n";

    }}


    const blob =
        new Blob(

            [csv],

            {{
                type:
                    "text/csv;charset=utf-8;"
            }}

        );


    const url =
        URL.createObjectURL(blob);


    const a =
        document.createElement(
            "a"
        );


    a.href = url;


    a.download =
        "hallucination_review_expanded_completed.csv";


    document.body.appendChild(a);


    a.click();


    document.body.removeChild(a);


    URL.revokeObjectURL(url);

}}


// ============================================================
// CLEAR REVIEW
// ============================================================

function clearSavedReview(){{

    const answer =
        confirm(
            "Are you sure you want to clear all saved labels?"
        );


    if (!answer){{

        return;

    }}


    localStorage.removeItem(
        STORAGE_KEY
    );


    location.reload();

}}


// ============================================================
// START
// ============================================================

loadSavedReview();

</script>


</body>

</html>
"""


# ============================================================
# SAVE HTML
# ============================================================

with open(
    HTML_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        html_content
    )


print()
print(
    "Reviewer created:"
)

print(
    HTML_PATH
)


# ============================================================
# SERVER
# ============================================================

class CustomHandler(
    http.server.SimpleHTTPRequestHandler
):

    def translate_path(
        self,
        path
    ):

        if path.startswith(
            "/audio/"
        ):

            filename = (
                path[len("/audio/"):]
            )

            return os.path.join(
                AUDIO_DIR,
                filename
            )

        return super().translate_path(
            path
        )


PORT = 8765


httpd = socketserver.TCPServer(
    ("127.0.0.1", PORT),
    CustomHandler
)


url = (
    f"http://127.0.0.1:{PORT}"
    f"/results/baseline/"
    f"hallucination_reviewer_expanded.html"
)


print()
print("=" * 60)
print("REVIEWER SERVER STARTED")
print("=" * 60)

print(
    "Open:"
)

print(
    url
)

print()

print(
    "Keep this terminal open."
)

print(
    "Press Ctrl+C when finished."
)

print("=" * 60)


threading.Thread(
    target=httpd.serve_forever,
    daemon=True
).start()


webbrowser.open(
    url
)


try:

    while True:

        input()

except KeyboardInterrupt:

    print()

    print(
        "Stopping reviewer server..."
    )

    httpd.shutdown()