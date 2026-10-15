from pathlib import Path
import re, json
import pandas as pd

RESULTS = Path.cwd() / "results" / "lora_generic_hindi"
BASE_FILE = RESULTS / "whisper_small_test_predictions.csv"
LORA_FILE = RESULTS / "whisper_small_lora_test_predictions.csv"
OUT = RESULTS / "hallucination_screening"
OUT.mkdir(parents=True, exist_ok=True)

required = {"record_id","source_shard","source_row","sp_id","state","district","reference","prediction","status"}

def load(path, name):
    d = pd.read_csv(path)
    missing = required - set(d.columns)
    if missing: raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    d = d[d.status.astype(str).str.lower().eq("success")].copy()
    d["reference"] = d.reference.fillna("").astype(str)
    d["prediction"] = d.prediction.fillna("").astype(str)
    d["key"] = d.source_shard.astype(str) + "::" + d.source_row.astype(str)
    if d.key.duplicated().any(): raise ValueError(f"Duplicate recording keys in {path.name}")
    return d.set_index("key", drop=False)

def tokens(s):
    return re.findall(r"[^\W_]+", str(s).lower(), flags=re.UNICODE)

def chars(s):
    return "".join(c.lower() for c in str(s) if c.isalnum())

def edit_distance(a,b):
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def repetition(s):
    t=tokens(s)
    if not t: return 0,0.0
    repeats=sum(t[i] in t[max(0,i-10):i] for i in range(len(t)))
    hits=0
    for n in (2,3):
        grams=[tuple(t[i:i+n]) for i in range(max(0,len(t)-n+1))]
        hits += sum(g in grams[:i] for i,g in enumerate(grams))
    return hits, repeats/len(t)

base=load(BASE_FILE,"baseline")
lora=load(LORA_FILE,"lora")
if set(base.index)!=set(lora.index):
    raise ValueError("Baseline and LoRA do not contain exactly the same successful recording keys.")
keys=sorted(base.index)
if base.loc[keys,"reference"].tolist()!=lora.loc[keys,"reference"].tolist():
    raise ValueError("Reference transcripts differ between baseline and LoRA.")
rows=[]
for k in keys:
    b,l=base.loc[k],lora.loc[k]
    r=str(b.reference)
    row={"key":k,"record_id":b.record_id,"source_shard":b.source_shard,"source_row":b.source_row,
         "sp_id":b.sp_id,"state":b.state,"district":b.district,"reference":r,
         "baseline_prediction":b.prediction,"lora_prediction":l.prediction}
    rt,rc=tokens(r),chars(r)
    for name,p in (("baseline",b.prediction),("lora",l.prediction)):
        pt,pc=tokens(p),chars(p)
        row[f"{name}_wer"]=edit_distance(rt,pt)/max(1,len(rt))
        row[f"{name}_cer"]=edit_distance(list(rc),list(pc))/max(1,len(rc))
        row[f"{name}_reference_words"]=len(rt)
        row[f"{name}_prediction_words"]=len(pt)
        row[f"{name}_length_ratio"]=len(pt)/max(1,len(rt))
        hits,frac=repetition(p)
        row[f"{name}_repeat_ngram_hits"]=hits
        row[f"{name}_repeated_token_fraction"]=frac
    # These flags are screening heuristics, not confirmed hallucinations.
    row["baseline_excess_length_flag"]=row["baseline_length_ratio"]>=1.8 and row["baseline_prediction_words"]>=12
    row["lora_excess_length_flag"]=row["lora_length_ratio"]>=1.8 and row["lora_prediction_words"]>=12
    row["baseline_repetition_flag"]=row["baseline_repeat_ngram_hits"]>0 or row["baseline_repeated_token_fraction"]>=0.25
    row["lora_repetition_flag"]=row["lora_repeat_ngram_hits"]>0 or row["lora_repeated_token_fraction"]>=0.25
    row["baseline_wer_minus_lora_wer"]=row["baseline_wer"]-row["lora_wer"]
    rows.append(row)

df=pd.DataFrame(rows)
df.to_csv(OUT/"recording_level_hallucination_screen.csv",index=False,encoding="utf-8-sig")
summary={}
for m in ("baseline","lora"):
    summary[m]={
      "recordings":int(len(df)),
      "excess_length_flag_count":int(df[f"{m}_excess_length_flag"].sum()),
      "excess_length_flag_rate":float(df[f"{m}_excess_length_flag"].mean()),
      "repetition_flag_count":int(df[f"{m}_repetition_flag"].sum()),
      "repetition_flag_rate":float(df[f"{m}_repetition_flag"].mean()),
      "mean_prediction_reference_word_ratio":float(df[f"{m}_length_ratio"].mean()),
      "mean_repeated_token_fraction":float(df[f"{m}_repeated_token_fraction"].mean()),
      "mean_recording_wer":float(df[f"{m}_wer"].mean()),
      "mean_recording_cer":float(df[f"{m}_cer"].mean())}
summary["paired_change_lora_minus_baseline"]={
 "excess_length_flag_rate":summary["lora"]["excess_length_flag_rate"]-summary["baseline"]["excess_length_flag_rate"],
 "repetition_flag_rate":summary["lora"]["repetition_flag_rate"]-summary["baseline"]["repetition_flag_rate"],
 "mean_prediction_reference_word_ratio":summary["lora"]["mean_prediction_reference_word_ratio"]-summary["baseline"]["mean_prediction_reference_word_ratio"]}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

cand=df[df[["baseline_excess_length_flag","lora_excess_length_flag","baseline_repetition_flag","lora_repetition_flag"]].any(axis=1) |
        (df["baseline_wer_minus_lora_wer"].abs()>=0.75)].copy()
cand["priority_score"]=cand[["baseline_excess_length_flag","lora_excess_length_flag","baseline_repetition_flag","lora_repetition_flag"]].astype(int).sum(axis=1)+(cand["baseline_wer_minus_lora_wer"].abs()>=0.75).astype(int)
cand=cand.sort_values(["priority_score","baseline_wer_minus_lora_wer"],ascending=[False,False]).head(30)
cols=["key","sp_id","state","district","reference","baseline_prediction","lora_prediction","baseline_wer","lora_wer",
      "baseline_length_ratio","lora_length_ratio","baseline_repeat_ngram_hits","lora_repeat_ngram_hits",
      "baseline_excess_length_flag","lora_excess_length_flag","baseline_repetition_flag","lora_repetition_flag"]
cand[cols].to_csv(OUT/"manual_review_candidates.csv",index=False,encoding="utf-8-sig")
print("SCREENING COMPLETE — flags are candidates, NOT confirmed hallucinations.")
print("Matched recordings:",len(df))
print(json.dumps(summary,ensure_ascii=False,indent=2))
print("Saved folder:",OUT)
print("Open manual_review_candidates.csv for examples to check against audio.")
