# MythBench v0.4 — Inference Script v5 (Kaggle T4 x2)
# Models: Gemma-2B + Phi-2 (with pre-load config fix) + Qwen-1.8B
# Goal: CMO across 3 models on 10 misconceptions + 10 controls

# ─────────────────────────────────────────────
# CELL 1: Install
# ─────────────────────────────────────────────
# !pip install -q transformers accelerate

# ─────────────────────────────────────────────
# CELL 2: Imports + config
# ─────────────────────────────────────────────
import json, re, csv, torch
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

MODELS = {
    "Gemma2B": "google/gemma-2b-it",
    "Qwen18B":  "Qwen/Qwen-1_8B-Chat",
    # "Phi2":  "microsoft/phi-2",   # uncomment if Qwen fails
}

BENCHMARK_PATH = "/kaggle/input/datasets/kevinsam77/mythbench-dataset/MythBench_v04.json"
OUTPUT_PATH    = "/kaggle/working/mythbench_results_v05.csv"

# ─────────────────────────────────────────────
# CELL 3: Pre-flight check
# ─────────────────────────────────────────────
from huggingface_hub import model_info
for name, path in MODELS.items():
    try:
        model_info(path, token=HF_TOKEN)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name} — {e}")

# ─────────────────────────────────────────────
# CELL 4: Load benchmark
# ─────────────────────────────────────────────
with open(BENCHMARK_PATH) as f:
    bench = json.load(f)

all_items = (
    [(item, "misconception") for item in bench["misconceptions"]] +
    [(item, "control")       for item in bench["controls"]]
)
print(f"Benchmark {bench['version']}: {len(all_items)} items ({len(bench['misconceptions'])} miscon + {len(bench['controls'])} control)")

# ─────────────────────────────────────────────
# CELL 5: Prompt builder + answer extractor
# ─────────────────────────────────────────────
def build_prompt(item, model_name):
    opts = "\n".join(f"{k}) {v}" for k, v in item["options"].items())
    if model_name in ("Phi2",):
        return (
            f"Instruct: Answer the multiple choice question. "
            f"Reply with ONLY the single letter A, B, C, or D.\n\n"
            f"Question: {item['question']}\n{opts}\n\nOutput:"
        )
    elif model_name == "Qwen18B":
        return (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\nAnswer the multiple choice question below. "
            f"Reply with ONLY the letter A, B, C, or D.\n\n"
            f"Question: {item['question']}\n{opts}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:  # Gemma and others
        return (
            f"Answer the following multiple choice question. "
            f"Reply with ONLY the letter of the correct answer (A, B, C, or D).\n\n"
            f"Question: {item['question']}\n\n{opts}\n\nAnswer:"
        )

def extract_answer(text):
    text = text.strip().upper()
    m = re.search(r'\b([ABCD])\b', text)
    if m:
        return m.group(1)
    return text[0] if text and text[0] in "ABCD" else "NONE"

# ─────────────────────────────────────────────
# CELL 6: Run inference (with pre-load config fix for Phi-2)
# ─────────────────────────────────────────────
def run_model(model_name, model_path, items):
    print(f"\n{'='*50}\nLoading {model_name}\n{'='*50}")

    # Pre-load config and patch pad_token_id BEFORE model init (fixes Phi-2)
    config = AutoConfig.from_pretrained(model_path, token=HF_TOKEN, trust_remote_code=True)
    if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id
        print(f"  Patched pad_token_id = eos_token_id ({config.eos_token_id})")

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, token=HF_TOKEN,
        torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    results = []

    for item, item_type in items:
        prompt = build_prompt(item, model_name)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        raw    = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answer = extract_answer(raw)
        correct = item["correct"]
        miscon  = item.get("misconception", "N/A")

        results.append({
            "model":                model_name,
            "item_id":              item["id"],
            "item_type":            item_type,
            "category":             item.get("category", "N/A"),
            "misconception_label":  item.get("misconception_label", "N/A"),
            "correct_option":       correct,
            "misconception_option": miscon,
            "model_answer":         answer,
            "raw_output":           raw.strip(),
            "is_correct":           answer == correct,
            "is_misconception":     answer == miscon if miscon != "N/A" else False,
        })
        tag = "✓" if answer==correct else ("✗ MISCON" if answer==miscon else f"✗ other({answer})")
        print(f"  {item['id']} | {tag} | ans={answer} correct={correct}")

    del model; torch.cuda.empty_cache()
    return results

# ─────────────────────────────────────────────
# CELL 7: Run all models + save
# ─────────────────────────────────────────────
all_results = []
for name, path in MODELS.items():
    all_results.extend(run_model(name, path, all_items))

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
    writer.writeheader()
    writer.writerows(all_results)
print(f"\nSaved {len(all_results)} rows → {OUTPUT_PATH}")

# ─────────────────────────────────────────────
# CELL 8: Position bias check
# ─────────────────────────────────────────────
print("\n" + "="*50 + "\nPOSITION BIAS CHECK\n" + "="*50)
biased = []
for name in MODELS:
    answers = [r["model_answer"] for r in all_results if r["model"]==name]
    counts  = Counter(answers)
    total   = len(answers)
    print(f"\n  {name}: ", end="")
    print("  ".join(f"{o}={counts.get(o,0)}({counts.get(o,0)/total*100:.0f}%)" for o in "ABCD"))
    if counts.most_common(1)[0][1]/total > 0.6:
        print(f"  ⚠ BIAS DETECTED")
        biased.append(name)
    else:
        print(f"  ✓ OK")
if biased:
    print(f"\n⚠ Exclude {biased} from CMO — results unreliable")

# ─────────────────────────────────────────────
# CELL 9: CMO + SEA per model pair
# ─────────────────────────────────────────────
by_item = defaultdict(dict)
for r in all_results:
    by_item[r["item_id"]][r["model"]] = r

model_names = [n for n in MODELS if n not in biased]

def compute_cmo_sea(m1, m2, itype):
    bsw = both_w = aow = 0
    for iid, ma in by_item.items():
        if m1 not in ma or m2 not in ma: continue
        if ma[m1]["item_type"] != itype: continue
        a1, a2 = ma[m1]["model_answer"], ma[m2]["model_answer"]
        c = ma[m1]["correct_option"]
        w1, w2 = a1!=c, a2!=c
        if w1 or w2:
            aow += 1
            if w1 and w2:
                both_w += 1
                if a1 == a2: bsw += 1
    cmo = bsw/aow   if aow    else 0.0
    sea = bsw/both_w if both_w else 0.0
    return cmo, sea, bsw, aow, both_w

from itertools import combinations
print("\n" + "="*50 + "\nCMO + SEA RESULTS\n" + "="*50)
for m1, m2 in combinations(model_names, 2):
    print(f"\n  Pair: {m1} vs {m2}")
    print(f"  {'':20} {'Misconception':>15} {'Control':>10} {'Gap':>8}")
    for itype in ["misconception", "control"]:
        cmo, sea, bsw, aow, bw = compute_cmo_sea(m1, m2, itype)
        if itype == "misconception":
            cmo_m, sea_m = cmo, sea
        else:
            cmo_c, sea_c = cmo, sea
    print(f"  {'CMO':<20} {cmo_m:>15.3f} {cmo_c:>10.3f} {cmo_m-cmo_c:>8.3f}")
    print(f"  {'SEA':<20} {sea_m:>15.3f} {sea_c:>10.3f} {sea_m-sea_c:>8.3f}")
    gap = cmo_m - cmo_c
    verdict = "✓ STRONG" if gap>0.20 else ("~ MODERATE" if gap>0.10 else "✗ WEAK")
    print(f"  CMO gap verdict: {verdict}")

# ─────────────────────────────────────────────
# CELL 10: PMD per misconception
# ─────────────────────────────────────────────
print("\n" + "="*50 + "\nPMD (fraction of models choosing misconception)\n" + "="*50)
print(f"{'ID':<8} {'Label':<42} {'PMD':>5}  Models")
print("-"*75)
for item in bench["misconceptions"]:
    iid = item["id"]
    mods = [m for m in model_names if by_item[iid].get(m,{}).get("is_misconception")==True]
    pmd  = len(mods)/len(model_names)
    flag = " ← ALL" if pmd==1.0 else ""
    print(f"{iid:<8} {item['misconception_label']:<42} {pmd:>5.2f}  {mods}{flag}")
