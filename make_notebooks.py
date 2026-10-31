r"""
Builds the two Kaggle notebooks from the current source files (re-run after changing src/).

  .venv\Scripts\python kaggle\make_notebooks.py   ->  kaggle\pai-prepare.ipynb, kaggle\pai-train.ipynb
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def src(name):
    return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def writefile(name):
    return code(f"%%writefile src/{name}\n" + src(name))


def notebook(cells, gpu):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": True,
                       "isGpuEnabled": gpu, "language": "python", "sourceType": "notebook"}}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


PREPARE = [
    md("""
# PAI 1/2: prepare data (CPU only, uses no GPU quota)

Downloads PG-19 and Literotica, cleans them (minors filter), trains the 8k tokenizer, writes token files.

**Settings (right panel):** Accelerator **None**, Internet **On** (needs a phone-verified account).
**Run:** *Save Version → Save & Run All (Commit)*. Takes ~2-3 h; you can close the browser.
The output (`tokens/`, `tokenizer.json`, `report.json`) becomes the input of notebook 2.
"""),
    code("!pip install -q tokenizers pyarrow\n!mkdir -p src\n!df -h /tmp /kaggle/working | head -5"),
    writefile("prepare_data.py"),
    code("""
import glob, os, time
from huggingface_hub import snapshot_download

RAW = "/tmp/data"  # big temporary files stay out of /kaggle/working (only the final output is saved)
for name, repo in [("pg19", "emozilla/pg19"), ("literotica", "detain/literotica-stories")]:
    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=["data/train-*.parquet"],
                      local_dir=f"{RAW}/{name}_hf", max_workers=8)
    os.makedirs(f"{RAW}/{name}", exist_ok=True)
    with open(f"{RAW}/{name}/download.log", "w") as log:
        for f in sorted(glob.glob(f"{RAW}/{name}_hf/data/train-*.parquet")):
            os.replace(f, f"{RAW}/{name}/{os.path.basename(f)}")
            log.write(f"{time.strftime('%H:%M:%S')} {os.path.basename(f)} exit=0\\n")
    print(name, len(glob.glob(f"{RAW}/{name}/*.parquet")), "files")
"""),
    code("!python src/prepare_data.py --stage clean --workers 4 --data-dir /tmp/data\n!rm -rf /tmp/data/pg19 /tmp/data/literotica /tmp/data/*_hf"),
    code("!python src/prepare_data.py --stage tokenizer --data-dir /tmp/data"),
    code("!python src/prepare_data.py --stage tokenize --data-dir /tmp/data --max-tokens 1.5e9"),
    code("""
!cp -r /tmp/data/tokens /kaggle/working/ && cp /tmp/data/tokenizer.json /tmp/data/clean/report.json /kaggle/working/
!ls -la /kaggle/working /kaggle/working/tokens && cat /kaggle/working/report.json
"""),
]

TRAIN = [
    md("""
# PAI 2/2: train the Core + Library language model on 2 GPUs

**Settings (right panel):**
1. **Add Input → Notebooks →** your `pai-prepare` notebook (its output holds the token files).
2. Accelerator **GPU T4 x2**. Internet can be Off.
3. *Save Version → Save & Run All (Commit)*. Runs up to ~11.3 h in the background, then saves and stops cleanly.

**To continue a stopped run:** also add **this notebook's previous version output** as an input, then commit again;
it resumes from `checkpoints/lm/latest.pt` automatically.

**Output:** `checkpoints/lm/` (download `latest.pt` or `final.pt` to `PAI\\checkpoints\\lm\\`), `logs/lm_train.log`,
`logs/lm_samples.txt` (sample stories every 30 min).
"""),
    code("!mkdir -p src /kaggle/working/logs /kaggle/working/checkpoints/lm\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    writefile("experiment.py"),
    writefile("multihop.py"),
    writefile("train_lm.py"),
    code("""
import glob, os, shutil

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None

TOK = find("tokenizer.json")
DATA = os.path.dirname(find("pg19_train.bin"))
print("tokenizer:", TOK, "\\ntokens:", DATA)

prev = find("checkpoints/lm/latest.pt")
if prev and not os.path.exists("/kaggle/working/checkpoints/lm/latest.pt"):
    shutil.copy(prev, "/kaggle/working/checkpoints/lm/latest.pt")
    for name in ("lm_train.log", "lm_samples.txt"):
        old = find(f"logs/{name}")
        if old:
            shutil.copy(old, f"/kaggle/working/logs/{name}")
    print("resuming from", prev)
"""),
    code("""
!torchrun --nproc_per_node=2 src/train_lm.py --tokens 3e9 --bs 32 --accum 2 \\
    --data-dir {DATA} --tokenizer {TOK} \\
    --ckpt-dir /kaggle/working/checkpoints/lm --log /kaggle/working/logs/lm_train.log --max-hours 11.3
"""),
    code("!tail -5 /kaggle/working/logs/lm_train.log\n!tail -40 /kaggle/working/logs/lm_samples.txt"),
]

if __name__ == "__main__":
    out = os.path.join(ROOT, "kaggle")
    for name, cells, gpu in (("pai-prepare", PREPARE, False), ("pai-train", TRAIN, True)):
        path = os.path.join(out, f"{name}.ipynb")
        json.dump(notebook(cells, gpu), open(path, "w", encoding="utf-8"), indent=1)
        print("wrote", path)
