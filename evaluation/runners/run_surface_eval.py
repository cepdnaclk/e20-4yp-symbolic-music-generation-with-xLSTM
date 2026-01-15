import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.surface_metrics.muspy_metrics import extract_muspy_metrics

BASE_PATH = "evaluation/data/generated"
MODELS = ["xlstm"] #, "music_transformer", "museformer"]

results = []

for model in MODELS:
    model_dir = os.path.join(BASE_PATH, model)

    if not os.path.exists(model_dir):
        continue

    for fname in os.listdir(model_dir):
        if not fname.endswith(".mid"):
            continue

        midi_path = os.path.join(model_dir, fname)

        try:
            metrics = extract_muspy_metrics(midi_path)
            metrics["model"] = model
            metrics["file"] = fname
            results.append(metrics)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

df = pd.DataFrame(results)
os.makedirs("evaluation/results", exist_ok=True)
df.to_csv("evaluation/results/surface_metrics.csv", index=False)

print("Surface-level evaluation complete.")
