# xLSTM Music Generation — Backend API

Flask API for generating symbolic music (MIDI) using a trained xLSTM model.
Accepts a prompt and sequence length, generates REMIGEN2 tokens with the
recurrent `xLSTMGenerator` (O(N) time, constant GPU memory), and returns a
downloadable `.mid` file.

---

## File Structure

```
backend/
├── app.py              ← Flask server
├── generator.py        ← xLSTMGenerator (self-contained, no helibrunna)
├── converter.py        ← REMIGEN2 tokens → MIDI
├── token_analysis.py   ← Grammar checking (pure Python)
├── requirements.txt    ← Pip dependencies (used by Docker)
├── environment.yml     ← Conda environment (no-Docker setup)
├── Dockerfile          ← Docker image definition
├── .dockerignore
└── README.md           ← This file
```

**Model checkpoint**: NOT stored here. Provided at runtime via `MODEL_PATH`.

---

## Setup — Option A: Local (Conda)

**Step 1**: Create the conda environment:
```bash
conda env create -n xlstm-api -f environment.yml
conda activate xlstm-api
```

**Step 2**: Point to your checkpoint:
```bash
export MODEL_PATH=/scratch1/.../checkpoint-46000-last
```

**Step 3**: Start the server:
```bash
python app.py
```

The server loads the model (~10s, includes CUDA kernel compilation on first run) then listens on `http://0.0.0.0:5000`.

---

## Setup — Option B: Docker (Recommended for deployment)

**Requirements on the target server:**
- Docker
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- An NVIDIA GPU with Compute Capability ≥ 8.0 (A100, RTX 3060+, T4, etc.)

**Step 1**: Build the image (run from the `backend/` folder):
```bash
docker build -t xlstm-music-api .
```

**Step 2**: Run the container, mounting your checkpoint folder:
```bash
docker run --gpus all -p 5000:5000 \
  -e MODEL_PATH=/app/model \
  -v /path/to/checkpoint-46000-last:/app/model \
  xlstm-music-api
```

On first startup the sLSTM CUDA kernels will compile (~30s). Subsequent starts are fast.

**Optional: persist the kernel cache** to avoid recompiling on every fresh container:
```bash
docker run --gpus all -p 5000:5000 \
  -e MODEL_PATH=/app/model \
  -v /path/to/checkpoint-46000-last:/app/model \
  -v xlstm-kernel-cache:/root/.cache/torch_extensions \
  xlstm-music-api
```

---

## API Reference

### `GET /health`

Liveness check.

```bash
curl http://localhost:5000/health
```

Response:
```json
{"status": "ok", "model_loaded": true, "model_path": "/app/model"}
```

---

### `POST /generate`

Generate a MIDI file.

**Request body** (JSON):

| Field | Type | Default | Valid range |
|---|---|---|---|
| `prompt` | string | `"s-9 o-0 t-38"` | Non-empty REMIGEN2 tokens |
| `length` | int | `2048` | `100` – `12288` |
| `temperature` | float | `0.8` | `0.1` – `2.0` |
| `seed` | int or null | null | Optional (for reproducibility) |

**Response**: Binary `.mid` file (`Content-Type: audio/midi`).

**Custom response headers** (for display in frontends):

| Header | Description |
|---|---|
| `X-Generation-Time` | Seconds taken to generate |
| `X-Tokens-Per-Second` | Throughput |
| `X-Actual-Tokens` | Tokens generated |
| `X-Num-Bars` | Number of complete bars in output |
| `X-Grammar-Error-Rate` | Fraction of tokens with grammar errors |
| `X-Target-Reached` | Whether `length` tokens were generated |

**Example — curl:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "s-9 o-0 t-38", "length": 2048, "temperature": 0.8}' \
  --output generated.mid
```

**Example — Python:**
```python
import requests

resp = requests.post("http://localhost:5000/generate", json={
    "prompt": "s-9 o-0 t-38",
    "length": 4096,
    "temperature": 0.8,
    "seed": 42,
}, timeout=120)  # generous timeout for long generations

resp.raise_for_status()
with open("generated.mid", "wb") as f:
    f.write(resp.content)

print(f"Generated in {resp.headers['X-Generation-Time']}s")
print(f"Speed: {resp.headers['X-Tokens-Per-Second']} tokens/sec")
```

---

## Expected Generation Times

| Tokens | Approximate time |
|---|---|
| 1,024 | ~7 seconds |
| 2,048 | ~13 seconds |
| 4,096 | ~25 seconds |
| 8,192 | ~51 seconds |
| 12,288 | ~77 seconds |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | *(required)* | Absolute path to checkpoint folder |
| `PORT` | `5000` | Server port |
| `HOST` | `0.0.0.0` | Server bind address |
| `CONTEXT_LENGTH` | `16384` | Inference context length override |
