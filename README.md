# Tribe Scorer

Neural creative evaluation powered by [Meta's TRIBE V2](https://github.com/facebookresearch/tribev2) — predict how the human brain responds to your content before you publish it.

Score videos, images, or audio across 6 neural dimensions: **Attention**, **Emotion**, **Memorability**, **Cognition**, **Social Resonance**, and **Language Clarity**.

## How It Works

TRIBE V2 is Meta's open-source brain encoding model trained on 500+ hours of fMRI data from 700+ subjects. It predicts cortical activity across ~20,000 vertices per hemisphere.

This project wraps it with:
- **ROI mapping** — maps fsaverage5 cortical vertices to 6 creative-relevant brain regions using the Destrieux atlas
- **Scoring engine** — converts raw neural predictions into 0-100 scores per metric
- **Cloud inference** — runs the heavy model (LLaMA 3.2 + V-JEPA2 + Wav2Vec-BERT) on Modal A100 GPUs
- **CLI** — `python score.py video.mp4` to score any creative

## Architecture

```
score.py → Modal Predictor (A100 GPU) → TRIBE V2 inference → ROI scoring → JSON results
```

The model runs entirely on Modal's cloud GPUs. Your local machine just sends files and receives scores.

## Setup

### Prerequisites

- Python 3.11+
- [Modal](https://modal.com) account
- [HuggingFace](https://huggingface.co) account with access to [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) (gated model — click "Agree" on the model page)

### Install

```bash
bash setup.sh

# Add your HuggingFace token to Modal
modal secret create huggingface HF_TOKEN=hf_YOUR_TOKEN

# Deploy the inference backend
source .venv/bin/activate
modal deploy tribe_scorer/modal_app.py
```

### Score

```bash
# Single file
python score.py video.mp4

# Directory of creatives
python score.py creatives/

# Save results to file
python score.py creatives/ -o results.json
```

### Use from Python

```python
import modal

Predictor = modal.Cls.from_name("tribe-scorer", "Predictor")
predictor = Predictor()

with open("video.mp4", "rb") as f:
    result = predictor.score.remote(f.read(), "video.mp4")

print(result["overall_score"])
print(result["metrics"]["attention"]["score"])
```

## The 6 Neural Metrics

| Metric | Brain Regions | What It Captures |
|---|---|---|
| **Attention** | Visual cortex (V1-V4), intraparietal sulcus | Visual salience and processing depth |
| **Emotion** | Anterior cingulate, insula, orbitofrontal | Emotional arousal and valence |
| **Memorability** | Parahippocampal, precuneus, posterior cingulate | Memory encoding likelihood |
| **Cognition** | dlPFC, inferior frontal, angular gyrus | Active evaluation and decision processing |
| **Social** | STS, fusiform face area, middle temporal | Face, voice, and social cognition |
| **Language** | Broca's area, Heschl's gyrus, angular gyrus | Speech comprehension depth |

## Cost

- ~$0.06 per creative on Modal A100
- First run is slow (~5 min cold start for model loading), then ~60s per file

## License

The scoring pipeline is MIT. TRIBE V2 itself is [CC-BY-NC-4.0](https://github.com/facebookresearch/tribev2/blob/main/LICENSE).
