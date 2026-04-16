"""
Modal app — TRIBE V2 inference + brain-region scoring on A100 GPUs.

Deploy:
    modal deploy tribe_scorer/modal_app.py

Call from Python:
    import modal
    Predictor = modal.Cls.from_name("tribe-scorer", "Predictor")
    predictor = Predictor()
    result = predictor.score.remote(file_bytes, "video.mp4")
"""

import modal

app = modal.App("tribe-scorer")

CACHE_DIR = "/cache/tribev2"
hf_secret = modal.Secret.from_name("huggingface")


def download_weights():
    """Pre-download model weights (including gated LLaMA 3.2) during image build."""
    import os
    from huggingface_hub import login, snapshot_download

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    snapshot_download("facebook/tribev2", cache_dir=CACHE_DIR)
    snapshot_download("meta-llama/Llama-3.2-3B", cache_dir=CACHE_DIR)


def download_atlas():
    """Pre-download Destrieux brain atlas during image build."""
    from nilearn.datasets import fetch_atlas_surf_destrieux
    fetch_atlas_surf_destrieux()


gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "tribev2 @ git+https://github.com/facebookresearch/tribev2.git",
        "nilearn>=0.10",
    )
    .env({"HF_HOME": CACHE_DIR})
    .run_function(download_weights, secrets=[hf_secret])
    .run_function(download_atlas)
    .add_local_python_source("tribe_scorer")
)


@app.cls(
    image=gpu_image,
    gpu="A100",
    timeout=1500,
    scaledown_window=300,
    startup_timeout=1500,
    secrets=[hf_secret],
)
class Predictor:
    """TRIBE V2 inference with ROI scoring."""

    @modal.enter()
    def load_model(self):
        import os
        from huggingface_hub import login

        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)

        from tribev2 import TribeModel
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2", cache_folder=CACHE_DIR
        )

    @modal.method()
    def predict(self, file_bytes: bytes, filename: str) -> dict:
        """Run raw TRIBE V2 inference. Returns {predictions, shape}."""
        import os
        import subprocess
        import tempfile
        from pathlib import Path
        import numpy as np

        ext = Path(filename).suffix.lower()
        video_exts = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
        audio_exts = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(file_bytes)
            input_path = f.name

        try:
            if ext in image_exts:
                video_path = input_path + ".mp4"
                subprocess.run(
                    ["ffmpeg", "-y", "-loop", "1", "-i", input_path,
                     "-c:v", "libx264", "-t", "5", "-pix_fmt", "yuv420p",
                     "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                     "-loglevel", "error", video_path],
                    check=True, capture_output=True,
                )
                os.unlink(input_path)
                input_path = video_path
                ext = ".mp4"

            if ext in video_exts:
                df = self.model.get_events_dataframe(video_path=input_path)
            elif ext in audio_exts:
                df = self.model.get_events_dataframe(audio_path=input_path)
            else:
                raise ValueError(f"Unsupported: {ext}")

            preds, _ = self.model.predict(events=df)
            preds = np.asarray(preds)
            return {"predictions": preds.tolist(), "shape": list(preds.shape)}
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)

    @modal.method()
    def score(self, file_bytes: bytes, filename: str) -> dict:
        """Run inference + ROI scoring. Returns scored JSON with 6 metrics."""
        import warnings
        import numpy as np
        warnings.filterwarnings("ignore", message=".*regions are present.*")

        raw = self.predict.local(file_bytes, filename)
        predictions = np.array(raw["predictions"])

        from tribe_scorer.metrics import compute_creative_scores, normalize_batch_scores
        from tribe_scorer.regions import build_roi_masks

        masks = build_roi_masks(predictions.shape[1])
        scores = compute_creative_scores(predictions, masks=masks)
        normalize_batch_scores([scores])
        return {"file": filename, **scores}
