#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Tribe Scorer Setup ==="

# Find Python 3.11–3.13 (avoid 3.14 — some deps don't support it yet)
PYTHON=""
for v in python3.13 python3.12 python3.11; do
    if command -v "$v" &>/dev/null; then PYTHON="$v"; break; fi
done
if [ -z "$PYTHON" ]; then
    command -v python3 &>/dev/null && PYTHON="python3" || { echo "Error: python3 not found"; exit 1; }
fi
echo "Python: $($PYTHON --version)"

# Virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate

# Install
echo "Installing..."
pip install -e . 2>&1 | tail -3

# Modal auth
if ! modal profile list 2>/dev/null | grep -q "active"; then
    echo ""
    echo "Modal needs authentication..."
    modal setup
fi

echo ""
echo "=== Next steps ==="
echo ""
echo "  1. Accept LLaMA 3.2 license:  https://huggingface.co/meta-llama/Llama-3.2-3B"
echo "  2. Add your HuggingFace token: modal secret create huggingface HF_TOKEN=hf_YOUR_TOKEN"
echo "  3. Deploy:                     source .venv/bin/activate && modal deploy tribe_scorer/modal_app.py"
echo "  4. Score:                      python score.py path/to/video.mp4"
echo ""
