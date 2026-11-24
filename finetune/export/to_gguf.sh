#!/usr/bin/env bash
# Convert the merged model to GGUF and quantize it to Q4_K_M with llama.cpp (Linux/macOS, e.g. Kaggle).
#
#   bash export/to_gguf.sh outputs/merged outputs/gguf
#
# Produces <out>/footnote-judge-f16.gguf and <out>/footnote-judge-q4_k_m.gguf (about 1.1 GB for
# Qwen3-1.7B). On Windows, use a prebuilt llama.cpp release (llama-quantize.exe) for step 3 and run
# step 2 with Python the same way.
set -euo pipefail

MERGED="${1:?usage: to_gguf.sh <merged-model-dir> <out-dir>}"
OUT="${2:?usage: to_gguf.sh <merged-model-dir> <out-dir>}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
mkdir -p "$OUT"

# 1. llama.cpp: the converter (Python) and the quantizer (C++).
if [ ! -d "$LLAMA_CPP" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
python -m pip install -q -r "$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt"

# 2. Hugging Face safetensors -> GGUF at f16.
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --outfile "$OUT/footnote-judge-f16.gguf" --outtype f16

# 3. Build llama-quantize once, then f16 -> Q4_K_M.
if [ ! -x "$LLAMA_CPP/build/bin/llama-quantize" ]; then
  cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null
  cmake --build "$LLAMA_CPP/build" --target llama-quantize -j "$(nproc 2>/dev/null || echo 4)" > /dev/null
fi
"$LLAMA_CPP/build/bin/llama-quantize" "$OUT/footnote-judge-f16.gguf" "$OUT/footnote-judge-q4_k_m.gguf" Q4_K_M

cp "$(dirname "$0")/Modelfile" "$OUT/Modelfile"
ls -lh "$OUT"
echo "Next: cd $OUT && ollama create footnote-judge -f Modelfile"
