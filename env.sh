# Source this: keeps every cache/tool inside the workspace.
export AICS_ROOT="$HOME/personal/ai-character-scope"
export PATH="$AICS_ROOT/bin:$PATH"
export HF_HOME="$AICS_ROOT/.cache/huggingface"
export TORCH_HOME="$AICS_ROOT/.cache/torch"
export UV_CACHE_DIR="$AICS_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$AICS_ROOT/.python"
export TRITON_CACHE_DIR="$AICS_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$AICS_ROOT/.cache/inductor"
export CUDA_CACHE_PATH="$AICS_ROOT/.cache/nv"
export npm_config_cache="$AICS_ROOT/.cache/npm"
export DAYDREAM_SCOPE_DIR="$AICS_ROOT/.daydream-scope"
export DAYDREAM_SCOPE_MODELS_DIR="$AICS_ROOT/models"
# Scope resolves these independently of DAYDREAM_SCOPE_DIR (logs otherwise land in ~/.daydream-scope).
export DAYDREAM_SCOPE_LOGS_DIR="$AICS_ROOT/.daydream-scope/logs"
export DAYDREAM_SCOPE_ASSETS_DIR="$AICS_ROOT/.daydream-scope/assets"
export DAYDREAM_SCOPE_LORA_DIR="$AICS_ROOT/models/lora"
export DAYDREAM_SCOPE_PLUGINS_DIR="$AICS_ROOT/.daydream-scope/plugins"
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export CUDA_VISIBLE_DEVICES=0
export USE_TAEHV=1
# Laptop thermals: cap CPU math threads (model loading pegged all 16 cores to 105C).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# torch.compile's parallel inductor workers pegged the CPU to 105C during startup.
export TORCHINDUCTOR_COMPILE_THREADS=2
