#!/usr/bin/env bash
# =============================================================================
# ComfyUI entrypoint — wires persistent volume, downloads models, launches UI.
# =============================================================================
set -euo pipefail

COMFY_HOME="${COMFY_HOME:-/opt/ComfyUI}"
WORKSPACE="${WORKSPACE:-/workspace/ComfyUI}"
PORT="${COMFYUI_PORT:-8188}"

# DGX Spark unified-memory friendly defaults (override with COMFYUI_FLAGS env)
DEFAULT_FLAGS="--listen 0.0.0.0 --port ${PORT} \
  --use-sage-attention \
  --disable-comfy-compiler \
  --bf16-unet --bf16-vae --bf16-text-enc \
  --disable-pinned-memory \
  --reserve-vram 2.0 \
  --preview-method auto \
  --enable-cors-header \
  --enable-manager \
  --enable-assets"
COMFYUI_FLAGS="${COMFYUI_FLAGS:-${DEFAULT_FLAGS}}"

log() { printf '\033[1;36m[entrypoint]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[entrypoint]\033[0m %s\n' "$*" >&2; }

log "ComfyUI for DGX Spark (sm_121a, CUDA 13)"
log "Python : $(python --version 2>&1)"
log "Torch  : $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
log "Arches : $(python -c 'import torch; print(torch.cuda.get_arch_list())')"
log "GPU    : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
log "Sage   : $(python -c "import sageattention,inspect; v=getattr(sageattention,'__version__',None); print(v or ('installed @ '+inspect.getfile(sageattention)))" 2>/dev/null || echo 'unavailable')"
log "Manager: $(python -c "import comfyui_manager; print('pip pkg present @ ' + comfyui_manager.__file__)" 2>/dev/null || echo 'pip pkg missing')"

# -----------------------------------------------------------------------------
# 1. Bootstrap persistent workspace volume
# -----------------------------------------------------------------------------
log "Bootstrapping workspace at ${WORKSPACE}"
mkdir -p "${WORKSPACE}"/{models,custom_nodes,output,input,user/default/workflows,temp,.cache/huggingface}
mkdir -p "${WORKSPACE}"/models/{diffusion_models,text_encoders,vae,loras,clip,clip_vision,controlnet,upscale_models,latent_upscale_models,embeddings,unet,checkpoints,style_models,gligen,hypernetworks,configs,photomaker,sams,ipadapter,inpaint,facerestore_models,facedetection,insightface}

# Symlink ComfyUI's data dirs to the persistent volume
for d in models custom_nodes output input user temp; do
    if [ -e "${COMFY_HOME}/${d}" ] && [ ! -L "${COMFY_HOME}/${d}" ]; then
        # Move any pre-existing content into the volume on first start
        if [ -d "${COMFY_HOME}/${d}" ] && [ -z "$(ls -A "${COMFY_HOME}/${d}" 2>/dev/null)" ]; then
            rm -rf "${COMFY_HOME}/${d}"
        else
            log "Migrating existing ${COMFY_HOME}/${d} into volume"
            rsync -a "${COMFY_HOME}/${d}/" "${WORKSPACE}/${d}/" 2>/dev/null || true
            rm -rf "${COMFY_HOME}/${d}"
        fi
    fi
    ln -snf "${WORKSPACE}/${d}" "${COMFY_HOME}/${d}"
done

# -----------------------------------------------------------------------------
# 2. Seed bundled custom nodes into the persistent volume (idempotent)
#    — only copies a node if its target directory does not exist yet, so users
#      can delete/replace nodes without them being re-added on every start.
# -----------------------------------------------------------------------------
if [ -d /opt/bundled_custom_nodes ]; then
    for src in /opt/bundled_custom_nodes/*/; do
        node_name="$(basename "${src}")"
        dst="${WORKSPACE}/custom_nodes/${node_name}"
        if [ ! -e "${dst}" ]; then
            log "Seeding custom node: ${node_name}"
            cp -a "${src}" "${dst}"
        fi
    done
fi

# -----------------------------------------------------------------------------
# 3. Install requirements.txt from any custom node (idempotent — fast no-op
#    after first start because pip caches resolved markers).
# -----------------------------------------------------------------------------
log "Installing requirements from custom nodes"
for req in "${WORKSPACE}"/custom_nodes/*/requirements.txt; do
    [ -f "${req}" ] || continue
    log "  -> $(dirname "${req}" | xargs basename)"
    pip install -q --no-deps -r "${req}" 2>/dev/null || \
      pip install -q -r "${req}" 2>/dev/null || \
      warn "  partial failure installing ${req}"
done

# -----------------------------------------------------------------------------
# 4. Seed default workflows
# -----------------------------------------------------------------------------
if [ -d /opt/default_workflows ]; then
    for wf in /opt/default_workflows/*.json; do
        [ -f "${wf}" ] || continue
        wf_name="$(basename "${wf}")"
        dst="${WORKSPACE}/user/default/workflows/${wf_name}"
        if [ ! -e "${dst}" ]; then
            log "Seeding workflow: ${wf_name}"
            mkdir -p "$(dirname "${dst}")"
            cp -a "${wf}" "${dst}"
        fi
    done
fi

# -----------------------------------------------------------------------------
# 5. Download models — only on first start, or when explicitly forced.
#    Skip with SKIP_MODEL_DOWNLOAD=1, force with FORCE_MODEL_DOWNLOAD=1.
# -----------------------------------------------------------------------------
SENTINEL="${WORKSPACE}/.models_seeded"
if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    log "SKIP_MODEL_DOWNLOAD=1 — skipping model fetch"
elif [ -f "${SENTINEL}" ] && [ "${FORCE_MODEL_DOWNLOAD:-0}" != "1" ]; then
    log "Models already seeded (delete ${SENTINEL} or set FORCE_MODEL_DOWNLOAD=1 to re-fetch)"
else
    log "Downloading models — this can take a while on first start..."
    if python /usr/local/bin/download_models.py --workspace "${WORKSPACE}"; then
        touch "${SENTINEL}"
        log "Model fetch complete"
    else
        warn "Model fetch reported errors; ComfyUI will still start. See log above."
    fi
fi

# -----------------------------------------------------------------------------
# 6. Launch ComfyUI
# -----------------------------------------------------------------------------
log "Launching ComfyUI on port ${PORT}"
log "Flags: ${COMFYUI_FLAGS}"
cd "${COMFY_HOME}"
exec python main.py ${COMFYUI_FLAGS}
