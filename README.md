# ComfyUI · AEON DGX Spark — Flux 2 + LTX 2.3 + ACE-Step + MiniMax-H3

> **One image, zero extra downloads.** Pre-built for NVIDIA DGX Spark (GB10 / Blackwell / sm_121a).

```bash
docker pull ghcr.io/aeon-7/comfyui-aeon-spark:latest
# prepare models weights and map your workspace in docker-compose.yml
cd ~/comfyui-spark && docker compose up -d
```

Open `http://<host>:8188` and start generating. That's it.

---

## ✨ What's included (out of the box)

| Category | Details |
| --- | --- |
| **ComfyUI** | Latest master (0.37) |
| **Models** | Flux 2 Dev, LTX 2.3 22B, ACE-Step v1.5, MiniMax-H3 + abliterated text encoders, RealESRGAN upscalers |
| **Custom nodes** | 20 bundled packs (Manager, LTXVideo, GGUF, KJNodes, rgthree, Ollama, PromptRelay, etc.) |
| **Workflows** | Flux 2 t2i, LTX 2.3 T2V/I2V/flf2v/id-lora, ACE-Step audio, MiniMax-H3, PromptRelay timeline control |
| **GPU optimization** | SageAttention v3 compiled for sm_121a, CUDA 13.0.2, NVFP4 hardware GEMMs, unified-memory tuned |
| **Sidecar** | Ollama (gemma3:4b) for prompt expansion in ACE-Step workflow |

**Model weights are NOT auto-downloaded by default.** Use `download_models.py` to download models on demand:
```bash
cd ~/comfyui-spark
export HF_TOKEN=hf_xxxxxxx python3 download_models.py --workspace <parent_of_ComfyUI_models>
```

Your workspace volume persists across container recreations.

---

## 🚀 Quick start (3 steps)

### 1. Get a HuggingFace token
Create a free Read token at https://huggingface.co/settings/tokens. It starts with `hf_`.

### 2. Accept gated model licenses (3 click-throughs)
Open each link and click **"Agree and access"**:
- https://huggingface.co/black-forest-labs/FLUX.2-dev
- https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8
- https://huggingface.co/black-forest-labs/FLUX.2-small-decoder

### 3. Run
```yaml

# map your workspace
    volumes:
      - /your/workspace/to/comfy/.ollama:/root/.ollama

    volumes:
      - /your/workspace/to/comfy/ComfyUI:/workspace/ComfyUI
      - /your/codebase/comfyui-aeon-spark/entrypoint.sh:/usr/local/bin/entrypoint.sh:ro

```

```bash
# start service
docker compose up -d

# check the logs
docker compose logs -f comfyui
```

Wait for `Launching ComfyUI on port 8188` then open your browser. You can use it immediatly if your model weights files are ready.

---

## 📋 Workflows

| # | Workflow | What it does |
| --- | --- | --- |
| 01 | `flux2_text_to_image` | Flux 2 Dev image generation |
| 02 | `ltx2.3_T2V_I2V_distilled` | LTX 2.3 video, 8-step distilled |
| 03 | `ltx2.3_T2V_two_stage` | LTX 2.3 video, two-stage cleaner motion |
| 04 | `ltx2.3_image_to_video` | LTX 2.3 image-to-video |
| 05 | `ltx2.3_first_last_frame_to_video` | LTX 2.3 first/last frame to video |
| 07 | `ltx2.3_id_lora` | LTX 2.3 with identity LoRA |
| 08 | `flux2_klein_9b_text_to_image` | Flux 2 Klein 9B variant |
| 09 | `acestep_ancient_sufi_xl` | ACE-Step audio with Ollama prompt expansion |
| 10 | `ltx2.3_prompt_relay` | LTX 2.3 with timeline-based per-second prompt control |
| **NEW** | **MiniMax-H3** | MiniMax-H3 Sol-Engine workflow (see below) |

---

## 🆕 MiniMax-H3 support

This image includes a dedicated workflow for the **MiniMax-H3** model, built on the excellent work of [drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-DGX-Spark](https://github.com/drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-speed-upgrades-upscaler-finish-Single-DGX-Spark.git).

### Model weights
The MiniMax-H3 weights are hosted at:
**https://huggingface.co/drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-DGX-Spark-weights**

Included models:
- **DiT**: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- **Text encoders**: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, plus two H3 variants in `text_encoders/H3/`
- **VAE**: Video (FP16) and Audio (FP32) variants
- **Upscalers**: RealESRGAN x2+ and x4+

All weights download automatically on first start — no manual download needed.

---

## 📁 Workspace persistence & custom nodes

Your workspace volume at `~/comfyui-spark/workspace/` is **fully persistent** and survives container recreations. You can safely manage:

- `models/` — add, remove, or replace model weights
- `custom_nodes/` — install additional custom node packs via Manager or manually
- `output/` — your generated images, videos, and audio
- `input/` — reference inputs for generation
- `user/default/workflows/` — your saved workflows
- `temp/` — temporary files

**You do not need to rebuild the image** when making changes to your workspace. Everything lives in your volume, not inside the container or image layers.

### Limitations

- **You cannot upgrade ComfyUI** inside the running container without modifying or rebuilding the image. The ComfyUI version is baked into the image at build time. To get a newer version, pull the latest image and recreate the container.
- **Custom node `requirements.txt` auto-install** — when you add custom nodes to your workspace, the container will attempt to automatically install their `requirements.txt` on startup. If network connectivity is poor or unavailable, this may fail. You can manually fix missing dependencies by entering the container:
  ```bash
  docker exec -it comfyui-spark bash
  pip install <missing-package>
  ```

---

## 🙏 Credits & references

| Project | Link |
| --- | --- |
| **Original repo** | https://github.com/AEon-7/comfyui-aeon-spark |
| **MiniMax-H3 work** | https://github.com/drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-speed-upgrades-upscaler-finish-Single-DGX-Spark.git |
| **MiniMax-H3 weights** | https://huggingface.co/drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-DGX-Spark-weights |
| **ComfyUI** | https://github.com/comfyanonymous/ComfyUI |
| **SageAttention** | https://github.com/thu-ml/SageAttention |

---

## 📄 License

MIT. Bundled custom-node packs and model weights retain their respective upstream licenses. **Flux 2 Dev is under Black Forest Labs's Non-Commercial license** — review before commercial use. **MiniMax H3 is released under the MiniMax H3 Community License Agreement**, which contains territorial restrictions and commercial-use thresholds — please refer to MiniMax H3's official license documentation for full terms before use.
