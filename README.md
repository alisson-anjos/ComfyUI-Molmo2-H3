# ComfyUI-Molmo2-H3

Experimental Molmo2-8B encoder for MiniMax H3. Supports the original Molmo model and the complete [INT8 ConvRot package](https://huggingface.co/Alissonerdx/Molmo2-8B-Minimax-H3). Model weights and projectors live on Hugging Face; this repository contains the ComfyUI node and reproducible conversion scripts.

## Install

Clone into ComfyUI's `custom_nodes` directory and install `requirements.txt` with ComfyUI's Python:

```bash
git clone https://github.com/alisson-anjos/ComfyUI-Molmo2-H3.git
python -m pip install -r ComfyUI-Molmo2-H3/requirements.txt
```

Use a separate Python environment for Molmo, so its Transformers version does not replace ComfyUI's:

```bash
python -m venv molmo-env
molmo-env/bin/python -m pip install -r ComfyUI-Molmo2-H3/requirements-molmo.txt
```

On Windows use `molmo-env\Scripts\python.exe`. Download the complete HF package, restart ComfyUI and add **Molmo2 as H3 encoder (projection, experimental)** (`MolmoH3EncoderLoader`). Set `molmo_path` to the downloaded folder and `molmo_python` to the separate environment's executable. Select `bundled_default` to find the separate projector under `projections/`; or install another projector in `models/clip_projections/` and select it explicitly.

Remove the previous standalone `comfyui_molmo_h3` custom-node installation before enabling this package, so the same node ID is not registered twice. The folder inside the downloaded HF model is an alternative copy of this node, not a second required installation.

## Workflow

Connect the loader's **CLIP** output to MiniMax H3's text/reference conditioning CLIP input. Native H3 VAE reference inputs continue to supply replacement appearance. Optional visual mode sends reference images and timestamped guide frames to Molmo before projecting text-suffix hidden states.

For source-photo experiments, connect the original-person photo to `source_person_photo` and original video frames to `source_guide`. Put the replacement photo in H3's ordinary reference input. These are separate roles. No SAM or edit mask is introduced by this loader. See `examples/loader_api.json` for the loader configuration; it is a fragment to insert into an existing H3 API workflow, not a standalone video generator.

## Limits and validation

The included projector was trained with floating-point Molmo hidden states. Its INT8 compatibility and correct target selection in multi-person video remain experimental. Text/image inference, finite layer-24 states, subprocess loading and projector-path selection passed; these checks do not establish downstream H3 video quality, tracking accuracy or performance parity. The INT8 runtime is eager PyTorch with INT8 CUDA GEMM, not a fused optimized engine.

The H3 base model, VAE, task LoRAs and native H3 nodes are separate. This is an alternative encoder, not a complete H3 installation. If the Molmo server is left warm in text-only mode, its GPU memory remains occupied until the process exits. Visual requests offload and terminate the subprocess after encoding.

## Release

Publishing follows [ComfyUI-BFSNodes](https://github.com/alisson-anjos/ComfyUI-BFSNodes): pushes changing `pyproject.toml` on main/master, or manual workflow dispatch, call `Comfy-Org/publish-node-action@v1` with publisher `alisson-anjos`. Configure the repository secret `REGISTRY_ACCESS_TOKEN` to enable publication. GitHub does not permit reading an existing repository secret back to copy its value. Without that secret the workflow reports a notice and skips registry publication.

## License

Apache 2.0. Ai2's original Molmo weights/code and their notices are distributed in the HF package. This community integration is not an official Ai2 or MiniMax release.
