# Changelog

## 0.5.0 - 2026-11-01

- Load Molmo2 INT8 ConvRot in the ComfyUI process and register it with ComfyUI's model manager for GPU placement and offloading. Remove the worker-process boundary and pass image/video tensors directly to the Molmo processor.
- Fully unload Molmo after each inference call so ComfyUI does not partially offload ConvRot's quantized buffers between Molmo and H3.
- Validation: changed Python files compile. Known issue: in a batch run, Molmo2 Analyze References / Points crashed ComfyUI with a CUDA device-side assert (`scatter gather kernel index out of bounds`) on its first request; not yet fixed.
- Publication: pushed to `main`.

## 0.4.2 - 2026-10-31

- Run the Molmo worker with ComfyUI's own Python; remove the extra `molmo-env` and `HENGE_MOLMO_PYTHON` setup from the user workflow. ConvRot registers the standard RoPE formula when Transformers 5.x omits its old `default` entry.
- Validation: ComfyUI's Python loads the INT8 ConvRot model and Molmo2Processor; the Transformers 5 processor compatibility path passed with image and video options. End-to-end video generation remains to be confirmed.
- Publication: source pushed to `main` as 0.4.2; registry package was not published.

## 0.4.1 - 2026-10-31

- Download only missing Molmo package files. An existing INT8 checkpoint is kept when only the separate projector is missing.
- Validation: Python compilation passed; current cloned package resolves without downloading model files.
- Publication: local changes only; not pushed or published.

## 0.4.0 - 2026-10-31

- Remove the Molmo package selector from the H3 encoder loader. The no-input loader downloads the public INT8 ConvRot model and separate projector on first use, then returns the projected `CLIP`.
- Restrict the download to runtime/model files; evaluation videos and reports in the model repository are not downloaded.
- Validation: Python compilation passed; the local no-input loader resolves the cloned INT8 package and separate projector; an INT8 Molmo text-generation smoke returned `Hello!` using Transformers 4.57.1. First-run download in a clean ComfyUI install remains untested.
- Publication: local changes only; not pushed or published.

## 0.3.1 - 2026-10-31

- Make video localization prompts request Molmo's native point/track coordinate format; timestamp-only prose now yields a clear no-points status and an empty SAM prompt instead of misleading output.
- Validation: source changes reviewed; live reference-video inference still needs to confirm coordinate generation for the user's exact clip and prompt.
- Publication: local changes only; not pushed or published.

## 0.3.0 - 2026-10-31

- Simplify the Molmo H3 encoder loader to one model-package dropdown and one `CLIP` output. The projector is resolved from the selected package, and the Molmo Python runtime is resolved internally from the ComfyUI service environment.
- Keep source-photo/video analysis controls in the dedicated analysis node instead of the encoder loader.
- Validation: Python compilation passed; live ComfyUI schema returns only the Molmo package dropdown and a `CLIP` output, with no optional workflow controls.
- Publication: code pushed to `main` as 0.3.0; the GitHub publishing workflow completed successfully.

## 0.2.0 - 2026-10-31

- Discover complete Molmo packages from the `text_encoders` dropdown; retain existing absolute-folder calls.
- Add native H3 Reference to Video conditioning with Molmo visual encoding and unchanged positive/latent outputs.
- Add native Molmo text/point responses, observed-point overlays and optional point-seeded SAM 3.1 video tracking. Text assistance uses the SAM checkpoint's own CLIP.
- Validation: ten CPU tests cover native schema parity, point/time/reference mapping, drawing, SAM seeding in both directions, cut boundaries and missed seeds. A live SAM 3.1 point-seeding/tracking test passed on five boxing frames; the native Molmo response-server test also returned a valid timestamped target point. Crossings, occlusion and reference-based identity selection remain experimental.
- Publication: source release 0.2.0; registry availability depends on the publishing workflow.

## 0.1.0 - 2026-10-31

- Export the complete INT8 model as a single `molmo2_8b_int8_convrot.safetensors` file, keeping the projector separate.
- Add Molmo2-to-H3 projected encoder with text and optional source-photo/video context.
- Detect the complete INT8 ConvRot package and load its separately stored projector automatically.
- Copy the BFS Comfy Registry publishing workflow and publisher metadata, adapting package identity. Registry publishing requires the repository secret REGISTRY_ACCESS_TOKEN.
- Validation: full quantized text/image inference, layer-24 finite hidden states, packaged subprocess protocol and bundled/external projector resolution passed on an RTX 5090. H3 video quality with INT8 is not yet evaluated.
- Publication: initial source release; registry availability and downstream selection accuracy are not established.
