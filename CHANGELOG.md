# Changelog

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
