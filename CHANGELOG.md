# Changelog

## 0.1.0 - 2026-10-31

- Add Molmo2-to-H3 projected encoder with text and optional source-photo/video context.
- Detect the complete INT8 ConvRot package and load its separately stored projector automatically.
- Copy the BFS Comfy Registry publishing workflow and publisher metadata, adapting package identity. Registry publishing requires the repository secret REGISTRY_ACCESS_TOKEN.
- Validation: full quantized text/image inference, layer-24 finite hidden states, packaged subprocess protocol and bundled/external projector resolution passed on an RTX 5090. H3 video quality with INT8 is not yet evaluated.
- Publication: initial source release; registry availability and downstream selection accuracy are not established.
