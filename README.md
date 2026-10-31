# ComfyUI-Molmo2-H3

Experimental Molmo2-8B encoder for MiniMax H3. Supports the original Molmo model and the complete [INT8 ConvRot package](https://huggingface.co/Alissonerdx/Molmo2-8B-Minimax-H3). The complete Molmo weight is a single `molmo2_8b_int8_convrot.safetensors` file; the projector remains a separate file. Both live on Hugging Face; this repository contains the ComfyUI node and reproducible conversion scripts.

## Install

Clone into ComfyUI's `custom_nodes` directory and install `requirements.txt` with ComfyUI's Python:

```bash
git clone https://github.com/alisson-anjos/ComfyUI-Molmo2-H3.git
python -m pip install -r ComfyUI-Molmo2-H3/requirements.txt
```

Add **Molmo2 as H3 encoder (projection, experimental)** (`MolmoH3EncoderLoader`); it has no inputs and returns a `CLIP` output. It uses ComfyUI's own Python and model folders. The node downloads missing files from the public INT8 package at `Alissonerdx/Molmo2-8B-Minimax-H3` into `ComfyUI/models/text_encoders/Molmo2-8B-Minimax-H3/`, keeping files already present. A fresh install needs about 9.7 GB; if only the projector is missing, it downloads only that projector. The custom ConvRot loader handles the single `.safetensors` model file and the separate projector.

Remove the previous standalone `comfyui_molmo_h3` custom-node installation before enabling this package, so the same node ID is not registered twice. The folder inside the downloaded HF model is an alternative copy of this node, not a second required installation.

## Workflow

Connect the loader's **CLIP** output to MiniMax H3's text/reference conditioning CLIP input. Native H3 VAE reference inputs continue to supply replacement appearance. The Molmo H3 reference node enables visual encoding for that request; the loader itself only loads Molmo and its projector.

For source-photo analysis, use **Molmo2 Analyze References / Points** to provide the original-person photo and guide frames. Put the replacement photo in H3's ordinary reference input. These are separate roles. No SAM or edit mask is introduced by the encoder loader. See `examples/loader_api.json` for the loader configuration; it is a fragment to insert into an existing H3 API workflow, not a standalone video generator.

## Reference, analysis and tracking nodes

- **MiniMax H3 Reference to Video (Molmo2)** accepts the same inputs as the native H3 node and returns `positive` and `latent`. Connect the projected Molmo loader to `clip`. It enables visual encoding for that request without changing the loader object.
- **Molmo2 Analyze References / Points** generates Molmo's text response, timestamped `points` JSON and a `sam_prompt` string. Pictures and video batches use the native reference names. Analysis consumes images/video only; it does not use a VAE or generate H3 video. For locate/mark/track requests with video, the node asks for Molmo's native point-coordinate tags and parses only observed coordinates. If Molmo returns prose or timestamp-only values instead, it reports that no points were produced and leaves `sam_prompt` empty; it never turns missing coordinates into `(0, 0)` marks.
- **Molmo2 Draw Video Points** overlays the returned points on the corresponding sampled frames. Connect its IMAGE output to a video saver. It does not interpolate missing observations.
- **Molmo2 Points to SAM 3.1 Track** takes those points, the original video frames and the native SAM checkpoint's MODEL output. It segments the object under an observed point, then propagates forward and backward within each supplied cut boundary. It returns a union `mask`, a tinted `preview`, and a `report`. Select one `object_id` for a separate target mask, or 0 for all IDs.

For optional text assistance, connect `sam_prompt` to **CLIP Text Encode** using the SAM checkpoint's CLIP, then pass that conditioning to `sam_conditioning`. The projected Molmo CLIP is for H3, not SAM. Failed seeds are reported and remain empty. Install the SAM 3.1 checkpoint separately through ComfyUI's normal checkpoint loader.

Use the same video, frame rate and frame order for analysis, point drawing and SAM. `video_index` selects the corresponding reference video. Points cover only sampled frames; SAM supplies the intervening segmentation. IDs belong to one analysis request and are not global identities. Provide camera cuts through `cut_frames` when needed: this node does not detect cuts automatically or guarantee identity recovery after occlusion. The SAM outputs are optional and are not inserted into H3 conditioning.

## Limits and validation

The included projector was trained with floating-point Molmo hidden states. Its INT8 compatibility and correct target selection in multi-person video remain experimental. Text/image inference, finite layer-24 states, subprocess loading and projector-path selection passed; these checks do not establish downstream H3 video quality, tracking accuracy or performance parity. The INT8 runtime is eager PyTorch with INT8 CUDA GEMM, not a fused optimized engine.

The original Qwen32B text-encoder weights/tokenizer are not needed when using this Molmo loader. The H3 base model, VAE, task LoRAs and native H3 nodes are separate. This is an alternative encoder, not a complete H3 installation. If the Molmo server is left warm in text-only mode, its GPU memory remains occupied until the process exits. Visual requests offload and terminate the subprocess after encoding.

A [native BF16/INT8 tracking comparison](https://huggingface.co/Alissonerdx/Molmo2-8B-Minimax-H3/tree/main/validation/tracking) is available: a short target description failed selection, while explicit shorts/gloves descriptions selected each requested performer in both variants on the same clip.

## License

Apache 2.0. Ai2's original Molmo weights/code and their notices are distributed in the HF package. This community integration is not an official Ai2 or MiniMax release.
