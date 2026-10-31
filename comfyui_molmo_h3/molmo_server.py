"""Molmo2 hidden-state server, run in Molmo2's own Python (its transformers version never touches ComfyUI's).

One JSON request per line on stdin: {"ids": [...], "tap": 24, "out": "/tmp/x.npy"}. The language model runs on those
exact token ids. Optional visuals are .npy paths with image/video type and timestamps; their prefix is
excluded from the returned text states. The states are written as float16 .npy, and one line "OK <path>" or
"ERR <message>" is printed. The model stays loaded between requests.

usage: python molmo_server.py /path/to/Molmo2-8B
"""
import json
import os
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from visual_inputs import prepare_inputs
from generation import generation_inputs, response_points


def prepare_processor_compatibility():
    """Keep Molmo2's processor code working with Transformers 5.x.

    Molmo2 passes image/video token options through ProcessorMixin.__init__.
    Transformers 5 removed these non-component options from its accepted kwargs,
    although Molmo2 still needs them as instance attributes.
    """
    from transformers.processing_utils import ProcessorMixin

    original_init = ProcessorMixin.__init__
    if getattr(original_init, '_molmo2_compat', False):
        return

    option_names = (
        'image_use_col_tokens', 'use_single_crop_col_tokens',
        'use_single_crop_start_token', 'video_use_col_tokens',
        'use_frame_special_tokens',
    )

    def compatible_init(self, *args, **kwargs):
        if self.__class__.__name__ != 'Molmo2Processor':
            return original_init(self, *args, **kwargs)
        options = {name: kwargs.pop(name) for name in option_names if name in kwargs}
        result = original_init(self, *args, **kwargs)
        for name, value in options.items():
            setattr(self, name, value)
        return result

    compatible_init._molmo2_compat = True
    ProcessorMixin.__init__ = compatible_init


def main():
    # Keep loader behavior/version compatibility with this ComfyUI node package;
    # the model folder may contain an older standalone copy of the runtime.
    runtime = Path(__file__).resolve().parents[1] / 'quantization' / 'convrot_runtime.py'
    if os.path.isfile(os.path.join(sys.argv[1], 'convrot_config.json')) and runtime.is_file():
        spec = importlib.util.spec_from_file_location('molmo_convrot_runtime', runtime)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.load_model(sys.argv[1])
    else:
        model = AutoModelForImageTextToText.from_pretrained(sys.argv[1], trust_remote_code=True, dtype=torch.bfloat16,
                                                            device_map='cuda').eval()
    print('READY', flush=True)
    processor = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('mode') == 'generate':
                if processor is None:
                    prepare_processor_compatibility()
                    processor = AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True)
                batch, presentation = generation_inputs(processor, request['prompt'], request.get('visuals', []), request.get('max_video_frames', 8))
                model.to('cuda'); batch = batch.to('cuda')
                with torch.inference_mode():
                    seq = model.generate(**batch, max_new_tokens=request.get('max_new_tokens', 1024), do_sample=False)
                answer = processor.tokenizer.decode(seq[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                result = {'text':answer, 'points':response_points(answer,presentation)}
                with open(request['out'],'w') as f:json.dump(result,f)
                del batch, seq
                model.to('cpu');torch.cuda.empty_cache()
                print('OK '+request['out'],flush=True)
                continue
            offset = 0
            if request.get('visuals'):
                if processor is None:
                    prepare_processor_compatibility()
                    processor = AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True)
                inputs, offset = prepare_inputs(processor, request['ids'], request['visuals'],
                                                request.get('max_video_frames', 8))
                inputs = inputs.to('cuda')
            else:
                tokens = torch.tensor([request['ids']], device='cuda')
                inputs = {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens)}
            model.to('cuda')
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            hidden = out.hidden_states[int(request['tap'])][0, offset:]
            if hidden.shape[0] != len(request['ids']):
                raise RuntimeError('Visual/text alignment failed')
            np.save(request['out'], hidden.float().cpu().numpy().astype(np.float16))
            del out, hidden, inputs
            if request.get('offload_after'):
                model.to('cpu')
                torch.cuda.empty_cache()
            print('OK ' + request['out'], flush=True)
        except Exception as exc:  # one bad request must not kill the warm model
            print('ERR ' + str(exc).replace('\n', ' ')[:2000], flush=True)


if __name__ == '__main__':
    main()
