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

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from visual_inputs import prepare_inputs


def main():
    runtime = os.path.join(sys.argv[1], 'convrot_runtime.py')
    if os.path.isfile(os.path.join(sys.argv[1], 'convrot_config.json')):
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
            offset = 0
            if request.get('visuals'):
                if processor is None:
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
