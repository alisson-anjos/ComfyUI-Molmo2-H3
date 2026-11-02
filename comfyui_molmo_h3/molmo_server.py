"""Molmo2 worker server, run in a subprocess with ComfyUI's Python environment.

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
import traceback
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


def prepare_transformers_mask_compatibility(model):
    """Bridge Molmo2 remote generation/mask calls to Transformers 5.x."""
    import inspect
    import sys
    from types import MethodType
    from transformers.masking_utils import create_causal_mask

    module = sys.modules.get(model.__class__.__module__)
    if module is None or not hasattr(module, 'create_causal_mask'):
        return
    try:
        parameters = inspect.signature(create_causal_mask).parameters
    except (TypeError, ValueError):
        parameters = {}
    if 'inputs_embeds' in parameters and 'input_embeds' not in parameters:
        def compatible_create_causal_mask(config, input_embeds=None, **kwargs):
            # Molmo2's cached remote implementation uses the singular keyword
            # and passes cache_position, removed from Transformers 5's helper.
            kwargs.pop('cache_position', None)
            return create_causal_mask(config=config, inputs_embeds=input_embeds, **kwargs)

        module.create_causal_mask = compatible_create_causal_mask

    # Apply this independently of the mask-helper API: Transformers 5 can
    # enter _prefill without initializing cache_position, while Molmo2's
    # remote prepare_inputs_for_generation indexes cache_position[0].
    if not hasattr(model, 'prepare_inputs_for_generation'):
        return
    original_prepare = model.prepare_inputs_for_generation
    if getattr(original_prepare, '_molmo2_cache_position_compat', False):
        return
    # Transformers 5 can enter _prefill without initializing cache_position;
    # Molmo2's remote prepare_inputs_for_generation indexes cache_position[0].
    def compatible_prepare(self, input_ids, *args, **kwargs):
        if kwargs.get('cache_position') is None:
            cache = kwargs.get('past_key_values')
            try:
                past_length = int(cache.get_seq_length()) if cache is not None else 0
            except (AttributeError, TypeError, ValueError):
                past_length = 0
            next_length = kwargs.get('next_sequence_length')
            if next_length is None:
                next_length = input_ids.shape[-1] if past_length == 0 else max(1, input_ids.shape[-1] - past_length)
            kwargs['cache_position'] = torch.arange(
                past_length, past_length + int(next_length), device=input_ids.device, dtype=torch.long)
        return original_prepare(input_ids, *args, **kwargs)
    compatible_prepare._molmo2_cache_position_compat = True
    model.prepare_inputs_for_generation = MethodType(compatible_prepare, model)


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
    prepare_transformers_mask_compatibility(model)
    print('READY', flush=True)
    processor = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('mode') == 'generate':
                if processor is None:
                    prepare_processor_compatibility()
                    processor = AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True)
                batch, presentation = generation_inputs(processor, request['prompt'], request.get('visuals', []),
                    request.get('max_video_frames', 8), (request.get('sampling') or {}).get('system_prompt', ''))
                model.to('cuda'); batch = batch.to('cuda')
                sampling = request.get('sampling') or {}
                options = {'max_new_tokens': request.get('max_new_tokens', 1024),
                           'do_sample': bool(sampling.get('do_sample', False))}
                if options['do_sample']:
                    options.update(temperature=max(1e-5, float(sampling.get('temperature', .7))),
                                   top_k=int(sampling.get('top_k', 64)), top_p=float(sampling.get('top_p', .95)))
                options['repetition_penalty'] = max(1e-5, float(sampling.get('repetition_penalty', 1.05)))
                processors = []
                min_p = float(sampling.get('min_p', .05)) if options['do_sample'] else 0.0
                if min_p > 0:
                    from transformers.generation.logits_process import MinPLogitsWarper
                    processors.append(MinPLogitsWarper(min_p))
                presence = float(sampling.get('presence_penalty', 0.0)) if options['do_sample'] else 0.0
                if presence:
                    from transformers.generation.logits_process import LogitsProcessor
                    prompt_length = batch['input_ids'].shape[1]
                    class PresencePenalty(LogitsProcessor):
                        def __call__(self, input_ids, scores):
                            for row in range(input_ids.shape[0]):
                                seen = input_ids[row, prompt_length:].unique()
                                if seen.numel(): scores[row, seen] -= presence
                            return scores
                    processors.append(PresencePenalty())
                if processors:
                    from transformers.generation.logits_process import LogitsProcessorList
                    options['logits_processor'] = LogitsProcessorList(processors)
                if sampling.get('seed') is not None:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(int(sampling['seed'])); torch.cuda.manual_seed(int(sampling['seed']))
                        with torch.inference_mode(): seq = model.generate(**batch, **options)
                else:
                    with torch.inference_mode(): seq = model.generate(**batch, **options)
                if seq is None:
                    raise RuntimeError('Molmo2 model.generate returned None')
                if isinstance(seq, dict): seq = seq.get('sequences')
                if seq is None or not hasattr(seq, 'shape') or seq.shape[0] == 0:
                    raise RuntimeError('Molmo2 model.generate returned no token sequences')
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
            detail = ' | '.join(traceback.format_exc().strip().splitlines()[-8:])
            print('ERR ' + (str(exc) + ' [' + detail + ']')[:3500], flush=True)


if __name__ == '__main__':
    main()
