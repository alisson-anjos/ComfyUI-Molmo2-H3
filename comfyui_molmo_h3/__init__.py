"""Molmo2 as the MiniMax H3 text encoder, through a calibrated projection (experimental).

A ClipProj-style projection (https://github.com/nicolab28/ComfyUI-ClipProj, MIT; same file format and maths) maps
Molmo2-8B's hidden states to the conditioning H3's 32B Qwen3-VL encoder produces. The model is loaded in-process
through ComfyUI's model manager. The projected CLIP is used for H3 conditioning; the separate Molmo2 Generate Text
node handles generation. Phase A covers text: reference pictures and videos keep their native H3 latent path, and
only their <Picture N>/<Video N> labels reach Molmo2.
"""
import importlib.util
import os
import threading
from types import MethodType
import inspect
from contextlib import contextmanager

import torch
from .model_paths import resolve_model, resolve_projector
import folder_paths
import comfy.model_management as mm

PROJECTIONS = os.path.join(folder_paths.models_dir, 'clip_projections')
os.makedirs(PROJECTIONS, exist_ok=True)
folder_paths.add_model_folder_path('clip_projections', PROJECTIONS)


def prepare_processor_compatibility():
    from transformers.processing_utils import ProcessorMixin
    original_init = ProcessorMixin.__init__
    if getattr(original_init, '_molmo2_compat', False):
        return
    option_names = ('image_use_col_tokens', 'use_single_crop_col_tokens', 'use_single_crop_start_token',
                    'video_use_col_tokens', 'use_frame_special_tokens')
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


def prepare_generation_compatibility(model):
    from transformers.masking_utils import create_causal_mask
    import sys
    module = sys.modules.get(model.__class__.__module__)
    try:
        params = inspect.signature(create_causal_mask).parameters
    except (TypeError, ValueError):
        params = {}
    if module is not None and hasattr(module, 'create_causal_mask') and 'inputs_embeds' in params and 'input_embeds' not in params:
        def compatible_mask(config, input_embeds=None, **kwargs):
            kwargs.pop('cache_position', None)
            return create_causal_mask(config=config, inputs_embeds=input_embeds, **kwargs)
        module.create_causal_mask = compatible_mask
    original = model.prepare_inputs_for_generation
    if getattr(original, '_molmo2_cache_position_compat', False):
        return
    def compatible_prepare(self, input_ids, *args, **kwargs):
        if kwargs.get('cache_position') is None:
            cache = kwargs.get('past_key_values')
            try:
                past = int(cache.get_seq_length()) if cache is not None else 0
            except (AttributeError, TypeError, ValueError):
                past = 0
            length = kwargs.get('next_sequence_length')
            if length is None:
                length = input_ids.shape[-1] if past == 0 else max(1, input_ids.shape[-1] - past)
            kwargs['cache_position'] = torch.arange(past, past + int(length), device=input_ids.device)
        return original(input_ids, *args, **kwargs)
    compatible_prepare._molmo2_cache_position_compat = True
    model.prepare_inputs_for_generation = MethodType(compatible_prepare, model)


def load_projection(path):
    from safetensors import safe_open
    with safe_open(path, 'pt') as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata() or {}
    tensors['tap'] = int(meta.get('tap', 24))
    return tensors


def build_residual(proj, device):
    """The optional residual network (keys mlp.N.weight / mlp.N.bias, GELU between Linear layers), as ClipProj."""
    layers = sorted({int(k.split('.')[1]) for k in proj if isinstance(k, str) and k.startswith('mlp.')})
    if not layers:
        return None
    modules = []
    for n, i in enumerate(layers):
        w = proj[f'mlp.{i}.weight']
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=f'mlp.{i}.bias' in proj)
        lin.weight.data = w.to(device)
        if lin.bias is not None:
            lin.bias.data = proj[f'mlp.{i}.bias'].to(device)
        modules.append(lin)
        if n < len(layers) - 1:
            modules.append(torch.nn.GELU())
    net = torch.nn.Sequential(*modules).to(device)
    net.train(False)
    return net


class MolmoRuntime:
    """Molmo2 loaded as a ComfyUI-managed model in this process."""
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_path):
        with cls._lock:
            runtime = cls._instances.get(model_path)
            if runtime is None:
                runtime = cls._instances[model_path] = cls(model_path)
            return runtime

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.patcher = None
        self.processor = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoProcessor
        import comfy.model_patcher
        runtime_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'quantization', 'convrot_runtime.py')
        spec = importlib.util.spec_from_file_location('molmo_h3_convrot_runtime', runtime_path)
        runtime_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime_module)
        load_model = runtime_module.load_model
        hf_model = load_model(self.model_path, device='cpu')
        prepare_generation_compatibility(hf_model)
        prepare_processor_compatibility()
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        model = MolmoComfyModel(hf_model)
        self.patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=mm.get_torch_device(), offload_device=mm.text_encoder_offload_device(),
            size=mm.module_size(model))
        self.model = model
        self.processor = processor

    def _load_gpu(self):
        self.load()
        mm.load_models_gpu([self.patcher], force_full_load=True)
        return self.model, self.processor, mm.get_torch_device()

    @contextmanager
    def _gpu_model(self):
        try:
            yield self._load_gpu()
        finally:
            if self.patcher is not None:
                mm.unload_model_and_clones(self.patcher)

    def hidden(self, ids, tap, visuals=None, max_video_frames=8):
        from .visual_inputs import prepare_inputs
        with self.lock, self._gpu_model() as (model, processor, device):
            if visuals:
                inputs, offset = prepare_inputs(processor, ids, visuals, max_video_frames)
                inputs = inputs.to(device)
            else:
                offset = 0
                tokens = torch.tensor([ids], device=device)
                inputs = {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens)}
            with torch.inference_mode():
                output = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            hidden = output.hidden_states[int(tap)][0, offset:]
            if hidden.shape[0] != len(ids):
                raise RuntimeError('Molmo2 visual/text alignment failed')
            return hidden.float().to('cpu', dtype=torch.float16)


    def generate(self, prompt, visuals, max_new_tokens=1024, max_video_frames=8, **sampling):
        from .generation import generation_inputs, response_points
        with self.lock, self._gpu_model() as (model, processor, device):
            batch, presentation = generation_inputs(processor, prompt, visuals, max_video_frames,
                                                     sampling.pop('system_prompt', ''))
            batch = batch.to(device)
            options = {'max_new_tokens': max_new_tokens, 'do_sample': bool(sampling.pop('do_sample', False))}
            repetition = max(1e-5, float(sampling.pop('repetition_penalty', 1.05)))
            if repetition != 1.0:
                # Transformers' repetition penalty gathers scores at every input id; Molmo's image-token ids lie past
                # the logits vocabulary and trip a CUDA index assert. Penalize only ids the logits contain.
                from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
                class RepetitionPenalty(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        for row in range(input_ids.shape[0]):
                            seen = input_ids[row][input_ids[row] < scores.shape[-1]].unique()
                            score = scores[row, seen]
                            scores[row, seen] = torch.where(score < 0, score * repetition, score / repetition)
                        return scores
                options['logits_processor'] = LogitsProcessorList([RepetitionPenalty()])
            if options['do_sample']:
                options.update(temperature=max(1e-5, float(sampling.pop('temperature', .7))),
                               top_k=int(sampling.pop('top_k', 64)), top_p=float(sampling.pop('top_p', .95)))
                min_p = float(sampling.pop('min_p', .05))
                if min_p > 0:
                    from transformers.generation.logits_process import MinPLogitsWarper
                    from transformers.generation.logits_process import LogitsProcessorList
                    options['logits_processor'] = LogitsProcessorList(list(options.get('logits_processor', [])) + [MinPLogitsWarper(min_p)])
                presence = float(sampling.pop('presence_penalty', 0.))
                if presence:
                    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
                    prompt_length = batch['input_ids'].shape[1]
                    class PresencePenalty(LogitsProcessor):
                        def __call__(self, input_ids, scores):
                            for row in range(input_ids.shape[0]):
                                seen = input_ids[row, prompt_length:].unique()
                                if seen.numel():
                                    scores[row, seen] -= presence
                            return scores
                    options['logits_processor'] = LogitsProcessorList(list(options.get('logits_processor', [])) + [PresencePenalty()])
            seed = sampling.pop('seed', None)
            with torch.inference_mode():
                if seed is None:
                    sequences = model.generate(**batch, **options)
                else:
                    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                        torch.manual_seed(int(seed))
                        if device.type == 'cuda': torch.cuda.manual_seed(int(seed))
                        sequences = model.generate(**batch, **options)
            if isinstance(sequences, dict):
                sequences = sequences.get('sequences')
            if sequences is None or sequences.shape[0] == 0:
                raise RuntimeError('Molmo2 returned no generated tokens')
            answer = processor.tokenizer.decode(sequences[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
            return {'text': answer, 'points': response_points(answer, presentation)}


class MolmoComfyModel(torch.nn.Module):
    """Give the read-only HF model.device property a ComfyUI-managed wrapper."""
    def __init__(self, model):
        super().__init__()
        self.transformer = model

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.transformer.generate(*args, **kwargs)


class MolmoProjectedCLIP:
    """Answers the calls MiniMax H3 nodes make on a CLIP (tokenize / encode_from_tokens_scheduled)."""

    def __init__(self, model, projection, phase_b_vision=False, phase_b_max_frames=8,
                 source_person_photo=None, source_guide=None, source_guide_fps=24.):
        from tokenizers import Tokenizer
        self.model, self.projection = model, projection
        self.runtime = MolmoRuntime.get(model)
        self.tokenizer = Tokenizer.from_file(os.path.join(model, 'tokenizer.json'))
        self.proj = load_projection(projection)
        self.cache = {}
        self.phase_b_vision = phase_b_vision
        self.phase_b_max_frames = phase_b_max_frames
        self.source_person_photo = source_person_photo
        self.source_guide = source_guide
        self.source_guide_fps = source_guide_fps

    def clone(self):
        return self

    def tokenize(self, text, return_word_ids=False, images=None, minimax_ref_items=None, **kwargs):
        ids = []
        def add(s):
            ids.extend(self.tokenizer.encode(s, add_special_tokens=False).ids)
        counters = {}
        for item in minimax_ref_items or []:
            kind = item['type'];counters[kind] = counters.get(kind, 0) + 1
            # Phase A: only the label; the picture/video itself reaches H3 through its VAE latent.
            add({'image': '<Picture %d>: ', 'video': '<Video %d>: ', 'audio': '<Audio %d>: '}.get(kind, '<Picture %d>: ') % counters[kind])
        images = images or []
        if not minimax_ref_items:
            for i, _ in enumerate(images):
                add('<Picture %d>: ' % (i + 1))
        has_visuals = self.source_person_photo is not None or self.source_guide is not None or bool(images) or any(item['type'] in ('image', 'video') and 'data' in item
                                         for item in minimax_ref_items or [])
        if self.phase_b_vision and has_visuals:
            # Visual labels are in the prefix; calibrated multimodal targets use the prompt suffix only.
            ids = []
        add(text)
        tokens = {'molmo': [[(t, 1.0) for t in ids or [151643]]]}
        if self.phase_b_vision:
            tokens['molmo_visuals'] = [item for item in minimax_ref_items or []
                                      if item['type'] in ('image', 'video') and 'data' in item]
            if not minimax_ref_items:
                tokens['molmo_visuals'] = [{'type': 'image', 'data': image} for image in images]
            if self.source_person_photo is not None:
                tokens['molmo_visuals'].insert(0, {'type':'source_person', 'data':self.source_person_photo[:1]})
            if self.source_guide is not None and not any(v['type']=='video' for v in tokens['molmo_visuals']):
                tokens['molmo_visuals'].append({'type':'video', 'data':self.source_guide,
                    'timestamps':[i/self.source_guide_fps for i in range(len(self.source_guide))]})
        return tokens

    def _encode(self, tokens):
        ids = [t[0] for t in tokens['molmo'][0]]
        h = self.runtime.hidden(ids, self.proj['tap'], tokens.get('molmo_visuals'), self.phase_b_max_frames)
        device = mm.get_torch_device()
        if self.cache.get('device') != device:
            self.cache = {'device': device, 'p': {k: self.proj[k].to(device=device, dtype=torch.float32) for k in ('W', 'mean_in', 'std_in', 'mean_out', 'std_out') if k in self.proj},
                          'mlp': build_residual(self.proj, device)}
        p = self.cache['p']
        with torch.no_grad():
            xn = (h.to(device).float() - p['mean_in']) / p['std_in']
            yn = xn @ p['W'] if 'W' in p else None
            if self.cache['mlp'] is not None:
                net = self.cache['mlp'];r = net(xn.to(net[0].weight.dtype)).float()
                yn = r if yn is None else yn + r
            cond = (yn * p['std_out'] + p['mean_out'])[None]
            # Token 0 is an attention sink with a constant direction: substitute its measured value (as ClipProj).
            if 'sink_out' in self.proj and cond.shape[1]:
                sink = self.proj['sink_out'].to(cond)
                if tokens.get('molmo_visuals'):
                    # The real attention sink is in the visual prefix, outside the selected text suffix.
                    cond = torch.cat([sink.reshape(1, 1, -1), cond], dim=1)
                else:
                    cond[:, 0] = sink
        return cond.to(mm.intermediate_device())

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond = self._encode(tokens)
        if return_dict:
            return {'cond': cond, 'pooled_output': None}
        return (cond, None) if return_pooled else cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        return [[self._encode(tokens), {'pooled_output': None, **add_dict}]]

    def encode(self, text):
        return self.encode_from_tokens(self.tokenize(text))


class MolmoH3EncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}

    RETURN_TYPES = ('CLIP',); FUNCTION = 'load'; CATEGORY = 'Molmo2/H3'

    def load(self):
        molmo_path = resolve_model()
        return (MolmoProjectedCLIP(molmo_path, resolve_projector(molmo_path)),)


NODE_CLASS_MAPPINGS = {'MolmoH3EncoderLoader': MolmoH3EncoderLoader}
NODE_DISPLAY_NAME_MAPPINGS = {'MolmoH3EncoderLoader': 'Molmo2 as H3 encoder (projection, experimental)'}

from .nodes import NODE_CLASS_MAPPINGS as _reference_nodes
NODE_CLASS_MAPPINGS.update(_reference_nodes)
