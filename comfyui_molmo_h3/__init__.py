"""Molmo2 as the MiniMax H3 text encoder, through a calibrated projection (experimental).

A ClipProj-style projection (https://github.com/nicolab28/ComfyUI-ClipProj, MIT; same file format and maths) maps
Molmo2-8B's hidden states to the conditioning H3's 32B Qwen3-VL encoder produces. Molmo2 runs in its own Python,
started once and kept warm. Phase A covers text: reference pictures and videos keep their native H3 latent path, and
only their <Picture N>/<Video N> labels reach Molmo2.
"""
import json
import os
import subprocess
import tempfile
import threading

import torch
import folder_paths
import comfy.model_management as mm

PROJECTIONS = os.path.join(folder_paths.models_dir, 'clip_projections')
os.makedirs(PROJECTIONS, exist_ok=True)
folder_paths.add_model_folder_path('clip_projections', PROJECTIONS)
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molmo_server.py')


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


class MolmoServer:
    """One warm Molmo2 process per (python, model)."""
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, python, model):
        with cls._lock:
            server = cls._instances.get((python, model))
            if server is None or server.proc.poll() is not None:
                server = cls._instances[(python, model)] = cls(python, model)
            return server

    def __init__(self, python, model):
        self.proc = subprocess.Popen([python, SERVER, model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        for line in self.proc.stdout:
            if line.startswith('READY'):
                break
        else:
            raise RuntimeError('Molmo2 server exited before it was ready')
        self.lock = threading.Lock()

    def hidden(self, ids, tap, visuals=None, max_video_frames=8):
        with self.lock, tempfile.TemporaryDirectory(prefix='molmo-h3-') as folder:
            import numpy as np
            items = []
            for n, item in enumerate(visuals or []):
                data = item['data'].detach().cpu()
                path = os.path.join(folder, f'visual_{n}.npy')
                np.save(path, (data.clamp(0, 1) * 255).round().to(torch.uint8).numpy())
                items.append({'type': item['type'], 'path': path,
                              'timestamps': item.get('timestamps', [])})
            out = os.path.join(folder, 'h.npy')
            self.proc.stdin.write(json.dumps({'ids': ids, 'tap': tap, 'out': out, 'visuals': items,
                                              'offload_after': bool(items),
                                              'max_video_frames': max_video_frames}) + '\n'); self.proc.stdin.flush()
            reply = self.proc.stdout.readline().strip()
            if not reply.startswith('OK '):
                raise RuntimeError('Molmo2 failed: ' + reply)
            import numpy as np
            res = torch.from_numpy(np.load(out))
            if items:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
            return res


class MolmoProjectedCLIP:
    """Answers the calls MiniMax H3 nodes make on a CLIP (tokenize / encode_from_tokens_scheduled)."""

    def __init__(self, python, model, projection, phase_b_vision=False, phase_b_max_frames=8,
                 source_person_photo=None, source_guide=None, source_guide_fps=24.):
        from tokenizers import Tokenizer
        self.python, self.model, self.projection = python, model, projection
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

    def tokenize(self, text, return_word_ids=False, images=[], minimax_ref_items=None, **kwargs):
        ids = []
        def add(s):
            ids.extend(self.tokenizer.encode(s, add_special_tokens=False).ids)
        counters = {}
        for item in minimax_ref_items or []:
            kind = item['type'];counters[kind] = counters.get(kind, 0) + 1
            # Phase A: only the label; the picture/video itself reaches H3 through its VAE latent.
            add({'image': '<Picture %d>: ', 'video': '<Video %d>: ', 'audio': '<Audio %d>: '}.get(kind, '<Picture %d>: ') % counters[kind])
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
        if tokens.get('molmo_visuals'):
            mm.unload_all_models()
            mm.soft_empty_cache()
        h = MolmoServer.get(self.python, self.model).hidden(ids, self.proj['tap'], tokens.get('molmo_visuals'), self.phase_b_max_frames)
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
        return {'required': {'molmo_path': ('STRING', {'default': os.environ.get('HENGE_MOLMO_MODEL', '')}),
                             'projection': (['bundled_default'] + folder_paths.get_filename_list('clip_projections'),),
                             'molmo_python': ('STRING', {'default': os.environ.get('HENGE_MOLMO_PYTHON', '')})},
                'optional': {'phase_b_vision': ('BOOLEAN', {'default': False}),
                             'phase_b_max_frames': ('INT', {'default': 8, 'min': 1, 'max': 64}),
                             'source_person_photo': ('IMAGE',), 'source_guide': ('IMAGE',),
                             'source_guide_fps': ('FLOAT', {'default':24.,'min':.1,'max':120.})}}
    RETURN_TYPES = ('CLIP',); FUNCTION = 'load'; CATEGORY = 'Henge/experimental'

    def load(self, molmo_path, projection, molmo_python, phase_b_vision=False, phase_b_max_frames=8,
             source_person_photo=None, source_guide=None, source_guide_fps=24.):
        if not os.path.isfile(os.path.join(molmo_path, 'tokenizer.json')):
            raise ValueError('molmo_path must be a Molmo2-8B checkpoint folder with tokenizer.json')
        if not os.path.isfile(molmo_python):
            raise ValueError('molmo_python must be the Python of the Molmo2 environment (HENGE_MOLMO_PYTHON)')
        if projection == 'bundled_default':
            projection_path = os.path.join(molmo_path, 'projections', 'molmo2-8b-h3-tap24-phaseb-pilot.safetensors')
            if not os.path.isfile(projection_path):
                raise ValueError('No bundled projector found; select an installed clip_projections file')
        else:
            projection_path = folder_paths.get_full_path('clip_projections', projection)
        vision = phase_b_vision or source_person_photo is not None or source_guide is not None
        return (MolmoProjectedCLIP(molmo_python, molmo_path, projection_path,
                vision, phase_b_max_frames, source_person_photo, source_guide, source_guide_fps),)


NODE_CLASS_MAPPINGS = {'MolmoH3EncoderLoader': MolmoH3EncoderLoader}
NODE_DISPLAY_NAME_MAPPINGS = {'MolmoH3EncoderLoader': 'Molmo2 as H3 encoder (projection, experimental)'}
