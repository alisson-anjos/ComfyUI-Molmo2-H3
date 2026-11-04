"""Standalone symmetric INT8 ConvRot inference for Molmo2 (no ai-toolkit dependency)."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def rotate(x, block):
    if block == 1:
        return x
    # Regular Hadamard R4, Kronecker-powered; symmetric and self-inverse.
    r = x.new_tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]) / 2
    h = r
    while h.shape[0] < block:
        h = torch.kron(h, r)
    return (x.reshape(-1, block) @ h).reshape(x.shape)


class ConvRotLinear(nn.Module):
    def __init__(self, in_features, out_features, block, bias=False):
        super().__init__()
        self.in_features, self.out_features, self.block = in_features, out_features, block
        self.register_buffer('qweight', torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scales', torch.empty(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False) if bias else None

    @property
    def weight(self):
        return rotate(self.qweight.float() * self.scales[:, None], self.block).to(torch.bfloat16)

    def forward(self, x):
        a = rotate(x.float(), self.block).reshape(-1, self.in_features)
        s = a.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
        q = (a / s).round().clamp(-127,127).to(torch.int8)
        if x.is_cuda:
            n = q.shape[0]
            q = F.pad(q, (0,0,0,max(32, ((n+7)//8)*8)-n))
            y = torch._int_mm(q.contiguous(), self.qweight.t().contiguous())[:n].float()
            y = y * s * self.scales[None]
        else:
            y = F.linear(q.float(), self.qweight.float()) * s * self.scales[None]
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


def load_model(folder, device='cpu'):
    """Load the complete quantized package without downloading original weights."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText
    from safetensors.torch import load_file
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # Molmo2's remote modeling code targets Transformers 4.x and asks for the
    # standard `default` RoPE initializer. Transformers 5.x removed that entry
    # from its registry, although the standard formula is unchanged.
    if 'default' not in ROPE_INIT_FUNCTIONS:
        def default_rope(config, device=None, seq_len=None):
            base = getattr(config, 'rope_theta', 10000.0)
            head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * getattr(config, 'partial_rotary_factor', 1.0))
            exponent = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
            return 1.0 / (base ** exponent), 1.0
        ROPE_INIT_FUNCTIONS['default'] = default_rope

    folder = Path(folder)
    manifest = json.loads((folder/'convrot_config.json').read_text())
    if manifest.get('format') != 'molmo2-convrot-int8-v1':
        raise ValueError('Unsupported ConvRot package format')
    config = AutoConfig.from_pretrained(folder, trust_remote_code=True)
    # Transformers 5.22+ ships Molmo2, and modeling_molmo2.py registers it again on import. Packages downloaded before
    # the repository added exist_ok=True to that call fail with "already used by a Transformers model".
    register = AutoModelForImageTextToText.register
    AutoModelForImageTextToText.register = lambda config_class, model_class, exist_ok=True: register(config_class, model_class, exist_ok=exist_ok)
    try:
        with init_empty_weights():
            model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
            for name, spec in manifest['modules'].items():
                parent, _, leaf = name.rpartition('.')
                setattr(model.get_submodule(parent), leaf, ConvRotLinear(**spec))
    finally:
        del AutoModelForImageTextToText.register
    expected = set(model.state_dict())
    seen = set()
    for shard in manifest['shards']:
        state = load_file(str(folder/shard))
        if seen.intersection(state):
            raise ValueError('Duplicate tensors across shards')
        seen.update(state)
        result = model.load_state_dict(state, strict=False, assign=True)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
    if expected != seen:
        raise ValueError(f'Tensor coverage mismatch: missing={expected-seen}, extra={seen-expected}')
    remaining = [n for n,p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if remaining:
        raise ValueError(f'Uninitialized tensors: {remaining}')
    from transformers import GenerationConfig
    if (folder/'generation_config.json').is_file():
        model.generation_config = GenerationConfig.from_pretrained(folder)
    return model.eval().to(device)
