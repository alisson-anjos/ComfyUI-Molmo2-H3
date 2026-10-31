"""Stream full Molmo2 shards into a standalone INT8 ConvRot package."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText
from convrot_runtime import rotate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    source, out=Path(args.source), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config=AutoConfig.from_pretrained(source, trust_remote_code=True)
    with init_empty_weights():
        model=AutoModelForImageTextToText.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    modules={}
    for name,m in model.named_modules():
        if isinstance(m,torch.nn.Linear) and m.in_features%16==0 and m.out_features%8==0:
            block=256
            while m.in_features%block:
                block//=4
            if block>=16:
                modules[name]=dict(in_features=m.in_features,out_features=m.out_features,block=block,bias=m.bias is not None)
    manifest=dict(format='molmo2-convrot-int8-v1', modules=modules, shards=[])
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.json','.py','.jinja','.txt') and path.name!='model.safetensors.index.json':
            shutil.copy2(path,out/path.name)
    shutil.copy2(Path(__file__).with_name('convrot_runtime.py'),out/'convrot_runtime.py')
    index=json.loads((source/'model.safetensors.index.json').read_text())
    tensors={}
    for i,filename in enumerate(sorted(set(index['weight_map'].values()))):
        with safe_open(source/filename,framework='pt',device='cpu') as f:
            for key in f.keys():
                w=f.get_tensor(key)
                name=key.removesuffix('.weight')
                if key.endswith('.weight') and name in modules:
                    spec=modules[name]
                    # Bounded GPU transient: one row chunk at a time.
                    qs,ss=[],[]
                    for chunk in w.split(256):
                        rotated=rotate(chunk.to('cuda',torch.float32),spec['block'])
                        scale=rotated.abs().amax(-1).clamp_min(1e-12)/127
                        qs.append((rotated/scale[:,None]).round().clamp(-127,127).to('cpu',torch.int8))
                        ss.append(scale.cpu())
                    tensors[name+'.qweight']=torch.cat(qs)
                    tensors[name+'.scales']=torch.cat(ss)
                else:
                    tensors[key]=w.to(torch.bfloat16) if w.is_floating_point() else w
        print('PROCESSED',filename,flush=True)
    dest='molmo2_8b_int8_convrot.safetensors'
    save_file(tensors,str(out/dest),metadata={'format':'pt','quantization':'molmo2-convrot-int8-v1'})
    manifest['shards']=[dest]
    print('SAVED',dest,flush=True)
    (out/'convrot_config.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',out,flush=True)

if __name__=='__main__': main()
