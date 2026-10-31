"""Validate a complete package with text and real image inference."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoProcessor
from PIL import Image
from convrot_runtime import load_model

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--image',required=True)
a=p.parse_args();torch.set_num_threads(8)
t=time.monotonic();model=load_model(a.model);processor=AutoProcessor.from_pretrained(a.model,trust_remote_code=True)
report={'load_seconds':time.monotonic()-t,'tests':[]}
for image in [False,True]:
 content=[{'type':'text','text':'Describe what you see briefly.' if image else 'What is two plus two?'}]
 if image:content.append({'type':'image','image':Image.open(a.image).convert('RGB')})
 inputs=processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
 with torch.inference_mode():
  out=model(**inputs,output_hidden_states=True,use_cache=False,logits_to_keep=1)
  h=out.hidden_states[24]
  assert torch.isfinite(h).all() and torch.isfinite(out.logits).all()
  shape=list(h.shape);del out,h
  result=model.generate(**inputs,max_new_tokens=32,do_sample=False)
 answer=processor.tokenizer.decode(result[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
 report['tests'].append({'image':image,'hidden_state_shape':shape,'answer':answer})
 print(json.dumps(report['tests'][-1]),flush=True)
report['peak_vram_bytes']=torch.cuda.max_memory_allocated()
Path(a.model,'validation.json').write_text(json.dumps(report,indent=2))
print('PASS full package inference',flush=True)
