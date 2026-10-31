"""Native video tracking comparison; no SAM, segmentation or H3 generation."""
import argparse,json,re,time,sys
from pathlib import Path
import torch
from transformers import AutoProcessor,AutoModelForImageTextToText
from molmo_utils import process_vision_info

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--video',required=True);p.add_argument('--out',required=True);p.add_argument('--variant',required=True)
a=p.parse_args();torch.set_num_threads(8)
folder=Path(a.model);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (folder/'convrot_config.json').exists():
 sys.path.insert(0,str(folder));from convrot_runtime import load_model
 model=load_model(folder)
else:
 model=AutoModelForImageTextToText.from_pretrained(folder,trust_remote_code=True,dtype=torch.bfloat16,device_map='cuda',local_files_only=True).eval()
processor=AutoProcessor.from_pretrained(folder,trust_remote_code=True,local_files_only=True)
results=[]
for target,query in [('red','Track the man wearing red shorts.'),('black','Track the man wearing black shorts.')]:
 messages=[{'role':'user','content':[dict(type='text',text=query),dict(type='video',video=a.video,max_fps=8)]}]
 _,videos,kw=process_vision_info(messages);videos,metadata=zip(*videos)
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 inputs=processor(videos=list(videos),video_metadata=list(metadata),text=text,padding=True,return_tensors='pt',**kw).to('cuda')
 started=time.monotonic()
 with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=2048,do_sample=False)
 answer=processor.tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
 points=[]
 for coord in re.finditer(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>',answer):
  for frame in re.finditer(r'(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)',coord.group(1)):
   for point in re.finditer(r'([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})',frame.group(2)):
    points.append(dict(time=float(frame.group(1)),id=int(point.group(1)),x=int(point.group(2))/1000,y=int(point.group(3))/1000))
 record=dict(variant=a.variant,target=target,prompt=query,generated_text=answer,points=points,seconds=time.monotonic()-started,input_tokens=inputs['input_ids'].shape[1],finite_coordinates=all(0<=q['x']<=1 and 0<=q['y']<=1 for q in points))
 results.append(record);(out/(a.variant+'.json')).write_text(json.dumps(results,indent=2))
 print(a.variant,target,'points',len(points),'seconds',record['seconds'],answer[:200],flush=True)
 del inputs,generated
 torch.cuda.empty_cache()
print('DONE',a.variant,flush=True)
