"""Native Molmo response inputs with explicit video-frame presentation metadata."""
import re
import numpy as np
from PIL import Image


def generation_inputs(processor,prompt,visuals,max_frames=8):
    content=[];presentation=[];counts={}
    for item in visuals:
        kind=item['type'];counts[kind]=counts.get(kind,0)+1
        data=np.load(item['path'],allow_pickle=False)
        if kind in ('image','source_person'):
            label=('SourcePerson' if kind=='source_person' else 'Picture')+f' {counts[kind]}'
            content.extend([{'type':'text','text':label},{'type':'image','image':Image.fromarray(data[0] if data.ndim==4 else data)}])
            presentation.append({'kind':kind,'index':counts[kind]})
        elif kind=='video':
            for i in np.linspace(0,len(data)-1,min(len(data),max_frames),dtype=int):
                t=float(item['timestamps'][i]);h,w=data[i].shape[:2]
                content.extend([{'type':'text','text':f"Video {counts[kind]}, frame at {t:.3f}s"},{'type':'image','image':Image.fromarray(data[i])}])
                presentation.append({'kind':'video','video':counts[kind],'time_seconds':t,'frame_width':w,'frame_height':h})
    content.append({'type':'text','text':prompt})
    batch=processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
    return batch,presentation


def response_points(text,presentation):
    points=[]
    for tag in re.finditer(r'<(?:points|tracks)[^>]*coords="([0-9\t:;, .]+)"',text):
        for group in re.finditer(r'(?:^|\t|:|,|;)([0-9.]+) ([0-9. ]+)',tag.group(1)):
            index=float(group.group(1))
            if not index.is_integer() or not 1<=index<=len(presentation):continue
            frame=presentation[int(index)-1]
            if frame['kind']!='video':continue
            for point in re.finditer(r'([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})',group.group(2)):
                oid,x,y=map(int,point.groups());x/=1000;y/=1000
                if 0<=x<=1 and 0<=y<=1:
                    points.append({**frame,'object_id':oid,'x':x,'y':y})
    # Keep only observed times; never fabricate per-frame tracking by interpolation.
    return {'version':1,'points':points,'description':re.sub(r'</?(?:points|tracks)\b[^>]*>','',text).strip(),
            'note':'Object IDs are scoped to this request. Points cover sampled frames only.'}
