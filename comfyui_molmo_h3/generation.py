"""Native Molmo response inputs with explicit video-frame presentation metadata."""
import re
import numpy as np
from PIL import Image
from .visual_inputs import visual_array


POINT_REQUEST = re.compile(
    r'\b(point|points|track|tracking|mark|locate|position|localize|find|marc[ae]|marcar|localiz[ae]|posi[cç][aã]o|rastre[ae])\b',
    re.IGNORECASE,
)


def prompt_with_point_contract(prompt, visuals):
    """Ask for Molmo's native coordinates on point/track requests with video frames."""
    if not any(v.get('type') == 'video' for v in visuals) or not POINT_REQUEST.search(prompt):
        return prompt
    return (prompt.rstrip() + '\n\n'
        'Treat the timestamped Video frames as one ordered clip. Locate the exact target requested above; '
        'when the request identifies a Picture, follow the same visual identity from that Picture. Track '
        'that one target across the supplied video frames and keep its object ID stable. Return Molmo native '
        'visual pointing coordinates inside a '
        '<points coords="..."/> or <tracks coords="..."/> tag. Each entry must contain '
        'the 1-based visual image index, a stable object ID, and normalized x and y coordinates '
        'from 000 to 1000, for example: <points coords="2 1 500 500;3 1 520 510"/>. '
        'The visual images are presented in order; skip reference-photo indices and include only '
        'indices belonging to video frames. Do not answer with a timestamp-only list or use 0 as '
        'a placeholder. Never invent a point: if the target cannot be located, say so and return no coordinates.')


def generation_inputs(processor,prompt,visuals,max_frames=8,system_prompt=''):
    content=[];presentation=[];counts={}
    for item in visuals:
        kind=item['type'];counts[kind]=counts.get(kind,0)+1
        data=visual_array(item['data'])
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
    messages=[]
    if system_prompt and system_prompt.strip():
        messages.append({'role':'system','content':[{'type':'text','text':system_prompt.strip()}]})
    messages.append({'role':'user','content':content})
    batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
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
    note='Object IDs are scoped to this request. Points cover sampled frames only.'
    if any(p.get('kind') == 'video' for p in presentation) and not points:
        note += ' Molmo returned no parseable native coordinates; no points were fabricated. Rephrase as “Point to … in each video frame” or “Track … through the video”.'
    return {'version':1,'points':points,'description':re.sub(r'</?(?:points|tracks)\b[^>]*>','',text).strip(),
            'note':note}
