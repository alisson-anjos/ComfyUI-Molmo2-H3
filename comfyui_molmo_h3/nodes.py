"""Native H3 conditioning, Molmo answers and optional point-seeded SAM tracking."""
import copy,json,math
from dataclasses import replace
import torch
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo,_resize
import comfy.model_management as mm
from . import MolmoProjectedCLIP
from .generation import prompt_with_point_contract


def molmo_clip(clip):
    if not isinstance(clip,MolmoProjectedCLIP):raise ValueError('Connect the Molmo2 H3 encoder loader CLIP output')
    local=copy.copy(clip);local.phase_b_vision=True
    return local


def points_data(value,video_index=1,object_id=0):
    packet=json.loads(value) if isinstance(value,str) else value
    points=packet.get('points',[])
    return [p for p in points if int(p.get('video',1))==video_index and (object_id==0 or int(p['object_id'])==object_id)
            and all(math.isfinite(float(p[k])) for k in ['x','y','time_seconds'])
            and 0<=float(p['x'])<=1 and 0<=float(p['y'])<=1 and float(p['time_seconds'])>=0]


def draw_points(images,points,fps,radius=6):
    result=images.clone();h,w=images.shape[1:3]
    colors=[(1.,.2,.15),(.1,.9,.3),(.15,.55,1.),(1.,.8,.1)]
    for p in points:
        k=round(float(p['time_seconds'])*fps)
        if not 0<=k<len(result):continue
        x=min(w-1,round(float(p['x'])*w));y=min(h-1,round(float(p['y'])*h))
        # Work only in the small marker window, not a full frame grid per point.
        l,r=max(0,x-radius),min(w,x+radius+1);t,b=max(0,y-radius),min(h,y+radius+1)
        yy,xx=torch.meshgrid(torch.arange(t,b,device=result.device),torch.arange(l,r,device=result.device),indexing='ij')
        mask=(xx-x)**2+(yy-y)**2<=radius**2
        color=torch.tensor(colors[(int(p['object_id'])-1)%len(colors)],device=result.device,dtype=result.dtype)
        view=result[k,t:b,l:r,:3];view[mask]=color
    return result


class MolmoH3ReferenceToVideo(MiniMaxH3ReferenceToVideo):
    @classmethod
    def define_schema(cls):
        return replace(MiniMaxH3ReferenceToVideo.define_schema(),node_id='MolmoH3ReferenceToVideo',
            display_name='MiniMax H3 Reference to Video (Molmo2)',category='Molmo2/H3',
            description='Native H3 reference conditioning and AV latent with the projected Molmo CLIP. Same reference numbering and VAE payload as the native node.')
    @classmethod
    def execute(cls,clip,**kwargs):
        return MiniMaxH3ReferenceToVideo.execute(clip=molmo_clip(clip),**kwargs)


class MolmoH3AnalyzeReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        native=MiniMaxH3ReferenceToVideo.define_schema()
        inputs=[x for x in native.inputs if x.id in ['clip','prompt','width','height','length','ref_image_size','ref_images','ref_videos']]
        inputs += [io.Int.Input('max_new_tokens',default=1024,min=16,max=8192)]
        return io.Schema(node_id='MolmoH3AnalyzeReferences',display_name='Molmo2 Analyze References / Points',category='Molmo2/Analysis',
            description='Generate the native Molmo answer and sampled video points. Pictures/videos use the same names as H3. No VAE or H3 sampling.',inputs=inputs,
            outputs=[io.String.Output(display_name='response'),io.String.Output(display_name='points'),io.String.Output(display_name='sam_prompt')])
    @classmethod
    def execute(cls,clip,prompt,width,height,length,ref_image_size='match',ref_images=None,ref_videos=None,max_new_tokens=1024):
        clip=molmo_clip(clip);visuals=[]
        if clip.source_person_photo is not None:visuals.append({'type':'source_person','data':clip.source_person_photo[:1]})
        for img in (ref_images or {}).values():
            if img is None:continue
            h,w=img.shape[1:3];scale=min(1.,math.sqrt(width*height/(w*h))) if ref_image_size=='match' else min(1.,2048/min(h,w))
            visuals.append({'type':'image','data':_resize(img[:1],max(32,round(w*scale/32)*32),max(32,round(h*scale/32)*32),'disabled')})
        videos=list((ref_videos or {}).values())
        if clip.source_guide is not None and not videos:videos=[clip.source_guide]
        for frames in videos:
            if frames is None or not len(frames):continue
            n=min(len(frames),length);ids=torch.linspace(0,n-1,min(n,clip.phase_b_max_frames)).round().long()
            h,w=frames.shape[1:3];scale=min(1.,math.sqrt(width*height/(w*h)))
            selected=_resize(frames[ids],max(32,round(w*scale/32)*32),max(32,round(h*scale/32)*32),'disabled')
            visuals.append({'type':'video','data':selected,'timestamps':[int(i)/clip.source_guide_fps for i in ids]})
        mm.unload_all_models();mm.soft_empty_cache()
        request=prompt_with_point_contract(prompt,visuals)
        answer=clip.generate_prompt(request,visuals,max_new_tokens)
        points=answer['points']
        response=answer['text']
        point_request=request != prompt
        if any(v['type']=='video' for v in visuals) and not points['points'] and point_request:
            response=('No native point coordinates were returned, so no points were created. Try a direct request '
                      'such as “Point to the person shown in Picture 1 in each frame of Video 1.”\n\nMolmo response: '+response)
        sam_prompt='' if point_request and not points['points'] else points['description']
        return io.NodeOutput(response,json.dumps(points),sam_prompt)


class MolmoH3DrawPoints(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MolmoH3DrawPoints',display_name='Molmo2 Draw Video Points',category='Molmo2/Tracking',
            inputs=[io.Image.Input('images'),io.String.Input('points',force_input=True),io.Float.Input('fps',default=24.,min=.1,max=120.),
                    io.Int.Input('video_index',default=1,min=1),io.Int.Input('object_id',default=0,min=0,tooltip='0 draws all returned IDs'),io.Int.Input('radius',default=6,min=1,max=64)],
            outputs=[io.Image.Output(display_name='marked_video')],description='Draw only observed points at their timestamps. SAM tracking is a separate step.')
    @classmethod
    def execute(cls,images,points,fps,video_index,object_id,radius):
        return io.NodeOutput(draw_points(images,points_data(points,video_index,object_id),fps,radius))


def segment_points(images,points,model,fps,detect,track,unpack,conditioning=None,bounds=None):
    n,h,w=images.shape[:3];masks=torch.zeros((n,h,w),dtype=torch.bool);warnings=[] if points else ['No matching video points were returned'];tracked=[]
    bounds=sorted({0,n,*[int(x) for x in (bounds or []) if 0<int(x)<n]})
    for oid in sorted({int(p['object_id']) for p in points}):
        for left,right in zip(bounds,bounds[1:]):
            anchors=sorted([p for p in points if int(p['object_id'])==oid and left<=round(float(p['time_seconds'])*fps)<right],key=lambda p:p['time_seconds'])
            seeded=False
            for p in anchors:
                k=round(float(p['time_seconds'])*fps);x=min(w-1,round(float(p['x'])*w));y=min(h-1,round(float(p['y'])*h))
                found=detect(model=model,image=images[k:k+1],conditioning=conditioning,positive_coords=json.dumps([{'x':x,'y':y}]),refine_iterations=1,individual_masks=True)[0].detach().cpu()>.5
                candidates=[m for m in found if m[y,x]]
                if not candidates:continue
                seed=max(candidates,key=lambda m:int(m.sum())).float()[None];full=torch.zeros((n,h,w),dtype=torch.bool);full[k]=seed[0].bool()
                for indices in [list(range(k,right)),list(range(k,left-1,-1))]:
                    if len(indices)<2:continue
                    data=track(images=images[indices],model=model,initial_mask=seed,max_objects=1)[0]
                    if data.get('packed_masks') is None:continue
                    propagated=unpack(data['packed_masks']).detach().cpu().float()[:,0]
                    if tuple(propagated.shape[-2:])!=(h,w):propagated=torch.nn.functional.interpolate(propagated[:,None],size=(h,w),mode='nearest')[:,0]
                    full[indices]=propagated>.5
                full[k]=seed[0].bool();masks|=full;tracked.append({'object_id':oid,'anchor_frame':k,'segment':[left,right]});seeded=True;break
            if anchors and not seeded:warnings.append(f'Object {oid}: SAM found no mask containing its points in frames {left}:{right}')
    return masks,{'tracked':tracked,'warnings':warnings,'note':'Object IDs belong to one Molmo request; automatic re-identification and camera-cut detection are not performed.'}


class MolmoH3SAM3Track(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MolmoH3SAM3Track',display_name='Molmo2 Points to SAM 3.1 Track',category='Molmo2/Tracking',
            inputs=[io.Model.Input('sam_model'),io.Image.Input('images'),io.String.Input('points',force_input=True),
                    io.Float.Input('fps',default=24.,min=.1,max=120.),io.Int.Input('video_index',default=1,min=1),
                    io.Int.Input('object_id',default=0,min=0,tooltip='0 tracks all returned IDs; choose one ID for a separate mask'),
                    io.Float.Input('opacity',default=.45,min=0.,max=1.),io.String.Input('cut_frames',default='',tooltip='Optional comma-separated camera-cut frame indices'),
                    io.Conditioning.Input('sam_conditioning',optional=True,tooltip='Optional SAM text conditioning; encode sam_prompt with the SAM checkpoint CLIP')],
            outputs=[io.Mask.Output(display_name='mask'),io.Image.Output(display_name='preview'),io.String.Output(display_name='report')])
    @classmethod
    def execute(cls,sam_model,images,points,fps,video_index,object_id,opacity,cut_frames='',sam_conditioning=None):
        from comfy_extras.nodes_sam3 import SAM3_Detect,SAM3_VideoTrack
        from comfy.ldm.sam3.tracker import unpack_masks
        selected=points_data(points,video_index,object_id)
        mask,report=segment_points(images,selected,sam_model,fps,SAM3_Detect.execute,SAM3_VideoTrack.execute,unpack_masks,sam_conditioning,[int(x.strip()) for x in cut_frames.split(',') if x.strip()])
        preview=images.clone();m=mask.to(device=preview.device)
        color=torch.tensor([.1,.85,.35],device=preview.device,dtype=preview.dtype)
        preview[...,:3][m]=preview[...,:3][m]*(1-opacity)+color*opacity
        return io.NodeOutput(mask.float(),preview,json.dumps(report))


NODE_CLASS_MAPPINGS={c.__name__:c for c in [MolmoH3ReferenceToVideo,MolmoH3AnalyzeReferences,MolmoH3DrawPoints,MolmoH3SAM3Track]}
