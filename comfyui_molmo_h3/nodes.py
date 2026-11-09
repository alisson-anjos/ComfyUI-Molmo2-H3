"""Native H3 conditioning, Molmo answers and optional point-seeded SAM tracking."""
import copy,json,math
from dataclasses import replace
import torch
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo,_resize
from comfy_extras.nodes_textgen import TextGenerate
from . import MolmoProjectedCLIP
from .generation import prompt_with_point_contract


def molmo_clip(clip):
    if not isinstance(clip,MolmoProjectedCLIP):raise ValueError('Connect the Molmo2 H3 encoder loader CLIP output')
    local=copy.copy(clip);local.phase_b_vision=True
    return local


def generate_molmo(clip,prompt,visuals,max_new_tokens=1024,**sampling):
    if not isinstance(clip,MolmoProjectedCLIP):raise ValueError('Connect Molmo2 as H3 encoder (projection, experimental)')
    return clip.runtime.generate(
        prompt,visuals,max_new_tokens,clip.phase_b_max_frames,**sampling)


class Molmo2TextGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Keep the native ComfyUI Generate Text controls and port order in sync.
        # Execution below is independent: this node sends directly to Molmo2.
        return replace(TextGenerate.define_schema(),node_id='Molmo2TextGenerate',
            display_name='Molmo2 Generate Text',category='Molmo2/Analysis',
            description='Standalone Molmo2 text/image/video generation with the native Generate Text interface. It does not connect to or execute through the native Generate Text node.')

    @classmethod
    def execute(cls,clip,prompt,max_length,sampling_mode,image=None,thinking=False,use_default_template=True,
                video=None,audio=None,mtp='auto',system_prompt=''):
        if audio is not None:
            raise ValueError('Molmo2 Generate Text currently supports text, images, and video; audio input is not supported by this Molmo2 checkpoint.')
        visuals=[]
        if image is not None:
            visuals.extend({'type':'image','data':image[i:i+1]} for i in range(len(image)))
        if video is not None and len(video):
            visuals.append({'type':'video','data':video,'timestamps':[i/24. for i in range(len(video))]})
        text=prompt
        if thinking:text='Think carefully through the visual evidence before answering.\n\n'+text
        # Molmo2 uses its processor's required chat template. The native toggle
        # controls only its optional system template; system_prompt is ignored
        # when the user disables that template.
        sampling_mode=sampling_mode or {}
        do_sample=sampling_mode.get('sampling_mode')=='on'
        answer=generate_molmo(clip,text,visuals,max_length,do_sample=do_sample,
            temperature=sampling_mode.get('temperature',.7),top_k=sampling_mode.get('top_k',64),
            top_p=sampling_mode.get('top_p',.95),min_p=sampling_mode.get('min_p',.05),
            repetition_penalty=sampling_mode.get('repetition_penalty',1.05),
            seed=sampling_mode.get('seed'),presence_penalty=sampling_mode.get('presence_penalty',0.),
            system_prompt=(system_prompt or '') if use_default_template else '')
        generated=answer['text'];reasoning,sep,result=generated.partition('</think>')
        if not sep:
            if generated.lstrip().startswith('<think>'):
                reasoning,result=generated.split('<think>',1)[1],''
            else:reasoning,result='',generated
        return io.NodeOutput(result.strip(),reasoning.replace('<think>','',1).strip())


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
        for collection, prefix in (
            ('ref_images', 'ref_image_'),
            ('ref_videos', 'ref_video_'),
            ('ref_video_audios', 'ref_video_audio_'),
            ('ref_audios', 'ref_audio_'),
        ):
            values = dict(kwargs.pop(collection, {}) or {})
            dynamic = sorted(
                (key for key in kwargs if key.startswith(prefix) and key[len(prefix):].isdigit()),
                key=lambda key: int(key[len(prefix):]),
            )
            for key in dynamic:
                values[key] = kwargs.pop(key)
            if values:
                kwargs[collection] = values
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
    def execute(cls,clip,prompt,width,height,length,ref_image_size='match',ref_images=None,ref_videos=None,max_new_tokens=1024,**kwargs):
        # COMFY_AUTOGROW_V3 inputs arrive flattened by the current ComfyUI
        # executor (ref_image_1/ref_video_1), while older integrations may
        # still pass the grouped ref_images/ref_videos dictionaries.
        ref_images=dict(ref_images or {})
        ref_videos=dict(ref_videos or {})
        for name,value in kwargs.items():
            if value is None:continue
            if name.startswith('ref_image_'):
                try:ref_images[int(name[len('ref_image_'):])]=value
                except ValueError:pass
            elif name.startswith('ref_video_'):
                try:ref_videos[int(name[len('ref_video_'):])]=value
                except ValueError:pass
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
        request=prompt_with_point_contract(prompt,visuals)
        answer=generate_molmo(clip,request,visuals,max_new_tokens,do_sample=False)
        points=answer['points']
        response=answer['text']
        point_request=request != prompt
        if any(v['type']=='video' for v in visuals) and not points['points'] and point_request:
            response=('No native point coordinates were returned, so no points were created. Try a direct request '
                      'such as “Point to the person shown in Picture 1 in each frame of Video 1.”\n\nMolmo response: '+response)
        sam_prompt='' if point_request and not points['points'] else points['description']
        return io.NodeOutput(response,json.dumps(points),sam_prompt)


class MolmoH3TrackSubjectVector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MolmoH3TrackSubjectVector',
            display_name='Molmo2 Track Subject + Binding Vector',
            category='Molmo2/Analysis',
            description=('Uses the original Molmo2 vision tower to track one described source performer and extract '
                         'the layer-24 generated-token vector expected by the BFS H3 subject-binding node.'),
            inputs=[
                io.Clip.Input('clip'),
                io.Image.Input('video', tooltip='The same prepared 24-fps guide frames connected to H3 Video 1.'),
                io.String.Input('source_description', default='person in the black outfit', multiline=True),
                io.Float.Input('fps', default=24.0, min=1.0, max=120.0, step=0.1),
                io.Int.Input('max_new_tokens', default=768, min=16, max=2048),
            ],
            outputs=[
                io.String.Output(display_name='tracks_json'),
                io.String.Output(display_name='vector_json'),
                io.String.Output(display_name='Molmo response'),
            ],
        )

    @classmethod
    def execute(cls, clip, video, source_description, fps=24.0, max_new_tokens=768):
        if not isinstance(clip, MolmoProjectedCLIP):
            raise ValueError('Connect Molmo2 as H3 encoder; this node uses its original Molmo2 vision tower.')
        if video is None or len(video) < 2:
            raise ValueError('Connect the prepared guide video (at least two frames).')
        description = str(source_description or '').strip().strip('. ')
        if not description:
            raise ValueError('Enter a visual description of the source performer to track.')
        fps = float(fps)
        if fps <= 0:
            raise ValueError('fps must be positive')
        # Same request as the training labels (binding/label_molmo.py): native video at 2 fps, layer 24.
        answer = clip.runtime.track_subject(f'Track the {description}.', video, fps, layer=24,
                                            max_new_tokens=int(max_new_tokens))
        if not answer['points']:
            raise ValueError('Molmo2 returned no track points. Try a clearer source_description. Response: '
                             + answer['text'][:300])
        # Training kept every returned point (t, x, y), whatever the object id.
        track = sorted([[t, x, y] for t, _, x, y in answer['points']], key=lambda p: p[0])
        ids = sorted({o for _, o, _, _ in answer['points']})
        tracks_json = json.dumps({'tracks': track, 'object_ids': ids, 'frame_count': len(video), 'fps': fps,
                                  'description': description}, separators=(',', ':'))
        vector_json = json.dumps(answer['vector'].float().flatten().tolist(), separators=(',', ':'))
        return io.NodeOutput(tracks_json, vector_json, answer['text'])


def _subject_packet(answer, picture, description, frame_count, fps):
    if not answer['points']:
        raise ValueError(f'Molmo2 returned no track points for <Picture {picture}> ("{description}"). '
                         'Try a clearer description. Response: ' + answer['text'][:300])
    # Training kept every returned point (t, x, y), whatever the object id.
    return {'picture': picture, 'description': description, 'frame_count': frame_count, 'fps': fps,
            'tracks': sorted([[t, x, y] for t, _, x, y in answer['points']], key=lambda p: p[0]),
            'object_ids': sorted({o for _, o, _, _ in answer['points']}),
            'vector': answer['vector'].float().flatten().tolist()}


DESCRIBE_SOURCE = ('Describe the person in this image as ONE short English noun phrase of 6 to 16 words that would '
                   'identify them in a video: apparent gender and age, hair, and the colors and types of their clothes. '
                   'No leading article, no background, no pose, no full sentence.')


def _clean_description(text):
    import re
    text = ' '.join(str(text or '').strip().strip('"').split()).rstrip('. ')
    return re.sub(r'^(the|a|an)\s+', '', text, flags=re.I)


class MolmoH3TrackSubjects(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MolmoH3TrackSubjects',
            display_name='Molmo2 Track Subjects (one per Picture) + Binding Vectors',
            category='Molmo2/Analysis',
            description=('One line per person to replace, in Picture order: line 1 becomes <Picture 1>, line 2 <Picture 2>... '
                         'Each line is tracked in its own Molmo2 request (as in training), with one model load. '
                         'Connect subjects_json to BFS H3 Molmo2 Subject Binding (Multi).'),
            inputs=[
                io.Clip.Input('clip'),
                io.Image.Input('video', tooltip='The same prepared 24-fps guide frames connected to H3 Video 1.'),
                io.String.Input('source_descriptions', multiline=True,
                                default='young woman with a dark ponytail in a black crop top and black shorts',
                                tooltip='One visual description per line; line k = the person who becomes <Picture k>. '
                                        'Position words are fine here (they never reach the H3 prompt).'),
                io.Float.Input('fps', default=24.0, min=1.0, max=120.0, step=0.1),
                io.Int.Input('max_new_tokens', default=768, min=16, max=2048),
            ] + [io.Image.Input(f'source_image_{k}', optional=True,
                                tooltip=f'Optional photo/crop of the person who becomes <Picture {k}> (e.g. a crop of a '
                                        f'guide frame). Molmo2 describes it and tracks that description; overrides line {k}.')
                 for k in range(1, 5)],
            outputs=[
                io.String.Output(display_name='subjects_json'),
                io.String.Output(display_name='Molmo responses'),
            ],
        )

    @classmethod
    def execute(cls, clip, video, source_descriptions, fps=24.0, max_new_tokens=768, **photos):
        if not isinstance(clip, MolmoProjectedCLIP):
            raise ValueError('Connect Molmo2 as H3 encoder; this node uses its original Molmo2 vision tower.')
        if video is None or len(video) < 2:
            raise ValueError('Connect the prepared guide video (at least two frames).')
        lines = [l.strip().strip('. ') for l in str(source_descriptions or '').splitlines()]
        while lines and not lines[-1]: lines.pop()
        photos = {int(k.rsplit('_', 1)[1]): v for k, v in photos.items() if k.startswith('source_image_') and v is not None}
        count = max([len(lines)] + list(photos))
        lines += [''] * (count - len(lines))
        described = {}
        for k, image in sorted(photos.items()):
            # Molmo2 cannot take an image and a native video in one request: describe the photo first, then track the
            # description in the video exactly as the training labels were made.
            text = clip.runtime.generate(DESCRIBE_SOURCE, [{'type': 'image', 'data': image[:1]}], 96,
                                         clip.phase_b_max_frames, do_sample=False, repetition_penalty=1.0)['text']
            lines[k - 1] = described[k] = _clean_description(text)
        if not count or any(not l for l in lines):
            raise ValueError('Each person needs a description line or a connected source_image (in Picture order).')
        lines = [l.strip().strip('. ') for l in lines]
        if len(lines) > 9:
            raise ValueError('At most 9 people.')
        fps = float(fps)
        answers = clip.runtime.track_subjects([f'Track the {d}.' for d in lines], video, fps, layer=24,
                                              max_new_tokens=int(max_new_tokens))
        subjects = [_subject_packet(a, k, d, len(video), fps) for k, (a, d) in enumerate(zip(answers, lines), 1)]
        for s_ in subjects:
            s_['from_image'] = s_['picture'] in described
        report = '\n\n'.join(f'<Picture {s["picture"]}> {s["description"]}{" (from photo)" if s["from_image"] else ""}: {len(s["tracks"])} points, '
                               f'ids {s["object_ids"]}\n{a["text"]}' for s, a in zip(subjects, answers))
        return io.NodeOutput(json.dumps({'version': 1, 'subjects': subjects}, separators=(',', ':')), report)


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
        preview = images.clone(); m = mask.to(device=preview.device)
        color = torch.tensor([.1, .85, .35], device=preview.device, dtype=preview.dtype)
        preview[..., :3][m] = preview[..., :3][m] * (1 - opacity) + color * opacity
        return io.NodeOutput(mask.float(), preview, json.dumps(report))


def detect_cuts(image, thresh=0.22):
    """Camera cuts: frame indices where a new shot starts (mean abs diff of small gray frames, plus a histogram check)."""
    import torch.nn.functional as F
    g = image[..., :3].float().mean(-1)[:, None]
    small = F.interpolate(g, size=(36, 64), mode="area")[:, 0]
    cuts = []
    for i in range(1, small.shape[0]):
        d = (small[i] - small[i - 1]).abs().mean().item()
        h0 = torch.histc(small[i - 1], 16, 0, 1); h1 = torch.histc(small[i], 16, 0, 1)
        hd = (h0 / h0.sum() - h1 / h1.sum()).abs().sum().item() / 2
        if d > thresh * 0.6 and hd > 0.25 or d > thresh:
            cuts.append(i)
    return cuts


def shots_of(n, cuts):
    edges = [0] + [c for c in cuts if 0 < c < n] + [n]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


def ensure_sam3_model(sam_model=None, checkpoint="sam3.1_multiplex_fp16.safetensors"):
    if sam_model is not None:
        return sam_model
    import folder_paths
    import comfy.sd
    path = folder_paths.get_full_path('checkpoints', checkpoint)
    if not path:
        for name in folder_paths.get_filename_list('checkpoints'):
            if 'sam3' in name.lower() or 'sam-3' in name.lower():
                path = folder_paths.get_full_path('checkpoints', name)
                break
    if not path:
        raise ValueError(f"SAM 3.1 checkpoint '{checkpoint}' not found in models/checkpoints.")
    loaded = comfy.sd.load_checkpoint_guess_config(
        path, output_vae=False, output_clip=False,
        embedding_directory=folder_paths.get_folder_paths('embeddings')
    )
    return loaded[0]


class MolmoSAM3Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        checkpoints = folder_paths.get_filename_list('checkpoints')
        default_ckpt = 'sam3.1_multiplex_fp16.safetensors'
        if default_ckpt not in checkpoints and checkpoints:
            default_ckpt = checkpoints[0]
        return io.Schema(
            node_id='MolmoSAM3Loader',
            display_name='Molmo2 SAM 3.1 Loader',
            category='Molmo2/Tracking',
            description='Loads SAM 3.1 checkpoint for video segmentation and propagation (sam3.1_multiplex_fp16.safetensors).',
            inputs=[
                io.Combo.Input('checkpoint', options=checkpoints or [default_ckpt], default=default_ckpt),
            ],
            outputs=[
                io.Model.Output('sam_model', display_name='sam_model'),
            ],
        )

    @classmethod
    def execute(cls, checkpoint='sam3.1_multiplex_fp16.safetensors'):
        return io.NodeOutput(ensure_sam3_model(None, checkpoint))


class MolmoSAM3Propagate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MolmoSAM3Propagate',
            display_name='Molmo2 SAM 3.1 Propagate (Video Segmentation)',
            category='Molmo2/Tracking',
            description=('Bidirectional SAM 3.1 video segmentation matching BFS propagate. Tracks forward and backward '
                         'from anchor points or initial mask per camera shot (auto camera-cut detection prevents bleed across cuts).'),
            inputs=[
                io.Image.Input('images', tooltip='The video frames to segment.'),
                io.Model.Input('sam_model', optional=True, tooltip='Optional SAM 3.1 model. If unconnected, auto-loads sam3.1_multiplex_fp16.safetensors.'),
                io.String.Input('points', optional=True, force_input=True, tooltip='Points JSON from Molmo2 Vision Analyzer or Molmo tracking.'),
                io.Mask.Input('initial_mask', optional=True, tooltip='Optional initial mask on an anchor frame.'),
                io.Int.Input('anchor_frame', default=-1, min=-1, max=100000, tooltip='-1 = auto from points or middle of shot.'),
                io.Int.Input('object_id', default=0, min=0, max=10, tooltip='0 = all objects, or specific subject ID (1, 2, 3).'),
                io.Boolean.Input('detect_camera_cuts', default=True, tooltip='Automatically detect camera cuts and track each shot independently, matching BFS propagate.'),
                io.Float.Input('fps', default=24.0, min=1.0, max=120.0),
                io.Float.Input('threshold', default=0.40, min=0.05, max=0.95, step=0.05),
                io.Float.Input('opacity', default=0.45, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Mask.Output('mask', display_name='mask'),
                io.Image.Output('preview', display_name='preview'),
                io.String.Output('report', display_name='report'),
            ],
        )

    @classmethod
    def execute(cls, images, sam_model=None, points=None, initial_mask=None, anchor_frame=-1,
                object_id=0, detect_camera_cuts=True, fps=24.0, threshold=0.40, opacity=0.45):
        from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack, SAM3_TrackToMask
        import torch.nn.functional as F

        model = ensure_sam3_model(sam_model)
        n, h, w = images.shape[:3]
        fps = float(fps)
        thresh = float(threshold)
        cuts = detect_cuts(images) if (detect_camera_cuts and n > 1) else []
        shots = shots_of(n, cuts)

        pts = points_data(points, video_index=1, object_id=object_id) if points else []
        full_mask = torch.zeros((n, h, w), dtype=torch.bool, device='cpu')
        tracked_records = []

        for s0, s1 in shots:
            shot_len = s1 - s0
            if shot_len <= 0:
                continue
            shot_images = images[s0:s1]
            shot_pts = [p for p in pts if s0 <= round(float(p['time_seconds']) * fps) < s1]
            seed_mask = None
            seed_frame_in_shot = None

            if shot_pts:
                mid = (s0 + s1 - 1) / 2.0
                best_pt = min(shot_pts, key=lambda p: abs(round(float(p['time_seconds']) * fps) - mid))
                global_frame = min(s1 - 1, max(s0, round(float(best_pt['time_seconds']) * fps)))
                seed_frame_in_shot = global_frame - s0
                px = min(w - 1, max(0, round(float(best_pt['x']) * w)))
                py = min(h - 1, max(0, round(float(best_pt['y']) * h)))

                frame_img = shot_images[seed_frame_in_shot:seed_frame_in_shot + 1]
                found = SAM3_Detect.execute(
                    model=model, image=frame_img,
                    positive_coords=json.dumps([{'x': px, 'y': py}]),
                    threshold=thresh, refine_iterations=1, individual_masks=True
                )[0].detach().cpu() > 0.5
                candidates = [m for m in found if m[py, px]]
                if candidates:
                    seed_mask = max(candidates, key=lambda m: int(m.sum())).float()[None]
            elif initial_mask is not None:
                init_idx = anchor_frame if anchor_frame >= 0 else 0
                if s0 <= init_idx < s1:
                    seed_frame_in_shot = init_idx - s0
                    m_ = initial_mask[0] if initial_mask.ndim == 3 else initial_mask
                    if m_.shape != (h, w):
                        m_ = F.interpolate(m_[None, None].float(), size=(h, w), mode="nearest")[0, 0]
                    seed_mask = (m_ > 0.5).float()[None]

            if seed_mask is not None and seed_frame_in_shot is not None:
                k = seed_frame_in_shot
                fwd_data = SAM3_VideoTrack.execute(images=shot_images[k:], model=model, initial_mask=seed_mask, max_objects=1).args[0]
                fwd_m = SAM3_TrackToMask.execute(track_data=fwd_data, object_indices="").args[0].detach().cpu().float()
                if tuple(fwd_m.shape[-2:]) != (h, w):
                    fwd_m = F.interpolate(fwd_m[:, None], size=(h, w), mode='nearest')[:, 0]

                parts = [fwd_m]
                if k > 0:
                    bwd_data = SAM3_VideoTrack.execute(images=shot_images[:k + 1].flip(0), model=model, initial_mask=seed_mask, max_objects=1).args[0]
                    bwd_m = SAM3_TrackToMask.execute(track_data=bwd_data, object_indices="").args[0].detach().cpu().float()
                    if tuple(bwd_m.shape[-2:]) != (h, w):
                        bwd_m = F.interpolate(bwd_m[:, None], size=(h, w), mode='nearest')[:, 0]
                    parts = [bwd_m.flip(0)[:k], fwd_m]

                shot_full = torch.cat(parts) > 0.5
                full_mask[s0:s1] |= shot_full
                tracked_records.append({'shot': [s0, s1], 'anchor_frame': s0 + k, 'status': 'tracked'})
            else:
                tracked_records.append({'shot': [s0, s1], 'status': 'no anchor/points'})

        preview = images.clone()
        m_dev = full_mask.to(device=preview.device)
        color = torch.tensor([0.1, 0.85, 0.35], device=preview.device, dtype=preview.dtype)
        preview[..., :3][m_dev] = preview[..., :3][m_dev] * (1.0 - opacity) + color * opacity

        rep = {'shots': tracked_records, 'cuts': cuts, 'total_frames': n, 'fps': fps}
        return io.NodeOutput(full_mask.float(), preview, json.dumps(rep))


VERIFY_SAME_PERSON = ('Picture 1 shows the reference person. '
                      'Picture 2 shows a candidate person from another frame/video. '
                      'Do Picture 1 and Picture 2 show the exact same individual wearing the exact same outfit or costume? '
                      'Answer with a single word: yes or no, followed by a brief explanation.')


class Molmo2VisionAnalyzer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='Molmo2VisionAnalyzer',
            display_name='Molmo2 Vision Analyzer & Tracker',
            category='Molmo2/Analysis',
            description=('Unified Molmo2 analysis and tracking node. Select a task (track by text, track by photo, multi-subject, '
                         'describe, verify, or custom prompt) and feed video/images. Outputs response text, points for Molmo2 Draw Video Points, '
                         'subjects_json for H3 conditioning, marked video preview, and binding vector.'),
            inputs=[
                io.Clip.Input('clip'),
                io.Combo.Input('task', options=[
                    'Track by text',
                    'Track by Reference Image 1',
                    'Track Multi-Subject (Image 1..3 -> Picture 1..3)',
                    'Describe person / subject',
                    'Describe video action',
                    'Verify same person (Image 1 vs Image 2 / Video)',
                    'Locate / Spatial points',
                    'Custom task / prompt',
                ], default='Track by text'),
                io.String.Input('prompt', multiline=True, default='',
                                tooltip='Prompt or target description (e.g. "boxer in red shorts", or custom prompt). For multi-subject, one description per line.'),
                io.Image.Input('video', optional=True, tooltip='Guide video frames to track or analyze.'),
                io.Image.Input('image_1', optional=True, tooltip='Reference image / photo 1 (for tracking, description, or comparison).'),
                io.Image.Input('image_2', optional=True, tooltip='Reference image / photo 2.'),
                io.Image.Input('image_3', optional=True, tooltip='Reference image / photo 3.'),
                io.Float.Input('fps', default=24.0, min=1.0, max=120.0, step=0.1),
                io.Int.Input('max_new_tokens', default=768, min=16, max=4096),
                io.Boolean.Input('draw_points_preview', default=True, tooltip='Draw colored points on the video output.'),
            ],
            outputs=[
                io.String.Output('response', display_name='response'),
                io.String.Output('points', display_name='points', tooltip='JSON points compatible with Molmo2 Draw Video Points and SAM3.'),
                io.String.Output('subjects_json', display_name='subjects_json', tooltip='Multi-subject tracking JSON for H3 Character Swap conditioning.'),
                io.Image.Output('marked_video', display_name='marked_video', tooltip='Preview of the video with tracked points drawn, or original image.'),
                io.String.Output('binding_vector', display_name='binding_vector', tooltip='Layer-24 token vector for H3 subject attention binding.'),
            ],
        )

    @classmethod
    def execute(cls, clip, task='Track by text', prompt='', video=None, image_1=None, image_2=None, image_3=None,
                fps=24.0, max_new_tokens=768, draw_points_preview=True):
        if not isinstance(clip, MolmoProjectedCLIP):
            raise ValueError('Connect Molmo2 as H3 encoder (Auto Download Load Molmo2).')
        fps = float(fps)
        tokens = int(max_new_tokens)
        prompt_text = str(prompt or '').strip()

        resp_text = ''
        pts_list = []
        subjects_list = []
        vec_list = []

        if task == 'Track by text':
            if video is None or len(video) < 2:
                raise ValueError('Track by text requires a video input with at least two frames.')
            desc = prompt_text or 'person'
            query = f'Track the {desc}.'
            ans = clip.runtime.track_subjects([query], video, fps=fps, layer=24, max_new_tokens=tokens)[0]
            resp_text = ans.get('text', '')
            pts_list = ans.get('points', [])
            subjects_list = [_subject_packet(ans, 1, desc, len(video), fps)]
            vec = ans.get('vector')
            if vec is not None:
                vec_list = vec.float().flatten().tolist()

        elif task == 'Track by Reference Image 1':
            if video is None or len(video) < 2:
                raise ValueError('Track by Reference Image 1 requires a video input.')
            if image_1 is None:
                raise ValueError('Connect image_1 (photo or crop of the person to track).')
            if prompt_text:
                desc = prompt_text
            else:
                raw = clip.runtime.generate(DESCRIBE_SOURCE, [{'type': 'image', 'data': image_1[:1]}], 96,
                                             clip.phase_b_max_frames, do_sample=False, repetition_penalty=1.0)['text']
                desc = _clean_description(raw)
            query = f'Track the {desc}.'
            ans = clip.runtime.track_subjects([query], video, fps=fps, layer=24, max_new_tokens=tokens)[0]
            resp_text = f"Target description: {desc}\n\nMolmo tracking response:\n" + ans.get('text', '')
            pts_list = ans.get('points', [])
            sub = _subject_packet(ans, 1, desc, len(video), fps)
            sub['from_image'] = True
            subjects_list = [sub]
            vec = ans.get('vector')
            if vec is not None:
                vec_list = vec.float().flatten().tolist()

        elif task == 'Track Multi-Subject (Image 1..3 -> Picture 1..3)':
            if video is None or len(video) < 2:
                raise ValueError('Multi-Subject tracking requires a video input.')
            lines = [l.strip().strip('. ') for l in prompt_text.splitlines() if l.strip()]
            photos = {1: image_1, 2: image_2, 3: image_3}
            photos = {k: v for k, v in photos.items() if v is not None}
            count = max([len(lines)] + list(photos.keys()) if photos else [len(lines)])
            if not count:
                raise ValueError('Provide description lines in prompt or connect reference images (image_1, image_2, image_3).')
            lines += [''] * (count - len(lines))
            described = {}
            for k, img in sorted(photos.items()):
                if k - 1 >= len(lines) or not lines[k - 1]:
                    raw = clip.runtime.generate(DESCRIBE_SOURCE, [{'type': 'image', 'data': img[:1]}], 96,
                                                 clip.phase_b_max_frames, do_sample=False, repetition_penalty=1.0)['text']
                    lines[k - 1] = described[k] = _clean_description(raw)
            queries = [f'Track the {d}.' for d in lines if d]
            answers = clip.runtime.track_subjects(queries, video, fps=fps, layer=24, max_new_tokens=tokens)
            subjects_list = []
            all_pts = []
            for k, (a, d) in enumerate(zip(answers, lines), 1):
                s = _subject_packet(a, k, d, len(video), fps)
                s['from_image'] = k in described
                subjects_list.append(s)
                for t, _, x, y in a.get('points', []):
                    all_pts.append((t, k, x, y))
            pts_list = sorted(all_pts, key=lambda p: p[0])
            resp_text = '\n\n'.join(f'<Picture {s["picture"]}> {s["description"]}: {len(s["tracks"])} points\n{a["text"]}'
                                    for s, a in zip(subjects_list, answers))
            if subjects_list and 'vector' in subjects_list[0]:
                vec_list = subjects_list[0]['vector']

        elif task == 'Describe person / subject':
            src = image_1[:1] if image_1 is not None else (video[:1] if video is not None else None)
            if src is None:
                raise ValueError('Connect image_1 or video to describe the person.')
            q = prompt_text or DESCRIBE_SOURCE
            ans = clip.runtime.generate(q, [{'type': 'image', 'data': src}], tokens,
                                        clip.phase_b_max_frames, do_sample=False, repetition_penalty=1.0)
            resp_text = ans.get('text', '')

        elif task == 'Describe video action':
            if video is None or not len(video):
                raise ValueError('Describe video action requires a video input.')
            q = prompt_text or 'Describe the action, scene, camera motion, and events in this video in detail.'
            timestamps = [i / fps for i in range(len(video))]
            ans = clip.runtime.generate(q, [{'type': 'video', 'data': video, 'timestamps': timestamps}],
                                        tokens, clip.phase_b_max_frames, do_sample=False)
            resp_text = ans.get('text', '')

        elif task == 'Verify same person (Image 1 vs Image 2 / Video)':
            if image_1 is None:
                raise ValueError('Verify requires image_1 as the reference person.')
            cand = image_2[:1] if image_2 is not None else (video[:1] if video is not None else None)
            if cand is None:
                raise ValueError('Connect image_2 or video as candidate.')
            q = prompt_text or VERIFY_SAME_PERSON
            visuals = [{'type': 'image', 'data': image_1[:1]}, {'type': 'image', 'data': cand}]
            ans = clip.runtime.generate(q, visuals, 128, clip.phase_b_max_frames, do_sample=False, repetition_penalty=1.0)
            resp_text = ans.get('text', '')

        elif task in ('Locate / Spatial points', 'Custom task / prompt'):
            visuals = []
            if image_1 is not None:
                visuals.append({'type': 'image', 'data': image_1[:1]})
            if image_2 is not None:
                visuals.append({'type': 'image', 'data': image_2[:1]})
            if image_3 is not None:
                visuals.append({'type': 'image', 'data': image_3[:1]})
            if video is not None and len(video):
                timestamps = [i / fps for i in range(len(video))]
                visuals.append({'type': 'video', 'data': video, 'timestamps': timestamps})
            q = prompt_text or ('Point to each person in each frame.' if task == 'Locate / Spatial points' else 'Describe the image.')
            if task == 'Locate / Spatial points':
                q = prompt_with_point_contract(q, visuals)
            ans = clip.runtime.generate(q, visuals, tokens, clip.phase_b_max_frames, do_sample=False)
            resp_text = ans.get('text', '')
            raw_pts = ans.get('points', {})
            if isinstance(raw_pts, dict) and raw_pts.get('points'):
                for p in raw_pts['points']:
                    pts_list.append((float(p.get('time_seconds', 0.0)), int(p.get('object_id', 1)), float(p.get('x', 0.0)), float(p.get('y', 0.0))))

        # Prepare formatted points_json matching MolmoH3DrawPoints spec
        formatted_pts = [
            {'time_seconds': float(t), 'object_id': int(o), 'x': float(x), 'y': float(y), 'video': 1}
            for t, o, x, y in pts_list
        ]
        points_packet = {
            'points': formatted_pts,
            'description': prompt_text or task,
            'frame_count': len(video) if video is not None else 1,
            'fps': fps
        }
        points_json = json.dumps(points_packet, separators=(',', ':'))
        subjects_json = json.dumps({'version': 1, 'subjects': subjects_list}, separators=(',', ':')) if subjects_list else ''
        vector_json = json.dumps(vec_list, separators=(',', ':')) if vec_list else ''

        # Prepare marked preview
        if video is not None and len(video):
            if draw_points_preview and formatted_pts:
                marked_video = draw_points(video, formatted_pts, fps)
            else:
                marked_video = video
        elif image_1 is not None:
            marked_video = image_1
        else:
            marked_video = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return io.NodeOutput(resp_text, points_json, subjects_json, marked_video, vector_json)


NODE_CLASS_MAPPINGS = {
    c.__name__: c for c in [
        MolmoH3ReferenceToVideo, Molmo2TextGenerate, MolmoH3AnalyzeReferences,
        MolmoH3TrackSubjectVector, MolmoH3TrackSubjects, MolmoH3DrawPoints,
        MolmoH3SAM3Track, MolmoSAM3Loader, MolmoSAM3Propagate, Molmo2VisionAnalyzer
    ]
}
