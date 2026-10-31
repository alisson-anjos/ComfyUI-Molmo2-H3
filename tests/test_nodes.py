import sys,json,importlib,os
import pytest
from pathlib import Path
import torch
if os.environ.get('COMFYUI_PATH'):sys.path.insert(0,os.environ['COMFYUI_PATH'])
pytest.importorskip('folder_paths')
sys.path.insert(0,str(Path(__file__).parents[1]))
from comfyui_molmo_h3.nodes import segment_points,draw_points,MolmoH3ReferenceToVideo
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

def test_reference_node_keeps_native_inputs_and_outputs():
    native=MiniMaxH3ReferenceToVideo.define_schema();ours=MolmoH3ReferenceToVideo.define_schema()
    assert [x.id for x in native.inputs]==[x.id for x in ours.inputs]
    assert [x.io_type for x in native.outputs]==[x.io_type for x in ours.outputs]

def test_point_preview_does_not_invent_between_sample_tracks():
    images=torch.zeros(5,10,20,3)
    result=draw_points(images,[dict(time_seconds=.5,x=.75,y=.2,object_id=1)],4,1)
    assert result[2,2,15].any() and not result[1].any() and not images.any()

def test_point_segmentation_tracks_both_directions_and_stops_at_cut():
    images=torch.zeros(8,8,16,3);images[:,0,0,0]=torch.arange(8)
    calls=[]
    def detect(**kw):
        assert json.loads(kw['positive_coords'])==[{'x':4,'y':4}]
        mask=torch.zeros(1,8,16);mask[:,2:6,2:7]=1;return mask,None
    def track(**kw):
        calls.append(kw['images'][:,0,0,0].tolist())
        return ({'packed_masks':kw['initial_mask'][:,None].repeat(len(kw['images']),1,1,1)},)
    mask,report=segment_points(images,[dict(object_id=1,time_seconds=3/24,x=.25,y=.5)],object(),24,detect,track,lambda x:x,bounds=[2,6])
    assert calls==[[3,4,5],[3,2]]
    assert mask[2:6].any() and not mask[:2].any() and not mask[6:].any()
    assert report['tracked'][0]['object_id']==1

def test_sam_misses_are_empty_and_reported_instead_of_selecting_another_person():
    def detect(**kw):return torch.zeros(1,8,16),None
    def unexpected(**kw):raise AssertionError('No tracking without a seed under the point')
    mask,report=segment_points(torch.zeros(5,8,16,3),[dict(object_id=2,time_seconds=0,x=.5,y=.5)],object(),24,detect,unexpected,lambda x:x)
    assert not mask.any() and not report['tracked'] and report['warnings']
