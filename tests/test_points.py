import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('generation',Path(__file__).parents[1]/'comfyui_molmo_h3/generation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_reference_image_points_never_become_video_marks():
    result=m.response_points('<points coords="1 1 500 500;2 1 250 750;3 2 600 800"/>',[
        {'kind':'image','index':1},{'kind':'video','video':1,'time_seconds':.5,'frame_width':320,'frame_height':240},
        {'kind':'video','video':2,'time_seconds':1.,'frame_width':320,'frame_height':240}])
    assert [(p['video'],p['object_id'],p['time_seconds'],p['x'],p['y']) for p in result['points']]==[(1,1,.5,.25,.75),(2,2,1.,.6,.8)]

def test_bad_coords_and_missing_image_indices_do_not_invent_tracks():
    result=m.response_points('<tracks coords="0 1 400 400;1 1 2000 100;8 1 300 500"/>',[{'kind':'video','video':1,'time_seconds':0}])
    assert result['points']==[]

def test_generation_frame_labels_preserve_original_timestamps(tmp_path):
    p=tmp_path/'v.npy';np.save(p,np.zeros((5,8,8,3),dtype=np.uint8))
    class Processor:
        def apply_chat_template(self,messages,**kw):return messages
    _,frames=m.generation_inputs(Processor(),'Locate the target',[{'type':'video','path':str(p),'timestamps':[0,.1,.2,.3,.4]}],3)
    assert [p['time_seconds'] for p in frames]==[0,.2,.4]
