import os,sys,json,importlib.util
from pathlib import Path
import pytest
if os.environ.get('COMFYUI_PATH'):sys.path.insert(0,os.environ['COMFYUI_PATH'])
pytest.importorskip('folder_paths')
spec=importlib.util.spec_from_file_location('model_paths',Path(__file__).parents[1]/'comfyui_molmo_h3/model_paths.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_dropdown_groups_shards_and_excludes_other_encoder_types(tmp_path,monkeypatch):
    roots={}
    for name,kind in [('molmo','molmo2'),('qwen','qwen3_vl')]:
        p=tmp_path/name;p.mkdir();(p/'tokenizer.json').write_text('{}');(p/'config.json').write_text(json.dumps({'model_type':kind}))
        for shard in [1,2]:roots[f'{name}/model-{shard}.safetensors']=str(p/f'model-{shard}.safetensors')
    monkeypatch.setattr(m.folder_paths,'get_filename_list',lambda key:list(roots))
    monkeypatch.setattr(m.folder_paths,'get_full_path',lambda key,name:roots.get(name))
    assert m.model_choices()==['molmo/model-1.safetensors']
    assert m.resolve_model('molmo/model-1.safetensors')==str(tmp_path/'molmo')

def test_legacy_absolute_package_paths_still_resolve(tmp_path):
    (tmp_path/'tokenizer.json').write_text('{}');(tmp_path/'config.json').write_text('{}')
    assert m.resolve_model(str(tmp_path))==str(tmp_path)

def test_missing_support_files_are_not_treated_as_complete_weights(tmp_path):
    with pytest.raises(ValueError,match='complete Molmo2 package'):m.resolve_model(str(tmp_path))
