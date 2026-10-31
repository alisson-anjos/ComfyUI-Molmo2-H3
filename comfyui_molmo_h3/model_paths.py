"""Discover complete Molmo packages through ComfyUI's text_encoders search roots."""
import json
import os
from pathlib import Path
import sys
import folder_paths

PROJECTOR_NAME = 'molmo2-8b-h3-tap24-phaseb-pilot.safetensors'


def model_choices():
    choices=[];seen=set()
    for name in folder_paths.get_filename_list('text_encoders'):
        path=folder_paths.get_full_path('text_encoders',name)
        if not path:continue
        root=Path(path).parent
        if str(root) in seen or not (root/'tokenizer.json').is_file():continue
        try:config=json.loads((root/'config.json').read_text())
        except (OSError,ValueError):continue
        if 'molmo' not in json.dumps(config.get('architectures',[])).lower() and 'molmo' not in str(config.get('model_type','')).lower():continue
        seen.add(str(root));choices.append(name)
    return choices


def resolve_model(selection):
    # Existing workflows may contain the former absolute checkpoint-folder input.
    root=Path(selection)
    if not root.is_absolute():
        path=folder_paths.get_full_path('text_encoders',selection)
        if not path:raise ValueError('Select an installed Molmo2 package from text_encoders')
        root=Path(path).parent
    elif root.is_file():root=root.parent
    if not (root/'tokenizer.json').is_file() or not (root/'config.json').is_file():
        raise ValueError('Install the complete Molmo2 package, including config and tokenizer, in text_encoders')
    return str(root)


def resolve_projector(model):
    """Use the projector shipped beside the selected Molmo package, with the ComfyUI folder as fallback."""
    bundled = Path(model) / 'projections' / PROJECTOR_NAME
    if bundled.is_file():
        return str(bundled)
    installed = folder_paths.get_full_path('clip_projections', PROJECTOR_NAME)
    if installed and Path(installed).is_file():
        return installed
    raise ValueError(f'Molmo2 projector is missing: install {PROJECTOR_NAME} in the package projections folder')


def resolve_molmo_python():
    """Find the isolated Molmo runtime without exposing implementation settings on the node."""
    candidates = [os.environ.get('HENGE_MOLMO_PYTHON'), os.environ.get('MOLMO_H3_PYTHON'),
                  '/opt/molmo/bin/python', sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise ValueError('Molmo2 runtime is unavailable; configure HENGE_MOLMO_PYTHON for the ComfyUI service')
