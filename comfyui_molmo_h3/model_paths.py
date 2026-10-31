"""Discover complete Molmo packages through ComfyUI's text_encoders search roots."""
import json
from pathlib import Path
import folder_paths


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
