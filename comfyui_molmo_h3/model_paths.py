"""Discover complete Molmo packages through ComfyUI's text_encoders search roots."""
import json
import os
from pathlib import Path
import threading
import folder_paths

PROJECTOR_NAME = 'molmo2-8b-h3-tap24-phaseb-pilot.safetensors'
MODEL_REPO = 'Alissonerdx/Molmo2-8B-Minimax-H3'
MODEL_FOLDER = 'Molmo2-8B-Minimax-H3'
PACKAGE_LOCK = threading.Lock()
PACKAGE_FILES = [
    'config.json', 'convrot_config.json', 'generation_config.json', 'processor_config.json',
    'preprocessor_config.json', 'video_preprocessor_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
    'merges.txt', 'vocab.json', 'configuration_molmo2.py', 'modeling_molmo2.py',
    'image_processing_molmo2.py', 'processing_molmo2.py', 'video_processing_molmo2.py',
    'convrot_runtime.py', 'molmo2_8b_int8_convrot.safetensors', f'projections/{PROJECTOR_NAME}',
]


def model_choices():
    """Retained for older workflows; the current loader has one fixed public model."""
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


def resolve_model(selection=None):
    """Download the one supported public package on first use and return its local folder."""
    # Keep a migration path for old workflows that stored the former dropdown selection.
    if selection:
        root=Path(selection)
        if not root.is_absolute():
            path=folder_paths.get_full_path('text_encoders',selection)
            if not path:raise ValueError('Selected Molmo2 package is not installed')
            root=Path(path).parent
        elif root.is_file():root=root.parent
        if not (root/'tokenizer.json').is_file() or not (root/'config.json').is_file():
            raise ValueError('Selected folder is not a complete Molmo2 package')
        return str(root)
    roots = folder_paths.get_folder_paths('text_encoders')
    if not roots:
        raise ValueError('ComfyUI has no text_encoders model folder configured')
    candidates=[]
    for base in map(Path, roots):
        if (base/'convrot_config.json').is_file():candidates.append(base)
        if base.is_dir():
            candidates.extend(child for child in base.iterdir() if child.is_dir() and (child/'convrot_config.json').is_file())
    root = next((p for p in candidates if (p/'molmo2_8b_int8_convrot.safetensors').is_file()),
                Path(roots[0]) / MODEL_FOLDER)

    def missing_files():
        missing=[]
        for name in PACKAGE_FILES:
            path=root/name
            if not path.is_file():
                missing.append(name);continue
            # Detect Git-LFS pointer files from a plain git clone without fetched LFS objects.
            if name == 'molmo2_8b_int8_convrot.safetensors' and path.stat().st_size <= 1_000_000_000:
                missing.append(name)
            elif name == f'projections/{PROJECTOR_NAME}' and path.stat().st_size <= 1_000_000:
                missing.append(name)
        return missing

    with PACKAGE_LOCK:
        missing=missing_files()
        if missing:
            from huggingface_hub import snapshot_download
            print(f'[Molmo H3] Downloading {len(missing)} missing package files from {MODEL_REPO} into {root}', flush=True)
            snapshot_download(repo_id=MODEL_REPO, local_dir=str(root), allow_patterns=missing)
        missing=missing_files()
        if missing:
            raise RuntimeError(f'Molmo package download is incomplete or contains Git-LFS pointer files: {missing}')
    return str(root)


def resolve_projector(model):
    """Load the separate projector bundled with the public Molmo H3 package."""
    bundled = Path(model) / 'projections' / PROJECTOR_NAME
    if bundled.is_file():
        return str(bundled)
    raise ValueError(f'Molmo2 package is missing its separate projector: {PROJECTOR_NAME}')
