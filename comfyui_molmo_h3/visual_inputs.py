"""Visual prefix followed by exact Phase A token ids, for the Phase B pilot."""
import numpy as np
import torch


def prepare_inputs(processor, ids, visuals, max_video_frames=8):
    images, prefix = [], []
    picture_count = video_count = source_count = 0
    for item in visuals:
        data = np.load(item['path'], allow_pickle=False)
        if item['type'] in ('image', 'source_person'):
            images.append(data[0] if data.ndim == 4 else data)
            if item['type'] == 'source_person':
                source_count += 1
                prefix.append(f"<SourcePerson {source_count}> (original person to replace): <|image|>\n")
            else:
                picture_count += 1
                prefix.append(f"<Picture {picture_count}>: <|image|>\n")
        elif item['type'] == 'video':
            video_count += 1
            timestamps = item['timestamps']
            if len(timestamps) != len(data):
                raise ValueError('Video timestamps must match the sampled frames')
            indices = np.linspace(0, len(data) - 1, min(len(data), max_video_frames), dtype=int)
            # This checkpoint rejects image+video tensors in the same forward.
            # Timestamped multi-image frames let references and guides share its supported image path.
            for i in indices:
                images.append(data[i])
                prefix.append(f"<Video {video_count}> frame at {timestamps[i]:.3f}s: <|image|>\n")
    kwargs = {}
    if images:
        kwargs['images'] = images
    batch = processor(text=''.join(prefix), return_tensors='pt', **kwargs)
    suffix = torch.tensor([ids], dtype=torch.long)
    offset = batch['input_ids'].shape[1]
    batch['input_ids'] = torch.cat([batch['input_ids'], suffix], dim=1)
    batch['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones_like(suffix)], dim=1)
    if 'token_type_ids' in batch:
        batch['token_type_ids'] = torch.cat([batch['token_type_ids'], torch.zeros_like(suffix)], dim=1)
    # Visual-prefix BPE boundaries cannot retokenize the calibrated text suffix.
    assert batch['input_ids'][0, offset:].tolist() == ids
    return batch, offset
