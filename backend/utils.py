from typing import Optional
import torch

def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = int(lens.max().item())
        # print(max_len)
        # max_len = 400
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask
