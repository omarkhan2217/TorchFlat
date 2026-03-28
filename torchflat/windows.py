"""Multi-scale sliding window extraction."""

from __future__ import annotations

import torch

DEFAULT_WINDOW_SCALES: list[tuple[int, int]] = [
    (256, 128),
    (512, 256),
    (2048, 512),
    (8192, 2048),
]


def extract_windows(
    flux: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_id: torch.Tensor,
    time: torch.Tensor,
    window_scales: list[tuple[int, int]] = DEFAULT_WINDOW_SCALES,
) -> dict[int, dict[str, torch.Tensor]]:
    """Multi-scale sliding window extraction with segment-aware validity.

    Processes one scale at a time to bound peak VRAM.  Each scale's GPU
    tensors are freed before the next scale begins.

    Args:
        flux: ``[B, L]`` normalized flux.
        valid_mask: ``[B, L]`` boolean mask.
        segment_id: ``[B, L]`` segment labels.
        time: ``[B, L]`` timestamps.
        window_scales: ``[(window_size, stride), ...]``.

    Returns:
        Dict keyed by *window_size*.  Each value is a dict with:

        - ``"windows"``: ``[N_total, window_size]`` CPU float tensor.
        - ``"window_times"``: ``[N_total, 2]`` CPU float tensor (start, end BJD).
        - ``"star_indices"``: ``[N_total]`` CPU int tensor.
    """
    _B, L = flux.shape
    results: dict[int, dict[str, torch.Tensor]] = {}

    for win_size, stride in window_scales:
        if L < win_size:
            # All stars too short for this scale
            results[win_size] = {
                "windows": torch.empty(0, win_size, dtype=flux.dtype),
                "window_times": torch.empty(0, 2, dtype=time.dtype),
                "star_indices": torch.empty(0, dtype=torch.long),
            }
            continue

        # Unfold into sliding windows  [B, N_win, win_size]
        windows_gpu = flux.unfold(dimension=1, size=win_size, step=stride)
        seg_windows = segment_id.unfold(dimension=1, size=win_size, step=stride)
        mask_windows = valid_mask.unfold(dimension=1, size=win_size, step=stride)
        time_windows = time.unfold(dimension=1, size=win_size, step=stride)

        # A window is valid iff every point shares the same segment AND is valid
        window_valid = (seg_windows == seg_windows[:, :, 0:1]).all(dim=-1)
        window_valid &= mask_windows.all(dim=-1)  # [B, N_win]

        # Gather valid windows
        valid_indices = window_valid.nonzero(as_tuple=False)  # [N_valid, 2]

        if valid_indices.numel() == 0:
            windows_cpu = torch.empty(0, win_size, dtype=flux.dtype)
            times_cpu = torch.empty(0, 2, dtype=time.dtype)
            star_idx_cpu = torch.empty(0, dtype=torch.long)
        else:
            windows_cpu = windows_gpu[window_valid].cpu()
            start_times = time_windows[:, :, 0][window_valid].cpu()
            end_times = time_windows[:, :, -1][window_valid].cpu()
            times_cpu = torch.stack([start_times, end_times], dim=-1)
            star_idx_cpu = valid_indices[:, 0].cpu()

        # Free GPU memory before next scale
        del windows_gpu, seg_windows, mask_windows, time_windows

        results[win_size] = {
            "windows": windows_cpu,
            "window_times": times_cpu,
            "star_indices": star_idx_cpu,
        }

    return results
