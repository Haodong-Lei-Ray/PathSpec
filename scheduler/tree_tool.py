import torch

def layerwise_mean_simple(prob_list: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
    """
    假设 layer_ids 从1开始连续递增、无跳层
    """
    if prob_list.numel() == 0:
        return torch.empty(0, device=prob_list.device)

    max_layer = layer_ids.max().item()
    if max_layer < 1:
        return torch.empty(0, device=prob_list.device)

    means = torch.full((max_layer,), float('nan'), device=prob_list.device, dtype=torch.float)

    for lv in range(1, max_layer + 1):
        mask = (layer_ids == lv) & (prob_list != -1)
        if mask.any():
            means[lv - 1] = prob_list[mask].mean()

    # 去尾部连续 nan
    valid = ~torch.isnan(means)
    if not valid.any():
        return torch.empty(0, device=prob_list.device)

    last = valid.nonzero(as_tuple=True)[0].max().item()
    return means[:last + 1]