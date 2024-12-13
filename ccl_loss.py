import torch

def _off_diagonal(x):
    n, m = x.shape[-2:]
    assert n == m
    off_diag_indices = torch.ones(n, n, dtype=bool).fill_diagonal_(0)
    return x[..., off_diag_indices].view(x.size(0), -1)


def correlation_loss(weak_out_logits, strong_out_logits):
    # Reshape and permute
    weak_out_logits = weak_out_logits.reshape(weak_out_logits.size(0), weak_out_logits.size(1), -1).permute(0, 2, 1)
    strong_out_logits = strong_out_logits.reshape(strong_out_logits.size(0), strong_out_logits.size(1), -1).permute(0, 2, 1)

    # Standardize batch-wise
    weak_mean = weak_out_logits.mean(dim=1, keepdim=True)
    weak_std = weak_out_logits.std(dim=1, keepdim=True)
    weak_out_logits = (weak_out_logits - weak_mean) / weak_std

    strong_mean = strong_out_logits.mean(dim=1, keepdim=True)
    strong_std = strong_out_logits.std(dim=1, keepdim=True)
    strong_out_logits = (strong_out_logits - strong_mean) / strong_std

    # Compute similarity matrix c for all samples
    c = torch.matmul(weak_out_logits.transpose(1, 2), strong_out_logits)
    c.div_(weak_out_logits.shape[1])  # Divide by the number of features (256*256)

    # Compute loss
    on_diag = torch.diagonal(c, dim1=-2, dim2=-1).add_(-1).pow_(2).sum(dim=-1)
    off_diag = _off_diagonal(c).add_(1).pow_(2).sum(dim=-1)

    loss = on_diag.mean() + 0.0051 * off_diag.mean()
    return loss





