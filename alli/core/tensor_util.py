import torch
from torch._six import inf

def unsqueeze_and_expand(
    t: torch.Tensor,
    dim: int,
    num_repeat: int,
):
    return t.unsqueeze(dim).expand(
        *(-1 for _ in range(dim)),
        num_repeat,
        *(-1 for _ in range(t.dim() - dim))
    )

# See clip_grad_norm_().
def get_grad_norm(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    assert len(parameters) > 0
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

