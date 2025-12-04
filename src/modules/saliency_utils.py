import torch
import torch.nn.functional as F
import numpy as np


def compute_saliency_vanilla(model, input_tensor, target_class=None):
    """Compute vanilla saliency maps (absolute gradient of output wrt input).

    Args:
        model: nn.Module in eval() or train() mode (we'll use eval by default).
        input_tensor: torch.Tensor shape [B, C, H, W] requires_grad=False
        target_class: Optional tensor/list of length B with target class indices.

    Returns:
        saliency: numpy array [B, H, W] with normalized saliency maps (0..1)
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    inp = input_tensor.to(device).detach().requires_grad_(True)

    outputs = model(inp)  # [B, C]
    if target_class is None:
        # pick predicted class per sample
        preds = outputs.argmax(dim=1)
    else:
        preds = torch.as_tensor(target_class, device=device)

    # pick outputs for the target classes
    selected = outputs.gather(1, preds.unsqueeze(1)).squeeze(1)
    # sum over batch to backprop all at once
    loss = selected.sum()
    model.zero_grad()
    loss.backward()

    grads = inp.grad.detach()  # [B, C, H, W]
    saliency = grads.abs().max(dim=1)[0]  # [B, H, W]
    saliency = _normalize_np(saliency.cpu().numpy())

    if was_training:
        model.train()
    return saliency


def compute_gradcam(model, input_tensor, target_layer, target_class=None):
    """Compute Grad-CAM maps for the given `target_layer` (nn.Module).

    Args:
        model: nn.Module
        input_tensor: torch.Tensor [B, C, H, W]
        target_layer: Module inside model (e.g., model.features[-1])
        target_class: Optional target class indices

    Returns:
        cam_maps: numpy array [B, H, W] normalized to [0,1]
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    # ensure the input allows gradient computations for backward
    inp = input_tensor.to(device).detach().requires_grad_(True)

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    outputs = model(inp)
    if target_class is None:
        preds = outputs.argmax(dim=1)
    else:
        preds = torch.as_tensor(target_class, device=device)

    selected = outputs.gather(1, preds.unsqueeze(1)).squeeze(1)
    loss = selected.sum()
    model.zero_grad()
    loss.backward()

    fh.remove(); bh.remove()

    # ensure hooks captured tensors
    if activations is None:
        fh.remove(); bh.remove()
        raise RuntimeError("Forward hook did not capture activations. Is target_layer valid?")
    if gradients is None:
        fh.remove(); bh.remove()
        raise RuntimeError("Backward hook did not capture gradients. Is the model's backward called correctly?")

    # activations: [B, K, h, w], gradients: [B, K, h, w]
    # compute channel weights by global average pooling of gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
    weighted = (weights * activations).sum(dim=1)  # [B, h, w]
    cam = F.relu(weighted)
    cam = _normalize_np(cam.cpu().numpy())

    if was_training:
        model.train()
    return cam


def _normalize_np(arr):
    # normalize each sample to [0,1]
    arr = arr.copy()
    B = arr.shape[0]
    out = np.zeros_like(arr)
    for i in range(B):
        a = arr[i]
        a_min = a.min()
        a_max = a.max()
        if a_max > a_min:
            out[i] = (a - a_min) / (a_max - a_min)
        else:
            out[i] = 0.0
    return out


def overlay_on_image(original_pil, cam_np, alpha=0.5, colormap="jet"):
    """Overlay a single-channel cam (H, W) onto a PIL image and return PIL image.

    Args:
        original_pil: PIL.Image RGB
        cam_np: numpy array [H, W] values in [0,1]
        alpha: float overlay strength
        colormap: str (not used for now; placeholder for future)
    """
    from PIL import Image, ImageOps
    import matplotlib.pyplot as plt

    # convert cam to uint8 heatmap via matplotlib
    cmap = plt.get_cmap(colormap)
    heatmap = (cmap(cam_np)[:, :, :3] * 255).astype('uint8')
    heatmap_pil = Image.fromarray(heatmap).convert('RGB')

    original = original_pil.convert('RGB')
    heatmap_resized = heatmap_pil.resize(original.size, resample=Image.BILINEAR)

    return Image.blend(original, heatmap_resized, alpha=alpha)


def save_saliency_overlay(output_path, original_pil, cam_np, alpha=0.5):
    """Create overlay and save as PNG to output_path."""
    over = overlay_on_image(original_pil, cam_np, alpha=alpha)
    over.save(output_path, format='PNG')

