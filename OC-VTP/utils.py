"""
This file contains useful functions and classes, including customized loss function, metrics and other tools.
"""
import random
import colorsys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AreaWeightedMSELoss(nn.Module):
    def __init__(self,
                 power_p: float = 0.5,
                 cap_w: float = 10.0,
                 norm_mode: str = "mean1",
                 reduce_channel: str = "mean",
                 reduction: str = "mean",
                 eps: float = 1e-8, ):
        super(AreaWeightedMSELoss, self).__init__()
        """
        Alpha weighted mse, better for objects with small areas.
        - Get slot mask areas inverse w_k, and use alpha_k and w_k to merge pixels.
        - Here, area_k is computed from slot-attention one-hot assignments (hard masks).
        - Ultimately, get alpha_final to do weighted mse.

        :param
        power_p: w_k ∝ (area_k)^(-p), p∈[0,1]
        cap_w: Upperbound of weighted of inverse areas.
        norm_mode: "mean1" | "sum1" | "none"
        reduce_channel: "mean" or "sum" along channel
        reduction: "mean"
        eps: 1e-8
        :return
        Weighted mse loss
        """
        assert norm_mode in {"mean1", "sum1", "none"}, f"norm_mode must be one of {'mean1', 'sum1', 'none'}"
        assert reduce_channel in {"mean", "sum"}, f"reduce_channel must be one of {'mean', 'sum'}"
        assert reduction in {"mean", "sum", "none"}, f"reduction must be one of {'mean', 'sum', 'none'}"
        self.power_p = power_p
        self.cap_w = cap_w
        self.norm_mode = norm_mode
        self.reduce_channel = reduce_channel
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        pred:   [B, T, C]
        target: [B, T, C]
        alpha:  [B, S, T]  slot attention map (soft masks)
        """
        assert pred.shape == target.shape, f"pred {pred.shape} does not match target {pred.shape}"
        B, T, C = pred.shape  # [Batch, tokens, channels]
        assert alpha.shape[0] == B and alpha.shape[2] == T, f"alpha {alpha.shape} does not match pred/target {pred.shape}"
        S = alpha.shape[1]

        # Detachment (no grad through alpha)
        alpha = alpha.detach()  # [B, S, T]

        # alpha: [B, S, T] -> [B, T]
        slot_idx = alpha.argmax(dim=1)  # [B, T]

        # one-hot： [B, T] -> [B, T, S]
        one_hot_mask = F.one_hot(slot_idx, num_classes=S).to(alpha.dtype)  # [B, T, S]

        # [B, S, T]
        one_hot_mask = one_hot_mask.permute(0, 2, 1).contiguous()  # [B, S, T]

        area_k = one_hot_mask.sum(dim=2)  # [B, S]

        # Inverse area_ratio - ((Tokens) / area_k) ** power_p
        area_ratio = area_k / float(T)  # [B, S]
        w = (area_ratio.clamp_min(self.eps)).pow(-self.power_p)  # [B, S]

        # Normalization
        if self.norm_mode == "mean1":
            w = w / (w.mean(dim=1, keepdim=True).clamp_min(self.eps))
        elif self.norm_mode == "sum1":
            w = S * w / (w.sum(dim=1, keepdim=True).clamp_min(self.eps))
        elif self.norm_mode == "none":
            pass
        else:
            raise ValueError("norm_mode must be in {'mean1','sum1','none'}")

        # Upperbound
        if self.cap_w is not None:
            w = w.clamp(max=self.cap_w)

        # Apply weights to alpha, got alpha_w - increase the intensity of small object area slots
        w_img = w.view(B, S, 1)  # [B, S, 1]
        alpha_w = (alpha * w_img).sum(dim=1)  # [B, T]

        alpha_final = alpha_w
        alpha_final = alpha_final.clamp_min(0.0)

        # MSE calculation
        if self.reduce_channel == "mean":
            sq_err_pix = (pred - target).pow(2).mean(dim=2)  # [B, T]
        elif self.reduce_channel == "sum":
            sq_err_pix = (pred - target).pow(2).sum(dim=2)  # [B, T]
        else:
            raise ValueError("reduce_channel must be 'mean' or 'sum'.")

        # Normalization
        denom = alpha_final.sum(dim=1, keepdim=True).clamp_min(self.eps)  # [B, 1]
        loss_per_img = (alpha_final * sq_err_pix).sum(dim=1, keepdim=True) / denom  # [B, 1]
        loss_per_img = loss_per_img.view(B)

        if self.reduction == "none":
            return loss_per_img
        elif self.reduction == "sum":
            return loss_per_img.sum()
        else:
            return loss_per_img.mean()


def interpolat_argmax_attent(attent, size, mode="bilinear", grid_hw=None):
    """
    attent: shape=(B, S, T)
    segment: shape=(B, H_out, W_out), dtype=uint8; index segment
    """
    B, S, T = attent.shape
    if grid_hw is None:
        H = W = int(T ** 0.5)
        assert H * W == T, "T is not a perfect square; pass grid_hw=(H, W)."
    else:
        H, W = grid_hw
        assert H * W == T, "grid_hw does not match T."

    attent_ = attent.view(B, S, H, W)  # (B, S, H, W)
    attent_ = F.interpolate(attent_, size=size, mode=mode)
    segment = attent_.argmax(1).byte()  # (B, H_out, W_out)
    return segment


def generate_spectrum_colors(num_color):
    spectrum = []
    for i in range(num_color):
        hue = i / float(num_color)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        spectrum.append([int(255 * c) for c in rgb])
    return np.array(spectrum, dtype="uint8")  # (s,c=3)


def draw_segmentation_np(image: np.ndarray, segment: np.ndarray, alpha=0.5, color=None):
    h, w, c = image.shape
    h2, w2, s = segment.shape
    assert h == h2 and w == w2 and c == 3, "image/segment 尺寸或通道不匹配"

    if image.dtype != np.uint8:
        img = image.astype(np.float32)
        if img.max() <= 1.0:          # 0-1 图
            img = img * 255.0
        image_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        image_u8 = image

    masks_bool = segment.astype(bool)

    if color is None:
        color = generate_spectrum_colors(s)

    image2 = torchvision.utils.draw_segmentation_masks(
        image=torch.from_numpy(image_u8).permute(2, 0, 1).contiguous(),      # (3,H,W) uint8
        masks=torch.from_numpy(masks_bool).permute(2, 0, 1).contiguous(),    # (S,H,W) bool
        alpha=alpha,
        colors=color.tolist(),
    )
    return image2.permute(1, 2, 0).cpu().numpy()
