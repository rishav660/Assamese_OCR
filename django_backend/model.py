"""
Assamese OCR Model — ResNet18 Backbone + Transformer Encoder + CTC
================================================================
Key design decisions:
  - img_height=64        : taller input preserves matra/diacritic structure
  - Layer4 stride=(1,1)  : stops height collapsing to 1 too early
  - adaptive_avg_pool2d  : collapses residual height cleanly before sequence modeling
  - d_model=512, nhead=8 : 64 dims/head, richer attention for complex conjuncts
  - num_encoder_layers=4 : deeper transformer for script complexity
  - Pretrained weights   : grayscale-adapted from ImageNet ResNet18
  - Explicit conv2 stride: prevents accidental width downsampling
  - nn_classes           : expects vocab_size + 1 (index 0 = CTC blank)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, ResNet18_Weights


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Input/output shape: (T, B, E)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)                          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                                        # (d_model/2,)

        pe = torch.zeros(max_len, 1, d_model)                                  # (max_len, 1, E)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, B, E)"""
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# ResNet18 helper
# ---------------------------------------------------------------------------

def _modify_resnet_layer(layer: nn.Sequential, stride: tuple) -> nn.Sequential:
    """
    Modify the first BasicBlock of a ResNet layer:
      - Set conv1 stride to `stride`
      - Force conv2 stride to (1, 1) to never accidentally reduce width
      - Adjust the downsample shortcut stride to match
    Works correctly for ResNet18/34 BasicBlock (no bottleneck).
    """
    block = layer[0]
    block.conv1.stride = stride
    block.conv2.stride = (1, 1)          # explicit: conv2 must never touch width
    if block.downsample is not None:
        block.downsample[0].stride = stride
    return layer


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AssameseOCR(nn.Module):
    """
    Architecture:
        Input  : (B, 1, H, W)   — grayscale, recommended H=64
        CNN    : Modified ResNet18
                   layer1: stride (2,2) → H/4,  W/4
                   layer2: stride (2,1) → H/8,  W/4   (width preserved)
                   layer3: stride (2,1) → H/16, W/4   (width preserved)
                   layer4: stride (1,1) → H/16, W/4   (height preserved too)
                 Followed by adaptive_avg_pool2d → (B, 512, 1, W/4)
        Proj   : Linear 512 → d_model
        Pos Enc: Sinusoidal
        Transf : TransformerEncoder (num_encoder_layers × TransformerEncoderLayer)
        Head   : Linear d_model → nn_classes
        Output : (T, B, nn_classes)  — feed directly to nn.CTCLoss

    Args:
        img_height        (int)   : Input image height. 64 recommended for Assamese.
        nn_classes        (int)   : len(char_vocab) + 1  (+1 for CTC blank at index 0).
        d_model           (int)   : Transformer hidden dimension.
        nhead             (int)   : Number of attention heads. d_model // nhead ≥ 32.
        num_encoder_layers(int)   : Depth of transformer encoder.
        dim_feedforward   (int)   : FFN inner dimension in each transformer layer.
        dropout           (float) : Dropout rate throughout.
        pretrained        (bool)  : Load ImageNet weights and adapt to grayscale.
    """

    def __init__(
        self,
        img_height: int = 64,
        nn_classes: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
            f"Currently d_model/nhead = {d_model/nhead:.1f}"
        )
        assert img_height >= 32, "img_height must be at least 32."

        # ------------------------------------------------------------------ #
        # 1. ResNet18 Backbone                                                 #
        # ------------------------------------------------------------------ #
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        # --- First conv: RGB → Grayscale ---
        # Transfer pretrained weights by averaging across the 3 RGB channels.
        # This is much better than random init and converges faster.
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            rgb_weight = resnet.conv1.weight.data          # (64, 3, 7, 7)
            self.conv1.weight.data = rgb_weight.mean(dim=1, keepdim=True)

        self.bn1    = resnet.bn1
        self.relu   = resnet.relu
        # kernel=2, stride=2: reduces H and W by 2× each → H/4, W/4 after conv1
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # layer1: stride (2,2) built-in → output (B, 64,  H/4,  W/4)
        self.layer1 = resnet.layer1

        # layer2: (2,1) → output (B, 128, H/8,  W/4)   width preserved
        self.layer2 = _modify_resnet_layer(resnet.layer2, stride=(2, 1))

        # layer3: (2,1) → output (B, 256, H/16, W/4)   width preserved
        self.layer3 = _modify_resnet_layer(resnet.layer3, stride=(2, 1))

        # layer4: (1,1) → output (B, 512, H/16, W/4)
        # We stop downsampling height here so matras/diacritics survive.
        # adaptive_avg_pool2d will cleanly collapse height afterward.
        self.layer4 = _modify_resnet_layer(resnet.layer4, stride=(1, 1))

        # ------------------------------------------------------------------ #
        # 2. Sequence Projection                                               #
        # ------------------------------------------------------------------ #
        # After pool → (B, 512, 1, T); squeeze → (B, 512, T); permute → (T, B, 512)
        self.feature_proj = nn.Linear(512, d_model)
        self.feature_norm = nn.LayerNorm(d_model)       # stabilises early training

        # ------------------------------------------------------------------ #
        # 3. Transformer Encoder                                               #
        # ------------------------------------------------------------------ #
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,          # we use (T, B, E) convention throughout
            norm_first=True,            # Pre-LN: more stable gradient flow
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False, # avoids shape warnings with variable-length
        )

        # ------------------------------------------------------------------ #
        # 4. Classifier Head                                                   #
        # ------------------------------------------------------------------ #
        self.classifier = nn.Linear(d_model, nn_classes)

        # ------------------------------------------------------------------ #
        # 5. Weight Init (non-pretrained parts)                                #
        # ------------------------------------------------------------------ #
        self._init_weights()

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def _init_weights(self):
        """Xavier/Kaiming init for projection and classifier layers."""
        for module in [self.feature_proj, self.classifier]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W)  — grayscale image batch

        Returns:
            logits: (T, B, nn_classes)
                    Pass to F.log_softmax then nn.CTCLoss.
                    CTC blank index = 0 (default).
        """
        # ---- CNN ----
        x = self.conv1(x)       # (B, 64,  H/2,  W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # (B, 64,  H/4,  W/4)

        x = self.layer1(x)      # (B, 64,  H/4,  W/4)
        x = self.layer2(x)      # (B, 128, H/8,  W/4)
        x = self.layer3(x)      # (B, 256, H/16, W/4)
        x = self.layer4(x)      # (B, 512, H/16, W/4)

        # Collapse height dimension cleanly regardless of exact img_height
        b, c, h, w = x.size()
        x = F.adaptive_avg_pool2d(x, (1, w))   # (B, 512, 1, W/4)
        x = x.squeeze(2)                        # (B, 512, W/4)
        x = x.permute(2, 0, 1)                 # (T, B, 512)   T = W/4

        # ---- Projection ----
        x = self.feature_proj(x)               # (T, B, d_model)
        x = self.feature_norm(x)

        # ---- Transformer ----
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)        # (T, B, d_model)

        # ---- Head ----
        x = self.classifier(x)                 # (T, B, nn_classes)
        return x


# ---------------------------------------------------------------------------
# Convenience: CTC decode (greedy)
# ---------------------------------------------------------------------------

def ctc_greedy_decode(logits: torch.Tensor, blank_idx: int = 0) -> list[list[int]]:
    """
    Greedy CTC decoder.

    Args:
        logits   : (T, B, C) raw logits from AssameseOCR.forward()
        blank_idx: index of the CTC blank token (default 0)

    Returns:
        List of decoded label sequences (one per batch item).
    """
    probs      = F.softmax(logits, dim=-1)          # (T, B, C)
    best_paths = probs.argmax(dim=-1)               # (T, B)
    best_paths = best_paths.permute(1, 0)           # (B, T)

    decoded = []
    for path in best_paths:
        path    = path.tolist()
        # Collapse consecutive duplicates, then remove blanks
        chars   = [c for i, c in enumerate(path)
                   if c != blank_idx and (i == 0 or c != path[i - 1])]
        decoded.append(chars)
    return decoded


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VOCAB_SIZE = 200        # example Assamese vocab (adjust to yours)
    BLANK_IDX  = 0
    B, H, W    = 4, 64, 320

    model = AssameseOCR(
        img_height         = H,
        nn_classes         = VOCAB_SIZE + 1,   # +1 for blank
        d_model            = 512,
        nhead              = 8,
        num_encoder_layers = 4,
        dim_feedforward    = 2048,
        dropout            = 0.1,
        pretrained         = False,            # set True when training
    )
    model.eval()

    dummy = torch.randn(B, 1, H, W)
    with torch.no_grad():
        out = model(dummy)                     # (T, B, nn_classes)

    T = out.size(0)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}   →  T={T} time-steps")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    # CTC loss example
    log_probs   = F.log_softmax(out, dim=-1)           # required by CTCLoss
    input_lens  = torch.full((B,), T, dtype=torch.long)
    target_lens = torch.randint(1, T // 2, (B,))
    targets     = torch.randint(1, VOCAB_SIZE + 1,     # avoid blank=0
                                (target_lens.sum().item(),))

    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens)
    print(f"CTC loss (random): {loss.item():.4f}")

    # Greedy decode
    decoded = ctc_greedy_decode(out, blank_idx=BLANK_IDX)
    print(f"Greedy decoded lengths: {[len(d) for d in decoded]}")