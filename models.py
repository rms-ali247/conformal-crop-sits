"""
models.py — Model architectures for crop type classification.

Single source of truth — imported by both pipeline/train.py (training)
and inference/predict.py (inference).  No code duplication.

Architectures:
  - MSTACNN         : Multi-Scale Temporal Attention CNN (primary)
  - Conv1DClassifier: Simple 1D-CNN baseline
  - LSTMClassifier  : LSTM baseline
  - S4DClassifier   : Structured State Space (S4D) for long-range temporal modelling
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Gradient reversal (for region-invariant / domain-adversarial training)
# ------------------------------------------------------------------ #

class _GradReverse(torch.autograd.Function):
    """Identity on the forward pass; negates and scales the gradient on the
    backward pass. Used to train features that are predictive of the crop but
    NOT of the district, i.e. region-invariant representations (Ganin et al.,
    domain-adversarial training)."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)


# ------------------------------------------------------------------ #
#  Building blocks
# ------------------------------------------------------------------ #

class MultiScaleConvBlock(nn.Module):
    """Parallel 1D convolutions at multiple temporal scales.

    Inspired by STPCNet temporal perceptive clues — captures short-range
    (bimonthly transitions), medium-range (growth phases), and long-range
    (seasonal arc) patterns simultaneously.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        ch_short = out_channels // 3
        ch_mid   = out_channels // 3
        ch_long  = out_channels - ch_short - ch_mid

        self.conv_short = nn.Conv1d(in_channels, ch_short, kernel_size=2, padding=1)
        self.conv_mid   = nn.Conv1d(in_channels, ch_mid,   kernel_size=3, padding=1)
        self.conv_long  = nn.Conv1d(in_channels, ch_long,  kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        s = self.conv_short(x)[..., :T]   # k=2, p=1 gives T+1 → trim
        m = self.conv_mid(x)               # k=3, p=1 gives T
        l = self.conv_long(x)              # k=5, p=2 gives T
        return self.bn(torch.cat([s, m, l], dim=1))


class TemporalAttention(nn.Module):
    """Learned temporal attention that weights each timestep by its
    discriminative importance.

    Returns both the attended feature vector and attention weights
    (for interpretability / visualization).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.query(x)                        # (B, T, 1)
        weights = torch.softmax(scores, dim=1)        # (B, T, 1)
        context = (x * weights).sum(dim=1)            # (B, C)
        return context, weights.squeeze(-1)            # (B, C), (B, T)


# ------------------------------------------------------------------ #
#  Full architectures
# ------------------------------------------------------------------ #

class MSTACNN(nn.Module):
    """Multi-Scale Temporal Attention CNN.

    Architecture:
        Input (B, T=7, C=28)
          → permute to (B, 28, 7)
          → MultiScaleConvBlock(28 → 64) + ReLU
          → MultiScaleConvBlock(64 → 128) + ReLU + residual
          → permute to (B, 7, 128)
          → TemporalAttention(128) → (B, 128) + attention weights (B, 7)
          → Dropout → Linear → (B, num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.ms_block1 = MultiScaleConvBlock(in_channels, 64)
        self.ms_block2 = MultiScaleConvBlock(64, 128)
        self.residual  = nn.Conv1d(64, 128, kernel_size=1)

        self.temporal_attention = TemporalAttention(128)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._attn_weights = None

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x = x.permute(0, 2, 1)                        # (B, C, T)

        h1 = F.relu(self.ms_block1(x))                # (B, 64, T)
        h2 = self.ms_block2(h1)                        # (B, 128, T)
        h2 = F.relu(h2 + self.residual(h1))            # residual connection

        h2 = h2.permute(0, 2, 1)                       # (B, T, 128)
        context, attn_w = self.temporal_attention(h2)   # (B, 128), (B, T)
        self._attn_weights = attn_w.detach()

        logits = self.classifier(context)
        if return_attention:
            return logits, attn_w
        return logits


class Conv1DClassifier(nn.Module):
    """Simple 1D-CNN baseline (v1 architecture, kept for comparison)."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    """LSTM baseline."""

    def __init__(self, in_channels: int, num_classes: int,
                 hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ------------------------------------------------------------------ #
#  S4D — Structured State Space (Diagonal) Block
# ------------------------------------------------------------------ #

class S4DKernel(nn.Module):
    """Diagonal SSM kernel with HiPPO-LegS initialisation.

    Implements the diagonal approximation from:
        "On the Parameterization and Initialization of Diagonal State Space
         Models" (Gu et al., NeurIPS 2022).

    Given diagonal state matrix A ∈ C^N, input-to-state B ∈ C^N,
    state-to-output C ∈ C^N, the SSM kernel of length L is:

        K[k] = Re( C · (exp(A·Δ))^k · (exp(A·Δ) - 1) / A · B )

    where Δ is a learnable discretisation step.
    This kernel is convolved with the input for efficient parallel computation.
    """

    def __init__(self, d_model: int, N: int = 64, dt_min: float = 0.001,
                 dt_max: float = 0.1, lr: float | None = None):
        super().__init__()
        self.N = N

        # HiPPO-LegS initialisation for diagonal A
        # A_n = -1/2 + ni  (real part = -1/2 ensures stability)
        A_real = torch.full((d_model, N // 2), -0.5)
        A_imag = math.pi * torch.arange(N // 2).float().unsqueeze(0).expand(d_model, -1)
        self.register_buffer("A_real_init", A_real)

        # Learnable parameters (real-valued parameterisation of complex numbers)
        self.log_A_real = nn.Parameter(torch.log(-A_real))  # log(-real) so real = -exp(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        self.C = nn.Parameter(torch.randn(d_model, N // 2, 2))  # complex as (real, imag)
        nn.init.normal_(self.C, mean=0.0, std=0.5 * N ** -0.5)

        # Discretisation step Δ (log-uniform init between dt_min and dt_max)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # B is fixed to 1 (absorbed into C during learning)
        B = torch.ones(d_model, N // 2)
        self.register_buffer("B", B)

    def forward(self, L: int) -> torch.Tensor:
        """Compute the SSM convolution kernel of length L.

        Returns: (d_model, L) real-valued kernel.
        """
        dt = self.log_dt.exp()                         # (d_model,)
        A_real = -self.log_A_real.exp()                # (d_model, N//2), negative
        A_imag = self.A_imag                           # (d_model, N//2)

        # Discretise: Ā = exp(A · Δ)
        dtA_real = A_real * dt.unsqueeze(-1)           # (d_model, N//2)
        dtA_imag = A_imag * dt.unsqueeze(-1)           # (d_model, N//2)

        # exp(a + bi) = exp(a)(cos(b) + i·sin(b))
        exp_real = dtA_real.exp()                      # magnitude decay
        C_real = self.C[..., 0]                        # (d_model, N//2)
        C_imag = self.C[..., 1]

        # Vandermonde-style computation: K[k] = Σ_n C_n · Ā_n^k · B_n
        # We build the full kernel via cumulative powers
        arange = torch.arange(L, device=dt.device).float()  # (L,)

        # Ā^k = exp(k · dtA_real) · (cos(k · dtA_imag) + i·sin(k · dtA_imag))
        # Shape: (d_model, N//2, L)
        pow_real = (dtA_real.unsqueeze(-1) * arange)   # (d_model, N//2, L)
        pow_imag = (dtA_imag.unsqueeze(-1) * arange)

        vand_real = pow_real.exp() * pow_imag.cos()    # (d_model, N//2, L)
        vand_imag = pow_real.exp() * pow_imag.sin()

        # C · Ā^k · B  (B=1 so just C · Ā^k)
        # Complex multiplication: (Cr + i·Ci)(Vr + i·Vi) = (Cr·Vr - Ci·Vi) + i(Cr·Vi + Ci·Vr)
        CB_real = C_real.unsqueeze(-1) * vand_real - C_imag.unsqueeze(-1) * vand_imag
        # We only need the real part of the output kernel
        K = CB_real.sum(dim=1) * 2  # ×2 to account for conjugate pairs (N//2 → N)

        return K  # (d_model, L)


class S4DBlock(nn.Module):
    """One S4D layer: SSM kernel convolution + gated MLP + residual + norm.

    Input:  (B, L, d_model)
    Output: (B, L, d_model)
    """

    def __init__(self, d_model: int, ssm_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.kernel = S4DKernel(d_model, N=ssm_dim)
        self.D = nn.Parameter(torch.randn(d_model))  # skip connection in SSM

        # Output projection with gated linear unit (GLU)
        self.out_proj = nn.Linear(d_model, d_model * 2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        residual = x
        x_t = x.transpose(1, 2)                        # (B, d_model, L)
        L = x_t.shape[-1]

        K = self.kernel(L)                              # (d_model, L)

        # FFT-based convolution: y = x * K
        x_fft = torch.fft.rfft(x_t, n=2 * L, dim=-1)  # zero-padded
        K_fft = torch.fft.rfft(K, n=2 * L, dim=-1)     # (d_model, L+1)
        y_fft = x_fft * K_fft.unsqueeze(0)             # (B, d_model, L+1)
        y = torch.fft.irfft(y_fft, n=2 * L, dim=-1)[..., :L]  # (B, d_model, L)

        # Skip connection (D parameter)
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_t

        y = y.transpose(1, 2)                           # (B, L, d_model)

        # GLU activation
        gate = self.out_proj(y)                         # (B, L, 2*d_model)
        y = F.glu(gate, dim=-1)                         # (B, L, d_model)

        y = self.dropout(y) + residual
        y = self.norm(y)
        return y


class S4DClassifier(nn.Module):
    """Structured State Space (S4D) classifier for temporal sequences.

    Architecture:
        Input (B, T, C=28)
          → Linear(C → d_model)
          → S4DBlock × n_layers (with residual + LayerNorm)
          → TemporalAttention(d_model) → (B, d_model) + attention weights
          → Dropout → Linear → (B, num_classes)

    The S4D blocks learn long-range temporal dependencies via structured
    state space convolutions, while temporal attention provides an
    interpretable pooling mechanism over the sequence.
    """

    def __init__(self, in_channels: int, num_classes: int,
                 d_model: int = 128, ssm_dim: int = 64,
                 n_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Linear(in_channels, d_model)

        self.s4_layers = nn.ModuleList([
            S4DBlock(d_model, ssm_dim=ssm_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.temporal_attention = TemporalAttention(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._attn_weights = None

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """x: (B, T, C)"""
        h = self.encoder(x)                            # (B, T, d_model)

        for layer in self.s4_layers:
            h = layer(h)                               # (B, T, d_model)

        context, attn_w = self.temporal_attention(h)   # (B, d_model), (B, T)
        self._attn_weights = attn_w.detach()

        logits = self.classifier(context)
        if return_attention:
            return logits, attn_w
        return logits


# ------------------------------------------------------------------ #
#  Spatial-Spectral SSM (pixel-level classification)
# ------------------------------------------------------------------ #

class SpatialEncoder(nn.Module):
    """Shared 2D-CNN encoder that extracts a spatial feature vector
    from each (P, P, C) patch independently.

    Input:  (B*T, C, P, P)
    Output: (B*T, out_channels)
    """

    def __init__(self, in_channels: int = 14, mid_channels: int = 32,
                 out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # (B*T, out_channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)   # (B*T, out_channels)


class SpatialSpectralSSM(nn.Module):
    """Spatial-Spectral SSM for pixel-level crop classification.

    Architecture:
        Input (B, T=21, P=5, P=5, C=14)
          → SpatialEncoder (shared across T): (B, T, 64)
          → Linear(64 → d_model)
          → S4DBlock × n_layers
          → TemporalAttention → (B, d_model)
          → Dropout → Linear → (B, num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int,
                 spatial_out: int = 64, d_model: int = 128,
                 ssm_dim: int = 64, n_layers: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels, mid_channels=32, out_channels=spatial_out,
        )
        self.proj = nn.Linear(spatial_out, d_model)

        self.s4_layers = nn.ModuleList([
            S4DBlock(d_model, ssm_dim=ssm_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.temporal_attention = TemporalAttention(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._attn_weights = None

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """x: (B, T, P, P, C)"""
        B, T, P1, P2, C = x.shape

        # Reshape to (B*T, C, P, P) for spatial encoding
        x_flat = x.reshape(B * T, P1, P2, C).permute(0, 3, 1, 2)  # (B*T, C, P, P)
        spatial_feats = self.spatial_encoder(x_flat)                # (B*T, spatial_out)
        spatial_feats = spatial_feats.reshape(B, T, -1)             # (B, T, spatial_out)

        h = self.proj(spatial_feats)                                # (B, T, d_model)

        for layer in self.s4_layers:
            h = layer(h)

        context, attn_w = self.temporal_attention(h)
        self._attn_weights = attn_w.detach()

        logits = self.classifier(context)
        if return_attention:
            return logits, attn_w
        return logits


# ------------------------------------------------------------------ #
#  Baselines: TempCNN and L-TAE (for the benchmark table)
# ------------------------------------------------------------------ #

class TempCNN(nn.Module):
    """Temporal CNN baseline (Pelletier et al., Remote Sensing 2019).

    Three Conv1d(k=5)+BN+ReLU+Dropout blocks over the time axis, global
    average pooling, then a dense head.  T-agnostic (works for 7 or 21 steps).
    """

    def __init__(self, in_channels: int, num_classes: int,
                 filters: int = 64, dropout: float = 0.5):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=5, padding=2),
                nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.Dropout(dropout),
            )

        self.features = nn.Sequential(
            block(in_channels, filters),
            block(filters, filters),
            block(filters, filters),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(filters, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                 # (B, C, T)
        x = self.features(x).squeeze(-1)        # (B, filters)
        return self.head(x)


def _sinusoidal_pos(max_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape (max_len, d_model)."""
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class LTAEClassifier(nn.Module):
    """Lightweight Temporal Attention Encoder baseline (Garnot & Landrieu, 2020).

    Compact master-query temporal self-attention: each of ``n_head`` heads has a
    *single learned query* that attends over time, giving an efficient temporal
    pooling.  A faithful-in-spirit, dependency-free implementation.
    """

    def __init__(self, in_channels: int, num_classes: int,
                 d_model: int = 128, n_head: int = 16, dropout: float = 0.2,
                 max_len: int = 64):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.in_proj = nn.Linear(in_channels, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Parameter(torch.randn(n_head, self.d_k) / self.d_k ** 0.5)
        self.register_buffer("pos", _sinusoidal_pos(max_len, d_model))

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self._attn_weights = None

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T, _ = x.shape
        h = self.in_proj(x) + self.pos[:T].unsqueeze(0)          # (B, T, d_model)
        k = self.key(h).view(B, T, self.n_head, self.d_k)        # (B, T, H, d_k)
        v = h.view(B, T, self.n_head, self.d_k)                  # (B, T, H, d_k)

        scores = torch.einsum("bthd,hd->bth", k, self.query) / self.d_k ** 0.5
        attn = torch.softmax(scores, dim=1)                     # over time
        ctx = torch.einsum("bth,bthd->bhd", attn, v).reshape(B, -1)  # (B, d_model)

        logits = self.classifier(self.mlp(ctx))
        attn_w = attn.mean(dim=-1)                              # (B, T), head-avg
        self._attn_weights = attn_w.detach()
        if return_attention:
            return logits, attn_w
        return logits


# ------------------------------------------------------------------ #
#  Proposed model: Multi-Scale State-Space network (MS-S4)
# ------------------------------------------------------------------ #

class MultiScaleS4(nn.Module):
    """Multi-Scale State-Space network — the proposed architecture.

    Unifies the two prior experiments: MSTACNN's *multi-scale temporal
    convolution* front-end (local phenological transitions at k=2,3,5) feeds an
    *S4D state-space* backbone (global, full-season temporal mixing), pooled by
    temporal attention.  Local feature extraction + long-range state-space
    modelling in a single network.

        Input (B, T, C)
          → MultiScaleConvBlock(C→64) → ReLU
          → MultiScaleConvBlock(64→d_model) + residual → ReLU
          → S4DBlock × n_layers
          → TemporalAttention(d_model)
          → Dropout → Linear → num_classes
    """

    def __init__(self, in_channels: int, num_classes: int,
                 d_model: int = 128, ssm_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.2,
                 use_multiscale: bool = True, use_attention: bool = True,
                 use_region_adv: bool = False, n_regions: int = 0):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention
        self.use_region_adv = use_region_adv
        self.grl_lambda = 1.0   # set per-batch by the trainer during ramp-up

        if use_multiscale:
            self.ms_block1 = MultiScaleConvBlock(in_channels, 64)
            self.ms_block2 = MultiScaleConvBlock(64, d_model)
            self.residual = nn.Conv1d(64, d_model, kernel_size=1)
        else:
            # Single-scale conv front-end of matched depth. Isolates the
            # contribution of the *multi-scale* kernels in the ablation.
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
            )

        self.s4_layers = nn.ModuleList([
            S4DBlock(d_model, ssm_dim=ssm_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        if use_attention:
            self.temporal_attention = TemporalAttention(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, num_classes),
        )
        # Region-invariant (district-adversarial) head on the pooled features.
        # Trained through a gradient-reversal layer so the backbone learns
        # representations that do not betray the district, narrowing the
        # geographic coverage gap at its source. Inactive at inference.
        if use_region_adv:
            assert n_regions > 1, "use_region_adv requires n_regions > 1"
            self.region_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d_model // 2, n_regions),
            )
        self._attn_weights = None

    def forward(self, x: torch.Tensor, return_attention: bool = False,
                return_region: bool = False):
        x = x.permute(0, 2, 1)                            # (B, C, T)
        if self.use_multiscale:
            h1 = F.relu(self.ms_block1(x))                # (B, 64, T)
            h2 = self.ms_block2(h1)                       # (B, d_model, T)
            h2 = F.relu(h2 + self.residual(h1))           # residual
        else:
            h2 = F.relu(self.proj(x))                     # (B, d_model, T)

        h = h2.permute(0, 2, 1)                           # (B, T, d_model)
        for layer in self.s4_layers:
            h = layer(h)

        if self.use_attention:
            context, attn_w = self.temporal_attention(h)
            self._attn_weights = attn_w.detach()
        else:
            context = h.mean(dim=1)                       # mean-pool over time
            attn_w = None
        logits = self.classifier(context)
        if return_region and self.use_region_adv:
            region_logits = self.region_head(grad_reverse(context, self.grl_lambda))
            return logits, region_logits
        if return_attention:
            return logits, attn_w
        return logits


class TemporalTransformer(nn.Module):
    """Transformer-encoder baseline for parcel-level SITS.

    Multi-head temporal self-attention over the per-field T x C monthly sequence.
    This is the parcel-vector analogue of attention-based SITS encoders. Note that
    spatio-temporal transformers such as TSViT operate on pixel/patch cubes and
    require a spatial dimension the field-aggregated representation does not have,
    so the temporal encoder is the appropriate transformer baseline here.

        Input (B, T, C)
          -> Linear C->d_model + learned positional encoding
          -> TransformerEncoder (nhead, n_layers)
          -> LayerNorm + mean-pool over time
          -> Dropout -> Linear -> num_classes
    """

    def __init__(self, in_channels: int, num_classes: int,
                 d_model: int = 128, nhead: int = 4, n_layers: int = 2,
                 dropout: float = 0.2, max_len: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        T = x.size(1)
        h = self.input_proj(x) + self.pos[:, :T, :]       # (B, T, d_model)
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))                       # mean-pool over time
        logits = self.classifier(h)
        if return_attention:
            return logits, None
        return logits


# ------------------------------------------------------------------ #
#  Shared factory — used by pipeline/train.py and inference/predict.py
# ------------------------------------------------------------------ #

# Models that expose temporal attention via forward(..., return_attention=True).
ATTENTION_MODELS = {"mstacnn", "s4d", "ms-s4", "ltae", "spatial-ssm", "ms-s4-noms"}

# Torch model types handled by build_model (excludes the sklearn 'rf').
TORCH_MODELS = {
    "mstacnn", "cnn", "tempcnn", "lstm", "ltae", "s4d", "ms-s4", "spatial-ssm",
    "transformer", "ms-s4-noattn", "ms-s4-noms", "ms-s4-bb",
}


def build_model(model_type: str, in_channels: int, num_classes: int,
                region_adv: bool = False, n_regions: int = 0) -> nn.Module:
    """Instantiate a torch model by name with the project's standard configs.

    Constructor arguments are fixed here so training and inference build
    identical architectures (state_dicts load cleanly). ``region_adv`` adds the
    district-adversarial head to PhenoSSM (ms-s4) for region-invariant training.
    """
    mt = model_type.lower()
    if region_adv and mt in ("ms-s4", "mss4", "ms4"):
        return MultiScaleS4(in_channels=in_channels, num_classes=num_classes,
                            use_region_adv=True, n_regions=n_regions)
    if mt == "mstacnn":
        return MSTACNN(in_channels=in_channels, num_classes=num_classes)
    if mt == "cnn":
        return Conv1DClassifier(in_channels=in_channels, num_classes=num_classes)
    if mt == "tempcnn":
        return TempCNN(in_channels=in_channels, num_classes=num_classes)
    if mt == "lstm":
        return LSTMClassifier(in_channels=in_channels, num_classes=num_classes)
    if mt == "ltae":
        return LTAEClassifier(in_channels=in_channels, num_classes=num_classes)
    if mt == "s4d":
        return S4DClassifier(in_channels=in_channels, num_classes=num_classes,
                             d_model=128, ssm_dim=64, n_layers=4, dropout=0.2)
    if mt in ("ms-s4", "mss4", "ms4"):
        return MultiScaleS4(in_channels=in_channels, num_classes=num_classes)
    if mt == "ms-s4-noattn":   # ablation: multi-scale conv + S4D, mean-pool (no attention)
        return MultiScaleS4(in_channels=in_channels, num_classes=num_classes,
                            use_attention=False)
    if mt == "ms-s4-noms":     # ablation: single-scale conv + S4D + attention
        return MultiScaleS4(in_channels=in_channels, num_classes=num_classes,
                            use_multiscale=False)
    if mt == "ms-s4-bb":       # ablation: single-scale conv + S4D, mean-pool (backbone)
        return MultiScaleS4(in_channels=in_channels, num_classes=num_classes,
                            use_multiscale=False, use_attention=False)
    if mt == "transformer":
        return TemporalTransformer(in_channels=in_channels, num_classes=num_classes)
    if mt == "spatial-ssm":
        return SpatialSpectralSSM(in_channels=in_channels, num_classes=num_classes,
                                  spatial_out=64, d_model=128, ssm_dim=64,
                                  n_layers=4, dropout=0.2)
    raise ValueError(f"Unknown torch model type: {model_type}")
