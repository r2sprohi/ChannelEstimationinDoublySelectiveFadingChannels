"""
dsce_model.py
Doubly-Selective Channel Estimation (DSCE) Transformer Network.

Architecture (Shi et al., ICC 2025):
  - Encoder : 3 Transformer blocks with RPE self-attention (over N_p pilot tokens)
  - Decoder : 1 Transformer block with RPE cross-attention
              (Q = all N_f×N_t grid tokens, K/V = encoder output at pilots)
  - Feature dimension d = 32
  - Relative Positional Encoding (RPE): 2-D sine-encoded lookup table
    Avoids the prohibitive O(N_q × N_k × d_rpe) intermediate tensor by
    precomputing a small (2N_t-1)×(2N_f-1) RPE table and then gathering.
  - Pilot augmentation: 20% random dropout during training

Memory complexity per attention layer:
  Encoder self-attention : O(N_p²)         N_p ≤ 576   → tiny
  Decoder cross-attention: O(N_total × N_p) N_total ≤ 5376, N_p ≤ 576 → ~200 MB
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sine_encode_2d


# ---------------------------------------------------------------------------
# Efficient RPE multi-head attention (lookup-table approach)
# ---------------------------------------------------------------------------

class RPEMultiHeadAttention(nn.Module):
    """
    Multi-head attention with 2-D Relative Positional Encoding.

    Positions are passed as integer grid indices (t, f). The RPE bias is
    built from a per-forward-pass lookup table of size
    (2*N_t-1) × (2*N_f-1) × n_heads, avoiding the O(N_q×N_k×d_rpe)
    intermediate tensor.

    Supports both self-attention (q_pos == k_pos) and cross-attention.
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_rpe: int = 32):
        super().__init__()
        assert d_model % n_heads == 0
        assert d_rpe % 4 == 0, "d_rpe must be divisible by 4 for 2D sine encoding"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_rpe = d_rpe

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # RPE projection: d_rpe → n_heads (projects sine-encoded Δ to per-head bias)
        self.W_rpe = nn.Linear(d_rpe, n_heads, bias=False)

    # ---- Lookup-table RPE bias ----------------------------------------
    def _rpe_bias(self,
                  t_q: torch.Tensor, f_q: torch.Tensor,
                  t_k: torch.Tensor, f_k: torch.Tensor,
                  N_t: int, N_f: int) -> torch.Tensor:
        """
        Compute RPE bias using a 2-D lookup table for efficiency.

        t_q, f_q : (N_q,) int – query positions
        t_k, f_k : (N_k,) int – key positions
        Returns  : (n_heads, N_q, N_k)
        """
        device = t_q.device

        # Step 1: build lookup table for all unique (Δt, Δf) pairs
        dt_range = torch.arange(-(N_t - 1), N_t,   dtype=torch.float32, device=device)
        df_range = torch.arange(-(N_f - 1), N_f,   dtype=torch.float32, device=device)

        dt_norm = dt_range / max(N_t - 1, 1)   # normalise to [-1, 1]
        df_norm = df_range / max(N_f - 1, 1)

        DT, DF = torch.meshgrid(dt_norm, df_norm, indexing='ij')  # (2Nt-1, 2Nf-1)
        rel_pos_all = torch.stack([DT, DF], dim=-1)                # (2Nt-1, 2Nf-1, 2)

        # Sine encode then project: (2Nt-1, 2Nf-1, n_heads)
        rpe_enc   = sine_encode_2d(rel_pos_all, self.d_rpe)        # (2Nt-1, 2Nf-1, d_rpe)
        rpe_table = self.W_rpe(rpe_enc)                             # (2Nt-1, 2Nf-1, n_heads)

        n_df = 2 * N_f - 1
        rpe_flat = rpe_table.view(-1, self.n_heads)                 # ((2Nt-1)*(2Nf-1), n_heads)

        # Step 2: compute integer offset indices for every (q, k) pair
        # Δt[i,j] = t_q[i] - t_k[j],  shifted to [0, 2Nt-2]
        dt_idx = (t_q[:, None] - t_k[None, :]) + (N_t - 1)        # (N_q, N_k)
        df_idx = (f_q[:, None] - f_k[None, :]) + (N_f - 1)        # (N_q, N_k)
        flat_idx = (dt_idx * n_df + df_idx).clamp(0, rpe_flat.shape[0] - 1)

        # Step 3: gather bias values  (N_q, N_k, n_heads) → (n_heads, N_q, N_k)
        rpe_bias = rpe_flat[flat_idx]                               # (N_q, N_k, n_heads)
        return rpe_bias.permute(2, 0, 1)                            # (n_heads, N_q, N_k)

    # ---- Forward --------------------------------------------------------
    def forward(self,
                q_tokens: torch.Tensor,
                k_tokens: torch.Tensor,
                v_tokens: torch.Tensor,
                t_q: torch.Tensor, f_q: torch.Tensor,
                t_k: torch.Tensor, f_k: torch.Tensor,
                N_t: int, N_f: int,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        q_tokens : (B, N_q, d_model)
        k_tokens : (B, N_k, d_model)
        v_tokens : (B, N_k, d_model)
        t_q, f_q : (N_q,) int – query symbol / subcarrier indices
        t_k, f_k : (N_k,) int – key symbol / subcarrier indices
        N_t, N_f : grid dimensions (for RPE normalisation)
        attn_mask: (N_q, N_k) additive mask (use -inf for padding)
        Returns  : (B, N_q, d_model)
        """
        B, N_q, _ = q_tokens.shape
        N_k = k_tokens.shape[1]
        h, dk = self.n_heads, self.d_k
        scale = math.sqrt(dk)

        Q = self.W_q(q_tokens).view(B, N_q, h, dk).transpose(1, 2)  # (B,h,N_q,dk)
        K = self.W_k(k_tokens).view(B, N_k, h, dk).transpose(1, 2)  # (B,h,N_k,dk)
        V = self.W_v(v_tokens).view(B, N_k, h, dk).transpose(1, 2)  # (B,h,N_k,dk)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale        # (B,h,N_q,N_k)

        # RPE bias: (n_heads, N_q, N_k) – independent of batch
        rpe_bias = self._rpe_bias(t_q, f_q, t_k, f_k, N_t, N_f)
        scores = scores + rpe_bias.unsqueeze(0)                       # broadcast over B

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, V)                                  # (B,h,N_q,dk)
        out  = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Feed-forward sub-layer
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Self-attention Transformer block (used in Encoder)
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Pre-LN Transformer block with RPE self-attention (Encoder use)."""

    def __init__(self, d_model: int, n_heads: int = 4, d_rpe: int = 32, d_ff: int = None):
        super().__init__()
        self.attn  = RPEMultiHeadAttention(d_model, n_heads, d_rpe)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                t: torch.Tensor, f: torch.Tensor,
                N_t: int, N_f: int) -> torch.Tensor:
        """
        x    : (B, N, d_model)
        t, f : (N,) int – symbol / subcarrier indices
        """
        xn = self.norm1(x)
        x  = x + self.attn(xn, xn, xn, t, f, t, f, N_t, N_f)
        x  = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Cross-attention Transformer block (used in Decoder)
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Pre-LN Transformer block with RPE cross-attention (Decoder use).
    Queries come from all grid tokens; keys/values from encoder output.
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_rpe: int = 32, d_ff: int = None):
        super().__init__()
        self.attn  = RPEMultiHeadAttention(d_model, n_heads, d_rpe)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)

    def forward(self,
                q: torch.Tensor, kv: torch.Tensor,
                t_q: torch.Tensor, f_q: torch.Tensor,
                t_k: torch.Tensor, f_k: torch.Tensor,
                N_t: int, N_f: int) -> torch.Tensor:
        """
        q   : (B, N_total, d_model) – all grid positions
        kv  : (B, N_p, d_model)    – encoder output at pilot positions
        """
        q_norm  = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        q = q + self.attn(q_norm, kv_norm, kv_norm,
                          t_q, f_q, t_k, f_k, N_t, N_f)
        q = q + self.ff(self.norm2(q))
        return q


# ---------------------------------------------------------------------------
# DSCE Transformer
# ---------------------------------------------------------------------------

class DSCETransformer(nn.Module):
    """
    Doubly-Selective Channel Estimation Transformer.

    Encoder: 3 self-attention blocks over N_p pilot tokens.
    Decoder: 1 cross-attention block (Q=full grid, K/V=encoder output).

    Input token: [Re(h_LS), Im(h_LS), t_norm, f_norm]  (4-dim → d_model)
    Non-pilot decoder queries: [0, 0, t_norm, f_norm]
    Output: [Re(H), Im(H)] for all N_f × N_t tokens.
    """

    def __init__(self,
                 d_model: int = 32,
                 n_encoder_blocks: int = 3,
                 n_decoder_blocks: int = 1,
                 n_heads: int = 4,
                 d_rpe: int = 32):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(4, d_model)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_rpe)
            for _ in range(n_encoder_blocks)
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_rpe)
            for _ in range(n_decoder_blocks)
        ])

        self.output_head = nn.Linear(d_model, 2)

    # ------------------------------------------------------------------
    def forward(self,
                h_ls_ri: torch.Tensor,
                pilot_f_idx: torch.Tensor,
                pilot_t_idx: torch.Tensor,
                all_f_idx: torch.Tensor,
                all_t_idx: torch.Tensor,
                n_f: int,
                n_t: int,
                pilot_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        h_ls_ri     : (B, N_p, 2) – [Re(h_LS), Im(h_LS)] at pilots
        pilot_f_idx : (N_p,) long
        pilot_t_idx : (N_p,) long
        all_f_idx   : (N_total,) long – all subcarrier indices (row-major by t)
        all_t_idx   : (N_total,) long – all symbol indices
        n_f, n_t    : grid dimensions
        pilot_mask  : (B, N_p) bool, True=keep (pilot augmentation)

        Returns
        -------
        h_est_ri : (B, N_total, 2)
        """
        B      = h_ls_ri.shape[0]
        device = h_ls_ri.device

        # ---- Normalised pilot position features ----------------------
        t_p_norm = pilot_t_idx.float() / max(n_t - 1, 1)   # (N_p,)
        f_p_norm = pilot_f_idx.float() / max(n_f - 1, 1)   # (N_p,)
        pilot_tf = torch.stack([t_p_norm, f_p_norm], dim=-1)          # (N_p, 2)
        pilot_tf = pilot_tf.unsqueeze(0).expand(B, -1, -1)            # (B, N_p, 2)

        # ---- Build pilot input tokens --------------------------------
        pilot_tokens = torch.cat([h_ls_ri, pilot_tf], dim=-1)         # (B, N_p, 4)

        if pilot_mask is not None:
            pilot_tokens = pilot_tokens * pilot_mask.unsqueeze(-1).float()

        enc_in = self.input_proj(pilot_tokens)                         # (B, N_p, d)

        # ---- Encoder (self-attention over pilots) --------------------
        enc_out = enc_in
        for blk in self.encoder_blocks:
            enc_out = blk(enc_out, pilot_t_idx, pilot_f_idx, n_t, n_f)

        # ---- Build decoder query tokens (all grid positions) ---------
        t_a_norm = all_t_idx.float() / max(n_t - 1, 1)
        f_a_norm = all_f_idx.float() / max(n_f - 1, 1)
        all_tf = torch.stack([t_a_norm, f_a_norm], dim=-1)             # (N_total, 2)
        all_tf = all_tf.unsqueeze(0).expand(B, -1, -1)                 # (B, N_total, 2)

        zeros   = torch.zeros(B, len(all_f_idx), 2, device=device)
        dec_q   = self.input_proj(torch.cat([zeros, all_tf], dim=-1))  # (B, N_total, d)

        # ---- Decoder (cross-attention: Q=grid, K/V=encoder output) --
        dec_out = dec_q
        for blk in self.decoder_blocks:
            dec_out = blk(dec_out, enc_out,
                          all_t_idx, all_f_idx,
                          pilot_t_idx, pilot_f_idx,
                          n_t, n_f)

        return self.output_head(dec_out)                               # (B, N_total, 2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dsce_model(d_model: int = 32,
                     n_encoder_blocks: int = 3,
                     n_decoder_blocks: int = 1,
                     n_heads: int = 4) -> DSCETransformer:
    model = DSCETransformer(
        d_model=d_model,
        n_encoder_blocks=n_encoder_blocks,
        n_decoder_blocks=n_decoder_blocks,
        n_heads=n_heads,
        d_rpe=d_model,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DSCE] d={d_model}, enc={n_encoder_blocks}, dec={n_decoder_blocks}, "
          f"heads={n_heads} | params={n_params:,}")
    return model
