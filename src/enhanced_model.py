"""
EnhancedHierCVAE model components.

This module contains:
- TemporalAttention: multi-head self-attention over LSTM outputs.
- EnhancedHierCVAE: per-apartment CVAE-style model with:
    * Bidirectional LSTM over time windows
    * Encoded lag/diff/rolling statistics features
    * Global latent variable z_g with condition-dependent prior
    * Residual connections from short lags
- EnhancedLoss: reconstruction (Huber) + KL divergence on z_g.

The implementation is refactored from `main_hier_enhanced.py`
without changing the original logic.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal self-attention over LSTM outputs.

    A learnable query attends over the sequence of hidden states to
    produce a single summary vector for the current window.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_dim)

        Returns:
            output: (batch_size, hidden_dim)
            attn_weights: (batch_size, num_heads, 1, seq_len)
        """
        batch_size = lstm_output.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        attn_output, attn_weights = self.multihead_attn(
            query, lstm_output, lstm_output
        )

        # Residual connection with last time step
        last_step = lstm_output[:, -1:, :]
        attn_output = self.layer_norm(attn_output + last_step)
        output = self.output_proj(attn_output.squeeze(1))
        return output, attn_weights


class EnhancedHierCVAE(nn.Module):
    """
    Enhanced HierCVAE model for single-apartment forecasting.

    Architecture:
    1. Temporal encoding: bidirectional LSTM + temporal attention.
    2. Feature fusion: lag, diff, and rolling statistics encoded and
       concatenated with temporal summary.
    3. Global latent space: z_g captures global patterns conditioned
       on temporal features + conditions (+ lag stats in prior).
    4. Decoder: combines z_g, temporal features and conditions to
       predict the next-step normalized target.
    5. Residual connections: learnable weights on short lags (lag1, lag2)
       to emphasize short-term persistence.
    """

    def __init__(
        self,
        cont_cond_dim: int,
        num_lag_features: int = 6,
        num_diff_features: int = 2,
        global_latent_dim: int = 64,
        cat_dims=None,
        embedding_dims=None,
        lookback_steps: int = 24,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        self.cont_cond_dim = cont_cond_dim
        self.num_lag_features = num_lag_features
        self.num_diff_features = num_diff_features
        self.global_latent_dim = global_latent_dim
        self.lookback_steps = lookback_steps
        self.hidden_dim = hidden_dim

        cat_dims = cat_dims or {"Hour": 24, "DayOfWeek": 7}
        embedding_dims = embedding_dims or {"Hour": 16, "DayOfWeek": 8}

        # Categorical embeddings for Hour and DayOfWeek
        self.hour_embed = nn.Embedding(cat_dims["Hour"], embedding_dims["Hour"])
        self.day_embed = nn.Embedding(cat_dims["DayOfWeek"], embedding_dims["DayOfWeek"])
        total_emb_dim = embedding_dims["Hour"] + embedding_dims["DayOfWeek"]
        self.cond_dim = cont_cond_dim + total_emb_dim

        # Encoder for lag + diff + rolling stats
        lag_diff_input_dim = num_lag_features + num_diff_features + 2  # +2 for roll stats
        self.lag_encoder = nn.Sequential(
            nn.Linear(lag_diff_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        lag_feat_dim = hidden_dim // 2  # e.g. 64

        # Bidirectional LSTM over time windows
        # Each step has: cont_cond_dim + 2 categorical + 1 target value
        temporal_input_dim = cont_cond_dim + 2 + 1
        self.lstm = nn.LSTM(
            input_size=temporal_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_output_dim = hidden_dim * 2

        # Temporal attention summarizing LSTM outputs
        self.temporal_attention = TemporalAttention(
            hidden_dim=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Fused temporal feature dimension
        temporal_feat_dim = lstm_output_dim + lag_feat_dim  # e.g. 256 + 64 = 320

        # Global encoder q(z_g | temporal_feat, c)
        global_enc_input = temporal_feat_dim + self.cond_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_enc_input, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.global_mu = nn.Linear(hidden_dim, global_latent_dim)
        self.global_logvar = nn.Linear(hidden_dim, global_latent_dim)

        # Global prior p(z_g | c, lag_encoded)
        self.global_prior = nn.Sequential(
            nn.Linear(self.cond_dim + lag_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, global_latent_dim * 2),
        )

        # Decoder (no local latent here)
        decoder_input = global_latent_dim + temporal_feat_dim + self.cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learnable residual weights for short lags
        self.residual_weight_1 = nn.Parameter(torch.tensor(0.5))
        self.residual_weight_2 = nn.Parameter(torch.tensor(0.3))

        self.temporal_feat_dim = temporal_feat_dim

    def process_conditions(self, c_cont: torch.Tensor, c_cat: torch.Tensor) -> torch.Tensor:
        """
        Embed categorical time features and concatenate with continuous features.

        Args:
            c_cont: (batch_size, cont_cond_dim)
            c_cat:  (batch_size, 2) - [Hour, DayOfWeek]
        """
        c_cat = c_cat.long()
        h = self.hour_embed(c_cat[:, 0])
        d = self.day_embed(c_cat[:, 1])
        return torch.cat([c_cont, h, d], dim=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard reparameterization trick for Gaussian latents."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x_window: torch.Tensor,
        c_cont: torch.Tensor,
        c_cat: torch.Tensor,
        lag_feats: torch.Tensor,
        diff_feats: torch.Tensor,
        roll_stats: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (used for both training and evaluation).

        Args:
            x_window:   (batch_size, lookback_steps * (cont+cat+1))
            c_cont:     (batch_size, cont_cond_dim)
            c_cat:      (batch_size, 2)
            lag_feats:  (batch_size, num_lag_features)
            diff_feats: (batch_size, num_diff_features)
            roll_stats: (batch_size, 2)
            targets:    (batch_size, 1), currently not used inside the model
                        but kept for API symmetry with the original function.
        """
        batch_size = x_window.shape[0]
        c = self.process_conditions(c_cont, c_cat)

        # Encode lag + diff + rolling stats
        lag_diff_input = torch.cat([lag_feats, diff_feats, roll_stats], dim=1)
        lag_encoded = self.lag_encoder(lag_diff_input)

        # LSTM over time window
        features_per_step = self.cont_cond_dim + 2 + 1
        x_window = x_window.view(batch_size, self.lookback_steps, features_per_step)
        lstm_out, _ = self.lstm(x_window)

        # Temporal attention
        attn_out, attn_weights = self.temporal_attention(lstm_out)

        # Fused temporal feature
        temporal_feat = torch.cat([attn_out, lag_encoded], dim=1)

        # Global encoder q(z_g | temporal_feat, c)
        global_enc_input = torch.cat([temporal_feat, c], dim=1)
        global_h = self.global_encoder(global_enc_input)
        global_mu_q = self.global_mu(global_h)
        global_logvar_q = self.global_logvar(global_h)

        # Global prior p(z_g | c, lag_encoded)
        prior_input = torch.cat([c, lag_encoded], dim=1)
        global_prior_out = self.global_prior(prior_input)
        global_mu_p, global_logvar_p = torch.chunk(global_prior_out, 2, dim=1)

        # Sample global latent
        z_g = self.reparameterize(global_mu_q, global_logvar_q)

        # Decode (no local latent)
        decoder_input = torch.cat([z_g, temporal_feat, c], dim=1)
        decoder_out = self.decoder(decoder_input)

        # Residual connection using short-term lags
        lag1 = lag_feats[:, 0:1]
        lag2 = lag_feats[:, 1:2]
        pred = (
            decoder_out
            + torch.sigmoid(self.residual_weight_1) * lag1
            + torch.sigmoid(self.residual_weight_2) * lag2
        )

        return {
            "pred": pred,
            "global_mu_q": global_mu_q,
            "global_logvar_q": global_logvar_q,
            "global_mu_p": global_mu_p,
            "global_logvar_p": global_logvar_p,
            "attention_weights": attn_weights,
        }


class EnhancedLoss(nn.Module):
    """
    Loss for the enhanced per-apartment model.

    Components:
    - Reconstruction loss: Smooth L1 (Huber) loss on normalized targets.
    - KL divergence: between q(z_g | temporal,c) and p(z_g | c, lag_stats),
      optionally scaled by `beta_factor` for KL warmup.
    """

    def __init__(self, beta_global: float = 0.0001):
        super().__init__()
        self.beta_global = beta_global

    @staticmethod
    def kl_divergence(
        mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor
    ) -> torch.Tensor:
        """
        KL(q || p) for diagonal Gaussians.

        Returns the mean KL over the batch.
        """
        kl = 0.5 * torch.sum(
            logvar_p
            - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1,
            dim=-1,
        )
        return kl.mean()

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, beta_factor: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss.

        Args:
            outputs: dict returned by EnhancedHierCVAE.forward()
            targets: (batch_size, 1)
            beta_factor: warmup factor for the KL term.
        """
        pred = outputs["pred"]

        # Robust reconstruction loss
        recon_loss = F.smooth_l1_loss(pred, targets, beta=0.5)

        kl_global = self.kl_divergence(
            outputs["global_mu_q"],
            outputs["global_logvar_q"],
            outputs["global_mu_p"],
            outputs["global_logvar_p"],
        )

        total = recon_loss + beta_factor * self.beta_global * kl_global

        return {"total": total, "recon": recon_loss, "kl_global": kl_global}


