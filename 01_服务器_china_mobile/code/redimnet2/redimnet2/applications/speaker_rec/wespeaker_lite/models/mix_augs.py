import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


def rms_normalize(X, eps=1e-8):
    """Per-sample RMS (energy) normalization to unit power.

    Mixup/Cutmix combine two source utterances. If one source is much louder than
    the other it dominates the mixture regardless of the mixing weight, so the soft
    label no longer reflects the acoustic content (label noise). Normalizing each
    source to equal energy first makes the mixing coefficient a faithful "presence"
    weight (cf. Margin-Mixup, Mertens et al. 2023, Eq. 1). The model front-end
    applies spectral mean subtraction, which absorbs the resulting global scale
    change, so normalizing to unit RMS is benign downstream.

    Normalizes over all non-batch dimensions, so it works for raw waveforms shaped
    (B, T) or (B, 1, T) as well as 2D feature maps.
    """
    bs = X.shape[0]
    rms = X.reshape(bs, -1).pow(2).mean(dim=1, keepdim=True).clamp_min(eps).sqrt()
    return X / rms.reshape([bs] + [1] * (X.dim() - 1))


class Cutmix(nn.Module):
    def __init__(self, mix_beta, energy_normalized=True):
        super(Cutmix, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.energy_normalized = energy_normalized

    def forward(self, X, Y, weight=None):
        bs = X.shape[0]

        n_dims = len(X.shape)
        # Squeeze the input if necessary to handle the dimensions
        unsqueeze = False
        if n_dims == 3:
            X = X.squeeze(1)
            n_dims = len(X.shape)
            unsqueeze = True

        if n_dims != 2:
            raise ValueError("Input dimensions must be 2D")

        # Equalize source energies so the duration-fraction label is a valid proxy
        # for each speaker's contribution to the pooled embedding.
        if self.energy_normalized:
            X = rms_normalize(X)

        # Generate permutation and mix coefficients
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.sample((bs,)).to(X.device).to(X.dtype)

        _, T = X.size()

        cut_ratios = coeffs
        cut_froms = (T * cut_ratios).round().long()

        Y_cutmix = Y.clone()
        X_cutmix = X.clone()
        for i in range(bs):
            cut_from = cut_froms[i].item()
            # Replace the segment from cut_from to end with the corresponding segment from another sample
            X_cutmix[i, cut_from:T] = X[perm[i], cut_from:T]

            # Mix the labels proportionally
            Y_cutmix[i] = cut_ratios[i] * Y[i] + (1 - cut_ratios[i]) * Y[perm[i]]

        # Unsqueeze the input back to the original shape if it was squeezed
        if unsqueeze:
            X_cutmix = X_cutmix.unsqueeze(1)

        if weight is None:
            return X_cutmix, Y_cutmix
        else:
            return X_cutmix, Y_cutmix, weight


class Mixup(nn.Module):
    def __init__(self, mix_beta, energy_normalized=True):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.energy_normalized = energy_normalized

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device).to(X.dtype)

        # Energy-normalize each source so the mix coefficient is a faithful
        # presence weight rather than being dominated by the louder utterance.
        if self.energy_normalized:
            X = rms_normalize(X)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1).to(Y.dtype) * Y + (1 - coeffs.view(-1, 1).to(Y.dtype)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight


class MixCutUp(nn.Module):
    def __init__(self,
            mixup_beta=0.5,
            cutmix_beta=1.0,
            mixup_prob=0.5,
            energy_normalized=True
        ):
        super(MixCutUp, self).__init__()
        self.mixup_beta_distribution = Beta(mixup_beta, mixup_beta)
        self.cutmix_beta_distribution = Beta(cutmix_beta, cutmix_beta)
        self.mixup_prob = mixup_prob
        self.energy_normalized = energy_normalized

    def forward(self, X, Y):
        bs = X.shape[0]

        n_dims = len(X.shape)
        # Squeeze the input if necessary to handle the dimensions
        unsqueeze = False
        if n_dims == 3:
            X = X.squeeze(1)
            n_dims = len(X.shape)
            unsqueeze = True

        if n_dims != 2:
            raise ValueError("Input dimensions must be 2D")

        # Equalize source energies once, up front: both the mixup branch (additive)
        # and the cutmix branch (temporal splice) then reflect the intended ratio.
        if self.energy_normalized:
            X = rms_normalize(X)

        # Generate permutation and mix coefficients
        perm = torch.randperm(bs)
        coeffs_mixup = self.mixup_beta_distribution.sample((bs,)).to(X.device).to(X.dtype)
        coeffs_cutmix = self.cutmix_beta_distribution.sample((bs,)).to(X.device).to(X.dtype)

        _, T = X.size()

        cut_froms = (T * coeffs_cutmix).round().long()

        apply_mixup = np.random.uniform(size=(bs,))

        Y_mix = Y.clone()
        X_mix = X.clone()
        for i in range(bs):
            if apply_mixup[i] < self.mixup_prob:
                # MixUp operation
                X_mix[i] = coeffs_mixup[i] * X[i] + (1 - coeffs_mixup[i]) * X[perm[i]]
                Y_mix[i] = coeffs_mixup[i] * Y[i] + (1 - coeffs_mixup[i]) * Y[perm[i]]
            else:
                cut_from = cut_froms[i].item()
                # Replace the segment from cut_from to end with the corresponding segment from another sample
                X_mix[i, cut_from:T] = X[perm[i], cut_from:T]

                # Mix the labels proportionally
                Y_mix[i] = coeffs_cutmix[i] * Y[i] + (1 - coeffs_cutmix[i]) * Y[perm[i]]

        # Unsqueeze the input back to the original shape if it was squeezed
        if unsqueeze:
            X_mix = X_mix.unsqueeze(1)

        return X_mix, Y_mix

class OverlapMix(nn.Module):
    def __init__(self,
            concat_beta=0.5,
            overlap_beta=1.0,
            min_scale=0.4,
            energy_normalized=True,
            rel_db_range=(-10.0, 10.0),
        ):
        #   OverlapMix simulates conversational speaker overlap — unlike Mixup (global blend) or Cutmix (non-overlapping splice), it creates a realistic timeline:
        #   |--- spk0 solo ---|--- overlap ---|--- spk1 solo ---|
        #   0              start1          end0                  T
        #   Speaker 0 is active on [0, t+h], speaker 1 on [t-h, T], and the overlap window is [t-h, t+h] with length 2h. Distributions controlling the shape:
        #   - concat_beta(0.5, 0.5) — U-shaped, so extreme splits (one speaker dominates duration) are more common than 50/50
        #   - overlap_beta(1.0, 1.0) — uniform overlap fraction
        #
        #   Leveling modes (see OverlapMixVad for the full rationale):
        #   - energy_normalized=True (legacy): unit-RMS sources, speaker 0 attenuated
        #     by scale ~ Uniform(min_scale, 1.0). Only for z-normalizing front-ends.
        #   - energy_normalized=False (level-faithful): speaker 0 untouched, intruder
        #     re-leveled to rel ~ Uniform(rel_db_range) dB vs speaker 0 (full-clip
        #     RMS anchor here — no VAD in this variant).
        #
        #   The label lam is an amplitude-weighted presence ratio
        #   m0*amp0 / (m0*amp0 + m1*amp1); duration × amplitude is the right
        #   contribution proxy for mean-style pooling (linear, not squared).

        super(OverlapMix, self).__init__()
        self.energy_normalized = energy_normalized
        self.concat_beta_distribution = Beta(concat_beta, concat_beta)
        self.overlap_beta_distribution = Beta(overlap_beta, overlap_beta)
        self.min_scale = min_scale
        self.rel_db_range = tuple(rel_db_range)

    def forward(self, X, Y):
        bs = X.shape[0]
        device = X.device

        n_dims = len(X.shape)
        # Squeeze the input if necessary to handle the dimensions
        unsqueeze = False
        if n_dims == 3:
            X = X.squeeze(1)
            n_dims = len(X.shape)
            unsqueeze = True

        if n_dims != 2:
            raise ValueError("Input dimensions must be 2D")

        # Equalize source energies once, up front: both the mixup branch (additive)
        # and the cutmix branch (temporal splice) then reflect the intended ratio.
        if self.energy_normalized:
            X = rms_normalize(X)

        # Generate permutation and mix coefficients
        perm = torch.randperm(bs, device=device)
        u = torch.rand(bs, device=device)
        if self.energy_normalized:
            # Legacy: unit-RMS sources, speaker 0 attenuated by scale.
            s0 = u * (1.0 - self.min_scale) + self.min_scale
            s1 = torch.ones_like(s0)
            amp0, amp1 = s0, s1
        else:
            # Level-faithful: speaker 0 untouched; intruder placed at a target
            # SIR rel dB relative to speaker 0 (full-clip RMS anchor).
            lo, hi = self.rel_db_range
            rel = 10.0 ** ((lo + u * (hi - lo)) / 20.0)
            a = X.pow(2).mean(dim=1).clamp_min(1e-16).sqrt()
            s0 = torch.ones_like(rel)
            s1 = rel * a / a[perm].clamp_min(1e-8)
            amp0, amp1 = torch.ones_like(rel), rel
        _, T = X.size()

        overlap_masks_dict = sample_overlap_speaker_masks(
            bs=bs, L=T,
            concat_beta=self.concat_beta_distribution,
            overlap_beta=self.overlap_beta_distribution,
            device=X.device,
        )

        mask0 = overlap_masks_dict["mask0"].float()
        mask1 = overlap_masks_dict["mask1"].float()

        w0 = mask0.mean(dim=1) * amp0
        w1 = mask1.mean(dim=1) * amp1
        lam = w0 / (w0 + w1)

        X_mix = mask0 * X * s0[:, None] + mask1 * X[perm] * s1[:, None]
        Y_mix = lam[:, None] * Y + (1 - lam[:, None]) * Y[perm]

        # Unsqueeze the input back to the original shape if it was squeezed
        if unsqueeze:
            X_mix = X_mix.unsqueeze(1)

        return X_mix, Y_mix

def sample_overlap_speaker_masks(
    bs: int,
    L: int,
    concat_beta:Beta,
    overlap_beta:Beta,
    device: torch.device = None,
):
    """
    Returns:
        mask0: [bs, L]
        mask1: [bs, L]

    Timeline:
        speaker 0 active on [0, t + h]
        speaker 1 active on [t - h, 1]

    overlap length = 2h
    """

    device = device or "cpu"

    # 1. Sample boundary / pure non-overlap split.
    # t close to 0.5 gives balanced speakers.
    # t close to 0 or 1 gives one dominant speaker.
    t = concat_beta.sample((bs,)).to(device)

    # 2. Max possible half-overlap.
    h_max = torch.minimum(t, 1.0 - t)  # [bs]

    # 3. Sample relative overlap amount in [0, 1].
    h_frac = overlap_beta.sample((bs,)).to(device)  # [bs]

    # 4. Half-overlap size.
    h = h_frac * h_max  # [bs]

    # 5. Speaker regions.
    end0 = t + h       # speaker 0 active until this point
    start1 = t - h     # speaker 1 active from this point

    # 6. Build masks.
    pos = (torch.arange(L, device=device) + 0.5) / L  # [L], frame centers

    mask0 = pos[None, :] < end0[:, None]
    mask1 = pos[None, :] >= start1[:, None]

    return {
        "mask0": mask0,
        "mask1": mask1,
        "t": t,
        "h": h,
        "overlap": 2.0 * h,
        "end0": end0,
        "start1": start1,
    }
