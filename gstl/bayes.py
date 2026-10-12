"""Lapisan Bayesian (Bayes-by-backprop). Port dari UString `src/BayesModels.py` (MIT, Bao et al. 2020).

Perbedaan dari asli (perilaku statistik sama):
  * epsilon disampel langsung di device parameter (`torch.randn_like`), bukan di CPU lalu dipindah;
  * sigma = softplus(rho) (identik dengan log1p(exp(rho)), lebih stabil untuk rho besar);
  * log-prior campuran skala memakai logsumexp (identik secara matematis, tanpa underflow ke log 0).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian:
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho

    @property
    def sigma(self):
        return F.softplus(self.rho)

    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.rho)

    def log_prob(self, x):
        sigma = self.sigma
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(sigma) - ((x - self.mu) ** 2) / (2 * sigma ** 2)).sum()


class ScaleMixtureGaussian:
    def __init__(self, pi, sigma1, sigma2):
        self.pi = float(pi)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)

    def _log_normal(self, x, sigma):
        return -0.5 * math.log(2 * math.pi) - math.log(sigma) - x ** 2 / (2 * sigma ** 2)

    def log_prob(self, x):
        a = self._log_normal(x, self.sigma1) + math.log(self.pi)
        b = self._log_normal(x, self.sigma2) + math.log(1 - self.pi)
        return torch.logsumexp(torch.stack([a, b]), dim=0).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, pi=0.5, sigma_1=None, sigma_2=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if sigma_1 is None or sigma_2 is None:
            sigma_1, sigma_2 = math.exp(-0), math.exp(-6)
        # Urutan pembuatan dan pemanggilan uniform_ sama dengan asli (penting untuk RNG; lihat models.py).
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.empty(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        self.weight_prior = ScaleMixtureGaussian(pi, sigma_1, sigma_2)
        self.bias_prior = ScaleMixtureGaussian(pi, sigma_1, sigma_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            # Sama dengan asli: dalam mode eval suku KL = 0 (loss uji tanpa KL).
            self.log_prior, self.log_variational_posterior = 0, 0
        return F.linear(x, weight, bias)
