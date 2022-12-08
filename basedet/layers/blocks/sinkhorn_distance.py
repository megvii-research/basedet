#!/usr/bin/env python3

import megengine.functional as F
import megengine.module as M


class SinkhornDistance(M.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps: regularization coefficient
        max_iter: maximum number of Sinkhorn iterations
    """

    def __init__(self, eps: float = 1e-3, max_iter: int = 100):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, mu, nu, cost):
        u = F.ones_like(mu)
        v = F.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v += self.eps * (F.log(nu + 1e-8) - F.logsumexp(
                F.transpose(self.M(cost, u, v, self.eps), (1, 0)), axis=-1
            ))
            u += self.eps * (F.log(mu + 1e-8) - F.logsumexp(
                self.M(cost, u, v, self.eps), axis=-1
            ))

        # Transport plan pi = diag(a) * K * diag(b)
        pi = F.exp(self.M(cost, u, v, self.eps)).detach()
        distance = (pi * cost).sum()  # Sinkhorn distance
        return distance, pi

    @classmethod
    def M(cls, cost, u, v, eps):
        """
        Modified cost for logarithmic updates.

        .. math::
            \\displaymath M_{ij} = (-cost_{ij} + u_i + v_j) / \\epsilon
        """
        return (-cost + F.expand_dims(u, axis=-1) + F.expand_dims(v, axis=-2)) / eps
