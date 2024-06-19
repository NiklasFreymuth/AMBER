from typing import Tuple

import torch
from torch import Tensor


class TorchRunningMeanStd:
    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        """
        Creates a running mean and std object

        Args:
            epsilon: small value to avoid division by zero
            shape: shape of the data
            dtype: data type of the data
        """
        self.mean = torch.zeros(*shape, dtype=dtype, device=device)
        self.var = torch.ones(*shape, dtype=dtype, device=device)
        self.count = epsilon
        self._device = device

    def to(self, device: str):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)

    def update(self, arr: Tensor) -> None:
        """
        Updates the running mean and std with a new batch of data

        Args:
            arr: new batch of data as a tensor
        """
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0)
        batch_var = torch.nan_to_num(batch_var, nan=0.0)
        batch_count = arr.shape[0]

        batch_mean = batch_mean.to(self._device)
        batch_var = batch_var.to(self._device)

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: Tensor, batch_var: Tensor, batch_count: int) -> None:
        """
        Internal method to update the running mean and std from the batch moments
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def __repr__(self):
        return f"TorchRunningMeanStd(mean={self.mean}, var={self.var}, count={self.count})"
