from collections import defaultdict
from itertools import product
from functools import reduce
import numpy as np
import scipy as sp
from scipy.sparse import linalg
import logging

try:
  import torch
  import torch.nn as nn
except:
  pass


class LipschitzBound:

  def __init__(self, kernel_shape, padding, sample=50, backend='torch',
               cuda=True):

    self.kernel_shape = kernel_shape
    self.padding = padding
    self.sample = sample
    self.backend = backend
    self.cuda = cuda

    cout, cin, ksize, _ = kernel_shape

    # verify the kernel is square
    if not kernel_shape[-1] == kernel_shape[-2]:
      raise ValueError("The last 2 dim of the kernel must be equal.")
    # verify if all kernel have odd shape
    if not kernel_shape[-1] % 2 == 1:
      raise ValueError("The dimension of the kernel must be odd.")

    # define search space
    x = np.linspace(0, 2*np.pi, num=self.sample)
    w = np.array(list(product(x, x)))
    self.w0 = w[:, 0].reshape(-1, 1)
    self.w1 = w[:, 1].reshape(-1, 1)

    # convert search space to torch tensor
    if self.backend == 'torch':
      self.w0 = torch.FloatTensor(np.float32(self.w0))
      self.w1 = torch.FloatTensor(np.float32(self.w1))
      if self.cuda:
        self.w0 = self.w0.cuda()
        self.w1 = self.w1.cuda()

   # create samples
    if self.backend == 'numpy':
      p_index = np.arange(-ksize + 1., 1.) + padding
      H0 = 1j * np.tile(p_index, ksize).reshape(ksize, ksize).T.reshape(-1)
      H1 = 1j * np.tile(p_index, ksize)
      self.samples = np.exp(self.w0 * H0 + self.w1 * H1).T

    elif self.backend == 'torch':
      p_index = torch.arange(-ksize + 1.0, 1.0) + padding
      H0 = p_index.repeat(ksize).reshape(ksize, ksize).T.reshape(-1)
      H1 = p_index.repeat(ksize)
      if self.cuda:
        H0 = H0.cuda()
        H1 = H1.cuda()
      real = torch.cos(self.w0 * H0 + self.w1 * H1).T
      imag = torch.sin(self.w0 * H0 + self.w1 * H1).T
      self.samples = (real, imag)

  def compute(self, *args, **kwargs):
    if self.backend == 'torch':
      return self._compute_from_torch(*args, **kwargs)
    return self._compute_from_numpy(*args, **kwargs)

  def _compute_from_numpy(self, kernel):
    """Compute the LipGrid Algorithm."""
    pad = self.padding
    cout, cin, ksize, _ = kernel.shape
    if cout > cin:
      kernel = np.transpose(kernel, axes=[1, 0, 2, 3])
      cout, cin = cin, cout
    ker = kernel.reshape(cout, cin, -1)[..., np.newaxis]
    poly = (ker * self.samples).sum(axis=2)
    poly = np.square(np.abs(poly)).sum(axis=1)
    sv_max = np.sqrt(poly.max(axis=-1).sum())
    return sv_max

  def _compute_from_torch_naive(self, kernel):
    """Compute the LipGrid Algo with Torch"""
    pad = self.padding
    cout, cin, ksize, _ = kernel.shape
    if cout > cin:
      kernel = torch.transpose(kernel, 0, 1)
      cout, cin = cin, cout

    ker = kernel.view(cout, cin, -1, 1)
    real, imag = self.samples

    poly_real = torch.mul(ker, real).sum(axis=2)
    poly_imag = torch.mul(ker, imag).sum(axis=2)

    poly = torch.mul(poly_real, poly_real) + \
        torch.mul(poly_imag, poly_imag)
    poly = poly.sum(axis=1)
    sv_max = torch.sqrt(poly.max(axis=-1)[0].sum())
    return sv_max

  def _compute_from_torch(self, kernel):
    """Compute the LipGrid Algo with Torch"""
    pad = self.padding
    cout, cin, ksize, _ = kernel.shape
    if cout > cin:
      kernel = torch.transpose(kernel, 0, 1)
      cout, cin = cin, cout

    real, imag = self.samples
    if not kernel.is_contiguous:
      kernel = kernel.contiguous()
    if not real.is_contiguous():
      real = real.contiguous()
    if not imag.is_contiguous():
      imag = imag.contiguous()
    ker = kernel.reshape(cout*cin, -1)
    poly_real = torch.matmul(ker, real).view(cout, cin, -1)
    poly_imag = torch.matmul(ker, imag).view(cout, cin, -1)

    poly1 = torch.einsum('ijk,ijk->ik', poly_real, poly_real)
    poly2 = torch.einsum('ijk,ijk->ik', poly_imag, poly_imag)
    poly = poly1 + poly2

    sv_max = torch.sqrt(poly.max(axis=-1)[0].sum())
    return sv_max


