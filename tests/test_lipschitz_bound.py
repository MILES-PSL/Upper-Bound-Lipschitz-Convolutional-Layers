from datetime import datetime
import numpy as np
import scipy as sp
from scipy.sparse import linalg
from scipy.sparse.linalg import svds, eigs

import torch
import torchvision.models as models

from block_matrices.ops import DoublyBlockToeplitzFromKernel
from lipschitz_bound import LipschitzBound


def generate_kernel(kshape, kernel_type):
  cout, cin, ksize, ksize = kshape
  kernel = np.random.randn(cout, cin, ksize, ksize)
  if kernel_type == 'torch':
    kernel = torch.FloatTensor(np.float32(kernel))
  return kernel


def compute(image_size, kshape, padding, kernel_type, version):
  kernel = generate_kernel(kshape, kernel_type) 
  A = DoublyBlockToeplitzFromKernel(
    image_size, kernel, padding=padding, return_sparse=True).generate()
  # compute sv max
  sv_max = svds(A, k=1, which='LM', return_singular_vectors=False)[0]
  # compute bound
  lb = LipschitzBound(kernel.shape, padding, sample=200, backend=kernel_type,
                      cuda=False, version=version)
  sv_bound = lb.compute(kernel)
  return sv_max <= sv_bound


def loop(kernel_type):

  cin, cout = 1, 1
  for version in ['v1', 'v2']:
    for image_size in [30]:
      for ksize in [3, 5, 7, 9, 11]:
        for padding in [0, 1, 2, 3, 4]:
          kshape = [1, 1, ksize, ksize]
          assert compute(image_size, kshape, padding, kernel_type, version)

  ksize = 3
  padding = 1
  for version in ['v1', 'v2']:
    for image_size in [30]:
      for cout in [6, 9, 12]:
        for cin in [6, 9, 12]:
          kshape = [cout, cin, ksize, ksize]
          assert compute(image_size, kshape, padding, kernel_type, version)

def test_bound_numpy():
  loop('numpy')

def test_bound_torch():
  loop('torch')



