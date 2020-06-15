import torch
import torch.nn as nn
import numpy as np
from scipy.signal import convolve2d

from block_matrices.ops import DoublyBlockToeplitzFromKernel



class CompareDoublyBlockToeplitzFromKernel:

  def __init__(self, x, kernel, 
               padding=None, mode=None):
    
    if x.ndim != 4:
      raise ValueError(
          "input should have shape [batch_size, channels, height, widht].")
    if x.shape[2] != x.shape[3]:
      raise NotImplementedError("Rectangular matrix or not implemented.")
    if padding is None and mode is None:
      raise ValueError("'padding' or 'mode' should be defined.")
    if mode is not None and mode not in ('valid', 'full', 'same'):
      raise ValueError("Acceptable mode flags are 'valid',"
                        " 'same', or 'full'.")
    if mode is not None and padding is not None:
      raise ValueError("Only one parameters bewteen 'padding' and 'mode' "
                        "should be set {} {}.".format(self.padding, self.mode))

    self.batch_size = x.shape[0]
    self.image_size = x.shape[2]
    self.x = x

    assert kernel.ndim == 4
    assert kernel.shape[2] == kernel.shape[3]
    self.kernel_size = kernel.shape[2]
    self.channels_out = kernel.shape[0]
    self.channels_in = kernel.shape[1]
    self.kernel = kernel

    self.mode = mode
    if mode is not None:
      if mode == "valid":
        self.padding = 0
      elif mode == "full":
        self.padding = self.kernel_size - 1
      elif mode == "same":
        self.padding = (self.kernel_size - 1) // 2
    else:
      self.padding = padding

    # relationship between input and output sizes
    # https://arxiv.org/abs/1603.07285
    self.output_size = \
      (self.image_size - self.kernel_size) + 2 * self.padding + 1


  def conv_pytorch(self):
    conv_layer = nn.Conv2d(
        self.channels_in, self.channels_out,
        (self.kernel_size, self.kernel_size), 
        stride=1, padding=self.padding, dilation=1, groups=1, 
        bias=False, padding_mode='zeros')
    kernel = self.kernel
    kernel = torch.FloatTensor(np.ascontiguousarray(kernel))
    conv_layer.weight.data = torch.FloatTensor(kernel)
    x = torch.FloatTensor(np.ascontiguousarray(self.x))
    conv = conv_layer(x).detach().numpy()
    return conv

  def conv_scipy(self):
    kernel = self.kernel[0][0]
    batch_conv = []
    for image in self.x:
      image = image.reshape(self.image_size, self.image_size)
      conv = convolve2d(image, kernel[::-1, ::-1], mode=self.mode)
      conv = conv.reshape(1, self.output_size, self.output_size)
      batch_conv.append(conv)
    return np.array(batch_conv)

  def conv_with_matrix(self):

    x = self.x.reshape(
      self.batch_size, self.channels_in * self.image_size**2)
    
    if self.mode:
      padding = None
    else:
      padding = self.padding

    op = DoublyBlockToeplitzFromKernel(
      self.image_size, self.kernel, padding, self.mode).generate() 
    conv = x @ op

    # reshape the result
    conv = np.array(conv)
    conv = conv.reshape(
      self.batch_size, self.channels_out, self.output_size, self.output_size)
    return conv

  def test(self):
    x1 = self.conv_pytorch()
    if self.mode is not None and \
       self.channels_in == 1 and self.channels_out == 1:
      x2 = self.conv_scipy()
      np.testing.assert_array_almost_equal(x1, x2, decimal=4)
    x3 = self.conv_with_matrix()
    np.testing.assert_array_almost_equal(x1, x3, decimal=4)


def test_conv_with_matrix_op_with_padding():
  for kernel_size in [1, 3, 5, 7]:
    for image_size in [28, 32, 35]:
      for pad in [0, 1, 2, 3, 4]:
        for channels_in in [1, 2, 3, 4]:
          for channels_out in [1, 2, 3, 4]:
            x = np.random.randn(8, channels_in, image_size, image_size)
            kernel = np.random.randn(
              channels_out, channels_in, kernel_size, kernel_size)
            test = CompareDoublyBlockToeplitzFromKernel(
              x, kernel, padding=pad)
            test.test()
  print('success')

def test_conv_with_matrix_op_with_mode():
  for kernel_size in [3, 5, 7]:
    for image_size in [28, 32, 35]:
      for mode in ['same', 'full', 'valid']:
        for channels_in in [1, 3]:
          for channels_out in [1, 3, 9]:
            x = np.random.randn(8, channels_in, image_size, image_size)
            kernel = np.random.randn(
              channels_out, channels_in, kernel_size, kernel_size)
            test = CompareDoublyBlockToeplitzFromKernel(
              x, kernel, mode=mode)
            test.test()
  print('success')

if __name__ == '__main__':
  test_conv_with_matrix_op_with_padding()
  test_conv_with_matrix_op_with_mode()



