
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.linalg import toeplitz



class DoublyBlockToeplitzFromKernel:

  def __init__(self, input_size, kernel,  
               padding=None, mode=None, with_padding=True, return_sparse=False):
    
    if padding is None and mode is None:
      raise ValueError("'padding' or 'mode' should be defined.")
    if mode is not None and mode not in ('valid', 'full', 'same'):
      raise ValueError("Acceptable mode flags are 'valid',"
                        " 'same', or 'full'.")
    if mode is not None and padding is not None:
      raise ValueError("Only one parameters bewteen 'padding' and 'mode' "
                        "should be set.")
    if kernel.ndim != 4:
      raise ValueError("the kernel should have 4 dimension.")
    if kernel.shape[2] != kernel.shape[3]:
      raise ValueError("The last 2 dimensions of the kernel should be equal.")

    self.image_size = input_size
    self.channels_out = kernel.shape[0]
    self.channels_in = kernel.shape[1]
    self.kernel_size = kernel.shape[2]
    self.kernel = kernel

    self.mode = mode
    if mode is not None:
      if kernel.shape[0] % 2 == 2:
        raise ValueError("If mode is set, kernel size should be odd.")
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

    self.with_padding = with_padding
    self.return_sparse = return_sparse

    self.all_blocks = []

  def padding_as_linear_op(self):
      e = np.eye(self.image_size**2)
      z1 = np.zeros((self.image_size**2, self.image_size+(2*self.padding+1)))
      z2 = np.zeros((self.image_size**2, 2*self.padding))
      a = []
      for i, split in enumerate(np.array_split(e, self.image_size, axis=1)):
        if i == 0 and self.padding >= 1:
          a.extend([z1] * self.padding)
        a.extend((split, z2))
        if i == self.image_size-1 and self.padding >= 1:
          a.pop()
          a.extend([z1] * self.padding)
      pad = np.hstack(a)
      return pad

  def _roll_zero(self, blocks, blocks_concat, shift, ix):
    if shift >= len(blocks):
      return np.zeros_like(np.vstack(blocks))
    if shift == 0:
      return np.vstack(blocks[:len(blocks)-shift])  
    start = ix - (blocks[0].shape[0] * shift)
    end = blocks_concat.shape[0] - (blocks[0].shape[0] * shift) 
    ret = blocks_concat[start:end]
    return ret

  def _create_toeplitz(self, kernel, padding):
    kernel = list(kernel)
    columns = []
    for i, j in list(zip(range(padding+1), range(padding, -1, -1))):
      c = np.array([0]*i + kernel + [0]*j).reshape(-1, 1)
      columns.append(c)
    return np.hstack(columns)

  def _generate_conv2d(self, kernel):

    # update image size
    image_size = self.image_size + 2*self.padding
    
    # shape of doubly block Toeplitz
    m = image_size**2
    n = self.output_size**2

    # created blocks of doubly block toeplitz
    # convert lines of padded kernel into toeplitz matrix
    cum_size = 0
    self.blocks = []
    padded_kernel = image_size - self.kernel_size
    for kernel_line in kernel:
      t = self._create_toeplitz(kernel_line, padded_kernel)
      tx, ty = t.shape
      self.blocks.append(t)
      cum_size += tx

    # add empty blocks
    z = np.zeros_like(t)
    while cum_size <= m:
      self.blocks.append(z)
      cum_size += tx

    n_columns = int(np.ceil(n / ty))
    z = np.zeros((self.blocks[0].shape[0] * n_columns, self.blocks[0].shape[1]))
    blocks_concat = np.vstack([z] + self.blocks[:len(self.blocks)])
    lenght_zeros = self.blocks[0].shape[0] * n_columns
      
    # define the doubly block toeplitz by stacking vertically all blocks
    self.columns_blocks = []
    for i in range(n_columns):
      col = self._roll_zero(self.blocks, blocks_concat, i, lenght_zeros)
      self.columns_blocks.append(col)
    op = np.hstack(self.columns_blocks)[:m, :n]

    self.all_blocks.extend(self.blocks)
    
    if self.with_padding:
      pad = self.padding_as_linear_op()
      op = csr_matrix(pad) @ csr_matrix(op)
      if not self.return_sparse:
        op = op.todense()
      return op
    return op


  def generate(self):
    ops = []
    for i in range(self.channels_out):
      self.ops_channels_in = []
      for j in range(self.channels_in):
        op = self._generate_conv2d(self.kernel[i][j])
        self.ops_channels_in.append(op)
      if self.return_sparse:
        ops.append(sp.sparse.vstack(self.ops_channels_in))
      else:
        ops.append(np.vstack(self.ops_channels_in))
    if self.return_sparse:
      return sp.sparse.hstack(ops)
    op = np.hstack(ops)
    return op




class BlockMatrix:
  """Base class for generating Block Matrices"""

  def __init__(self, block_size, n_blocks, block_initializer=None):
    """
      Args:
        block_size: size of the block 
        n_blocks: number of block to use
        block_initializer: function to initialize the blocks

      Returns:
        An object to generate a Block Matrix
    """
    self.block_size = block_size
    self.n_blocks = n_blocks
    self.block_initializer = block_initializer
    if self.block_initializer is None:
      # use default initializer
      self.block_initializer = self._default_block_initializer

  def _default_block_initializer(self, size, k):
    return np.random.normal(0, 1, size=size)
        
  def generate_block(self, block_size=None):
    """Function to generate a square matrix use as block of the global matrix.
      Args:
        block_size: size of the block
      Returns:
        An numpy array of size (block_size, block_size)
    """
    raise NotImplementedError('Must be implemented in derived classes')

  def generate(self):
    """Function to generate the block matrix. 
      Returns:
        An numpy array of size (block_size * n_blocks, block_size * n_blocks)
    """
    raise NotImplementedError('Must be implemented in derived classes')



class BlockToeplitz(BlockMatrix):
  """Generate a Block Toeplitz Matrix.
    Similar to a Toeplitz Matrix but instead of values the matrix is composed 
    of blocks.
  """
  def __init__(self, block_size, n_blocks, block_initializer=None,
               symmetric=False):
    super(BlockToeplitz, self).__init__(
        block_size, n_blocks, block_initializer=block_initializer)
    self.symmetric = symmetric
  
  def generate_block(self, k=None):
    block_shape = (self.block_size, self.block_size)
    block = self.block_initializer(size=block_shape, k=k)
    return block

  def generate(self):
    col_blocks = []
    for k in range(0, self.n_blocks):
      col_blocks.append(self.generate_block(k))
    if self.symmetric:
      self.blocks = col_blocks + col_blocks[1:][::-1]
    else:
      line_blocks = []
      for k in range(1, self.n_blocks)[::-1]:
        line_blocks.append(self.generate_block(k))
      self.blocks = col_blocks + line_blocks
    columns = []
    for i in range(self.n_blocks):
      columns.append(np.vstack(np.roll(self.blocks, i, axis=0)[:self.n_blocks]))
    return np.hstack(columns)


class ToeplitzBlockToeplitz(BlockToeplitz):
  """Generate a Doubly Block Toeplitz (DBT) or Toeplitz Block Toeplitz.
    Similar to a Toeplitz Matrix but instead of values the matrix is composed 
    of block with Toeplitz structure.
  """
  def __init__(self, block_size, n_blocks, block_initializer=None):
    super(ToeplitzBlockToeplitz, self).__init__(
        block_size, n_blocks, block_initializer=block_initializer)
  
  def generate_block(self, k=None):
    from scipy.linalg import toeplitz
    r = self.block_initializer(size=(self.block_size, ), k=k)
    c = self.block_initializer(size=(self.block_size, ), k=k)
    block = toeplitz(r, c)
    return block


class CirculantBlockToeplitz(BlockToeplitz):
  """Generate a Circulant Block Toeplitz.
    Similar to a Toeplitz Matrix but instead of values the matrix is composed 
    of block with Circulant structure.
  """
  def __init__(self, block_size, n_blocks, block_initializer=None):
    super(CirculantBlockToeplitz, self).__init__(
        block_size, n_blocks, block_initializer=block_initializer)
  
  def generate_block(self, k=None):
    from scipy.linalg import circulant
    c = self.block_initializer(size=(self.block_size, ), k=k)
    block = circulant(c)
    return block





class BlockCirculant(BlockMatrix):
  """Generate a Block Circulant.
    Similar to a Circulant Matrix but instead of values the matrix is composed 
    of blocks.
  """
  def __init__(self, block_size, n_blocks, block_initializer=None):
    super(BlockCirculant, self).__init__(
        block_size, n_blocks, block_initializer=block_initializer)

  def generate_block(self, k=None):
    block_shape = (self.block_size, self.block_size)
    block = self.block_initializer(size=block_shape, k=k)
    return block

  def generate(self):
    self.blocks = []
    for k in range(self.n_blocks):
      self.blocks.append(self.generate_block(k))
    columns = []
    for i in range(self.n_blocks):
      columns.append(np.vstack(np.roll(self.blocks, i, axis=0)))
    return np.hstack(columns)
    

class ToeplitzBlockCirculant(BlockCirculant):
  """Generate a Toeplitz Block Circulant.
    Similar to a Circulant Matrix but instead of values the matrix is composed 
    of block with Toeplitz structure.
  """
  def __init__(self, block_size, n_blocks, block_initializer=None):
    super(ToeplitzBlockCirculant, self).__init__(
        block_size, n_blocks, block_initializer=block_initializer)
  
  def generate_block(self, k=None):
    from scipy.linalg import toeplitz
    r = self.block_initializer(size=(self.block_size, ), k=k)
    c = self.block_initializer(size=(self.block_size, ), k=k)
    block = toeplitz(r, c)
    return block


class CirculantBlockCirculant(BlockCirculant):
  """Generate a Toeplitz Block Circulant.
    Similar to a Circulant Matrix but instead of values the matrix is composed 
    of block with Toeplitz structure.
  """
  def __init__(self, block_size, n_blocks, block_initializer=None):
    super(CirculantBlockCirculant, self).__init__(
        block_size, n_blocks, block_initializer=block_initializer)
  
  def generate_block(self, k=None):
    from scipy.linalg import circulant
    c = self.block_initializer(size=(self.block_size, ), k=k)
    block = circulant(c)
    return block
