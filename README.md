This repository contains the code the paper the paper:

### On Lipschitz Regularization of Convolutional Layers using Toeplitz Matrix Theory

[https://arxiv.org/abs/2006.08391](https://arxiv.org/abs/2006.08391)


Code to use the bound:

```python
import torch
import numpy as np
from lipschitz_bound import LipschitzBound

kernel_numpy = np.random.randn(1, 3, 3, 3)
kernel_torch = torch.FloatTensor(kernel_numpy)
lb = LipschitzBound(kernel_numpy.shape, padding=1, sample=50, backend='numpy')
sv_bound = lb.compute(kernel_numpy)
print(f'LipBound computed with numpy: {sv_bound:.3f}')

lb = LipschitzBound(kernel_torch.shape, padding=1, sample=50, backend='torch', cuda=False)
sv_bound = lb.compute(kernel_torch)
print(f'LipBound computed with torch: {sv_bound:.3f}')
```

Reference:
```
@article{araujo2021lipschitz,
  title={On Lipschitz Regularization of Convolutional Layers using Toeplitz Matrix Theory},
  author={Araujo, Alexandre and Negrevergne, Benjamin and Chevaleyre, Yann and Atif, Jamal},
  journal={Thirty-Fifth AAAI Conference on Artificial Intelligence},
  url={https://arxiv.org/abs/2006.08391},
  year={2021}
}
```
