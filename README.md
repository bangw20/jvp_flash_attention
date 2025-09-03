<div align="center">

# JVP Flash Attention

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

</div>


## Description

Flash Attention Triton kernel with support for second-order derivatives, such as Jacobian-Vector Products (JVPs) and Hessian-Vector Products (HVPs)

## Installation

Using `pip`, one can install `jvp_flash_attention` as follows.

```bash
# Install package
pip install jvp_flash_attention

# [OPTIONAL, for development] Install package and pre-commit hooks
pip install -e .
pre-commit install
```

## Usage

Once installed, one can use `jvp_flash_attention` in place of PyTorch's `scaled_dot_product_attention` as follows.

```python
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel
from jvp_flash_attention.jvp_attention import attention as jvp_attention

with sdpa_kernel(SDPBackend.MATH):
  # Regular attention
  # x = F.scaled_dot_product_attention(
  #     q,
  #     k,
  #     v,
  #     attn_mask=attn_mask,
  #     dropout_p=attn_dropout_p if self.training else 0.0,
  # )

  # Flash attention
  x = jvp_attention(
      q,
      k,
      v,
      # attn_mask=attn_mask,  # NOTE: Attention masking is not yet supported
  )
```

Contributions or enhancements are welcome!

## Tests

If you want to run the unit tests verifying the correctness of the JVP Flash Attention Triton kernel, run the following command(s).

```bash
python tests/test_jvp_attention.py --dtype {float16,bfloat16,float32}
```

In principle, the kernel should support ROCm systems as well, though it has not yet been tested on them. macOS is currently unsupported.

## Citing this work

If you use the code associated with this package or otherwise find this work useful, please use GitHub's `Cite this repository` feature or the BibTeX below.

```bibtex
@software{Morehead_JVP_Flash_Attention_2025,
author = {Morehead, Alex},
doi = {10.5281/zenodo.17050188},
license = {MIT},
month = sep,
title = {{JVP Flash Attention}},
url = {https://github.com/amorehead/jvp_flash_attention},
version = {0.0.1},
year = {2025}
}
```

## Acknowledgements

`jvp_flash_attention` builds upon the contributions and insights from the following sources:

- [flash-attention](https://github.com/Dao-AILab/flash-attention)
  - [JVP Triton kernel thread](https://github.com/Dao-AILab/flash-attention/issues/1672)
    - [benjamin-dinkelmann](https://gist.github.com/benjamin-dinkelmann)
    - *[Birch-san](https://github.com/Birch-san)*
    - [dabeschte](https://github.com/dabeschte)
    - [IsaacYQH](https://gist.github.com/IsaacYQH)
    - [KohakuBlueleaf](https://github.com/KohakuBlueleaf)
    - [leon](https://github.com/leon532)
    - [limsanky](https://github.com/limsanky)
    - [lucidrains](https://github.com/lucidrains)
    - [Peterande](https://gist.github.com/Peterande)
    - *[Ryu1845](https://github.com/Ryu1845)*
    - [tridao](https://github.com/tridao)

We thank each and every contributor!
