"""Candidatos temporales experimentales basados en KronLSTM.

Adaptación de NX-AI/KronLSTM, commit 3d800c9a153a88ee9624fe6acf7f39a58017a17e.
Referencia: Hartl et al., Kronecker Delta Rule for Long Context Linear Attention,
arXiv:2610.26834v1. La dimensión compacta y la entrada financiera son propias.

Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
Copyright (c) 2026 NXAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .precomputed import kronlstm_scan_precomputed
from .price_encoder import CompactPriceKronLSTM
from .rule import kronlstm_scan

__all__ = ["CompactPriceKronLSTM", "kronlstm_scan", "kronlstm_scan_precomputed"]
