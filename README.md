# Infini-Attention

[![GitHub Stars](https://img.shields.io/github/stars/jlamprou/Infini-Attention?style=social)](https://github.com/jlamprou/Infini-Attention/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/jlamprou/Infini-Attention)](https://github.com/jlamprou/Infini-Attention/issues)
[![License](https://img.shields.io/github/license/jlamprou/Infini-Attention)](https://github.com/jlamprou/Infini-Attention/blob/main/LICENSE)


**Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention Pytorch Implementation** ([Paper](https://arxiv.org/abs/2404.07143))

This repository provides a PyTorch implementation of the Infi-Attention mechanism, following the paper step-by-step. The code is written using the HuggingFace template for seamless integration with their ecosystem.

## Features

- PyTorch implementation of Infi-Attention
- Follows the paper's methodology closely
- Utilizes the HuggingFace template for easy integration

## Current Limitations and Areas for Improvement

1. **Segment-wise Attention**: The paper does not provide clear guidance on when the segmentation should occur during the model training process. Two potential theories are being explored:
  - Theory 1: Segment the input within the attention class and perform in-class operations using a loop for each segment. However, this approach does not result in significant memory savings compared to classic SDPA.
  - Theory 2: Segment the input during the training loop and feed each segment to the model with gradient accumulation steps equal to Sequence Length / Segment Length.

2. **Delta Use**: There is currently a shape mismatch when utilizing the delta values. This issue will be addressed in the near future, as the current focus is on resolving the segmentation aspect.

## Contributing

Contributions to this project are welcome! If you have any ideas, suggestions, or would like to discuss the implementation, please feel free to open an issue on the [GitHub repository](https://github.com/jlamprou/Infi-Attention). Your feedback and contributions can help improve the Infi-Attention implementation.

## License

This project is licensed under the MIT License.

---

🌟 Give this project a star on [GitHub](https://github.com/jlamprou/Infini-Attention) to show your support!
