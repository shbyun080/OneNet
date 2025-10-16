# OneNet: A Channel-Wise 1D Convolutional U-Net

This repository serves as the official codebase for the paper:
> **OneNet: A Channel-Wise 1D Convolutional U-Net**
>
> [Sanghyun Byun](https://shbyun080.github.io/), [Kayvan Shah](https://github.com/KayvanShah1), [Ayushi Gang](https://github.com/ayu-04), [Christopher Apton](https://github.com/chrisapton), Jacob Song and Woo Seong Chung
>
> [arXiv:2411.09838](https://arxiv.org/abs/2411.09838)

### Video Presentation
The work was presented at the `2025 IEEE International Conference on Image Processing, Theory and Tools` (IPTA 2025)

[Click here to watch the presentation](https://www.youtube.com/watch?v=ufhQdo-_Hh4) 

## About
OneNet is a streamlined alternative to traditional U-Net architectures, specifically designed to address the significant computational demands of multi-resolution convolutional models that often limit deployment on edge devices.

### Core Architectural Innovation

OneNet replaces the standard 2D convolutions typically used in the U-Net backbone with a novel convolution block that relies solely on **1D convolutions**.

This efficiency is achieved by leveraging specialized pixel-repositioning techniques inspired by image super-resolution tasks:

1.  **Downscaling:** The encoder block utilizes **pixel-unshuffle downscaling (PUD)** instead of max pooling. This operation transfers spatial knowledge to the channel axis, allowing subsequent **channel-wise 1D convolutions** to effectively capture spatial and channel relationships while reducing computational complexity.
2.  **Upscaling:** The decoder block utilizes **pixel-shuffle upscaling (PSU)**, which moves information from the channel dimension back to the spatial dimension.

This channel-wise 1D approach avoids the computational overhead associated with traditional 2D convolutions, making the model suitable for lightweight deployment.

### Quantified Efficiency Gains

OneNet provides substantial reductions in both parameters and computational load, demonstrating its feasibility for resource-constrained environments. (Metrics shown are for the 4-layer variant, using a sample tensor of size (1, 3, 256, 256) for comparison against baselines):

| Replacement Type | Model Size Reduction | Model Size (`OneNet_{ed,4}`) | Parameters (`OneNet_{ed,4}`) | FLOPs (`OneNet_{ed,4}`) |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder-only** (`OneNet_{e,4}`) | Up to **47%** reduction in parameters | 65.42 MB | 16.39M | 78.42 GB |
| **Encoder + Decoder (Fully 1D)** | Up to **71%** reduction in size | **36.30 MB** (below 40MB) | **9.08M** (69% reduction) | **22.92 GB** (78% reduction in FLOPs) |

## Performance Summary and Trade-Offs

OneNet is highly effective for tasks focused on local feature detection.

*   **High Accuracy Retention:** The architecture **preserves accuracy effectively** and performs **on par** with established baseline architectures for medical tumor detection datasets (e.g., MSD Heart, Brain, Lung). For instance, on the MSD Heart dataset, OneNet showed a slight edge in accuracy, achieving a 2% improvement over baselines, while simultaneously achieving a 47% reduction in parameters.
*   **Optimal Use Case:** OneNet excels in mask-generating tasks that are **local-feature-centric** or involve small segmentation mask counts.
*   **Trade-Offs:** The full 1D encoder-decoder structure (`OneNet_{ed,4}`) shows an expected drop in accuracy (11% to 15%) on general multi-class segmentation datasets (like PASCAL VOC and full Oxford Pet), highlighting a necessary trade-off for achieving significant resource reduction required for edge deployment.

## Installation
Environment (model has not been tested on other environments)
- Linux
- Python 3.12
- CUDA 12.1

Please set the environment with
```bash
export VENV_DIR=<YOUR-VENV>
export NAME=OneNet

python -m venv $VENV_DIR/$NAME
source $VENV_DIR/$NAME/bin/activate
```

For general use
```bash
pip install .
```

For development use, do an editable installation locally to avoid importing issues
```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

## Dataset
- MSD Brain/Heart/Lung
- Oxford PET
- Pascal VOC
- COCO

## Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex
@misc{onenet-2024,
  title={OneNet: A Channel-Wise 1D Convolutional U-Net},
  author={Sanghyun Byun and Kayvan Shah and Ayushi Gang and Christopher Apton and Jacob Song and Woo Seong Chung},
  archivePrefix={arXiv},
  eprint={2411.09838},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.09838}, 
  month={November},
  year={2024},
}
```

### Authors
1. [Sanghyun Byun](https://shbyun080.github.io/) | `MS in Computer Science @ USC`, `AI Partner @ LG Electronics`
3. [Kayvan Shah](https://github.com/KayvanShah1) | `MS in Applied Data Science @ USC`
2. [Ayushi Gang](https://github.com/ayu-04) | `MS in Computer Science @ USC`
4. [Christopher Apton](https://github.com/chrisapton) | `MS in Applied Data Science @ USC`
5. Jacob Song | `Principal Researcher @ LG Electronics`
6. Woo Seong Chung | `Principal Researcher @ LG Electronics`

### Acknowledgement
We thank `Professor Yan Liu` at the `University of Southern California` for guidance.

## LICENSE
This project is licensed under the `CC-BY-4.0` License. See the [LICENSE](LICENSE) file for details.

#### Disclaimer
<sub>
The content and code provided in this repository are for educational and demonstrative purposes only. The project may contain experimental features, and the code might not be optimized for production environments. The authors and contributors are not liable for any misuse, damages, or risks associated with the use of this code. Users are advised to review, test, and modify the code to suit their specific use cases and requirements. By using any part of this project, you agree to these terms.
</sub>
