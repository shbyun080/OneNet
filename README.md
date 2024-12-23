# OneNet: A Channel-Wise 1D Convolutional U-Net

This repository serves as the official codebase for the paper:
> **OneNet: A Channel-Wise 1D Convolutional U-Net**
>
> [Sanghyun Byun](https://shbyun080.github.io/), [Kayvan Shah](https://github.com/KayvanShah1), [Ayushi Gang](https://github.com/ayu-04), [Christopher Apton](https://github.com/chrisapton), Jacob Song and Woo Seong Chung
>
> [arXiv:2411.09838](https://arxiv.org/abs/2411.09838)

### About
Channel-Wise 1D convolutional encoder that retains U-Net’s accuracy while enhancing its suitability for edge applications by halving the model parameters.

## 🚧 Roadmap
11/7/2024: Project Repo Initialized

11/22/2024: Initial Model Code Uploaded

## ⚙️ Installation
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

## 📦 Dataset
- MSD Brain/Heart/Lung
- Oxford PET
- Pascal VOC
- COCO

## 📜 Citation
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

## 🪪 LICENSE
This project is licensed under the `CC-BY-4.0` License. See the [LICENSE](LICENSE) file for details.

#### Disclaimer
<sub>
The content and code provided in this repository are for educational and demonstrative purposes only. The project may contain experimental features, and the code might not be optimized for production environments. The authors and contributors are not liable for any misuse, damages, or risks associated with the use of this code. Users are advised to review, test, and modify the code to suit their specific use cases and requirements. By using any part of this project, you agree to these terms.
</sub>
