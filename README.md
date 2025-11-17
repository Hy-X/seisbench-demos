# SeisBench Tutorial

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![SeisBench](https://img.shields.io/badge/SeisBench-Tutorial-green)](https://seisbench.readthedocs.io/)
[![License](https://img.shields.io/badge/License-GPL-yellow)](https://opensource.org/licenses/GPL-3-0)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Stars](https://img.shields.io/github/stars/Hy-X/seisbench-tutorial?style=social)](https://github.com/Hy-X/seisbench-tutorial/stargazers)
[![Forks](https://img.shields.io/github/forks/Hy-X/seisbench-tutorial?style=social)](https://github.com/Hy-X/seisbench-tutorial/network/members)
[![Issues](https://img.shields.io/github/issues/Hy-X/seisbench-tutorial)](https://github.com/Hy-X/seisbench-tutorial/issues)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Hy-X/seisbench-tutorial/issues)
[![Awesome](https://img.shields.io/badge/Awesome-Yes-brightgreen.svg)](#)
[![Build Passing](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](#)



This repository provides a hands-on tutorial for **SeisBench**, a Python library for working with seismic datasets and machine learning models for seismology. It includes example scripts and Jupyter notebooks to get you started quickly.

## Repository Structure

```
seisbench-tutorial/
├── README.md
├── requirements.txt
├── data/
│   └── 01_basic_dataset.hdf5
├── notebooks/
│   └── 01_basic_usage.ipynb
└── scripts/
    └── 01_basic_loader.py
```

- **notebooks/**: Jupyter notebooks demonstrating SeisBench usage interactively.  
- **scripts/**: Python scripts for running SeisBench workflows programmatically.  
- **requirements.txt**: Python dependencies needed to run the tutorials.

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/Hy-X/seisbench-tutorial.git
cd seisbench-tutorial
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
# Activate:
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> Ensure you have `seisbench` and optionally `torch` installed. If you are using GPU, install the corresponding CUDA version for PyTorch.

## Usage

### Jupyter Notebook

Open the notebook to explore SeisBench interactively:

```bash
jupyter notebook notebooks/01_basic_usage.ipynb
```

### Python Scripts

Run the example script:

```bash
python scripts/01_basic_loader.py
```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for additional tutorials, datasets, or models.

## License

This project is licensed under the GPL License.

