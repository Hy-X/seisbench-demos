# SeisBench Tutorial

This repository provides a hands-on tutorial for **SeisBench**, a Python library for working with seismic datasets and machine learning models for seismology. It includes example scripts and Jupyter notebooks to get you started quickly.

## Repository Structure

```
seisbench-tutorial/
├── README.md
├── requirements.txt
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
git clone https://github.com/yourusername/seisbench-tutorial.git
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

