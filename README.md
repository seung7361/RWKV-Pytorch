# RWKV Implementation in Pytorch

This repository contains an implementation of Receptance, Weight, Key, Value (RWKV) model in PyTorch. RWKV is a unique technique designed to enhance the efficiency and effectiveness of machine learning models.

## Requirements

Ensure that you have installed the latest versions of:

- Python 3.6 or later
- PyTorch 1.8.1 or later
- CUDA compatible with your PyTorch and Python versions (if you plan on using a GPU)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/seung7361/RWKV-Pytorch
```

2. Navigate to the cloned directory:
```bash
cd RWKV-Pytorch
```

3. (Optional) Create a new Python virtual environment. This step helps to keep your workspace clean and allows you to manage dependencies efficiently.

If you are using venv:
```bash
python3 -m venv env
source env/bin/activate
```

If you are using conda:
```bash
conda create --name rwkv_env python=3.8
conda activate rwkv_env
```

4. Install the required packages:
```
./requirements.sh
```

## Usage

1. To use the RWKV model, follow the given steps:

Import the RWKV model from the module:
```python
from rwkv_model import RWKVModel
```

2. Initialize the model with your parameters:
```python
vocab_size = 10000
n_layers = 6
hidden_size = 512

model = RWKVModel(vocab_size=vocab_size, n_layers=n_layers, hidden_size=hidden_size).cuda()
```

In this example, we have used vocab_size=10000, n_layers=6, and hidden_size=512, which you can modify as per your requirements.

Note: The .cuda() function is used to move the model to GPU for faster computations. If you don't have a GPU, you can remove .cuda() to run the model on CPU.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions, please feel free to reach out to me by creating an issue with my repository or email me to these email address:

- seung7361@naver.com
- seung7361@gmail.com
- seung7361@hanyang.ac.kr

