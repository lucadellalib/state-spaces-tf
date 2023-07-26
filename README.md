# State Spaces - TensorFlow

[![Python version: 3.6 | 3.7 | 3.8 | 3.9 | 3.10](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/lucadellalib/state-spaces-tf/blob/master/LICENSE)

TensorFlow implementation of state space models based on [Long Range Language Modeling via Gated State Spaces](https://arxiv.org/abs/2206.13947v3).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Python 3.6 or later](https://www.python.org).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a terminal and run:

```bash
pip install -r requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

```python
import tensorflow as tf

from dss import DSS


batch_size = 4
seq_length = 2048
input_size = 64
model = DSS(input_size)
input = tf.random.normal([batch_size, seq_length, input_size])
output = model(input)
print(output.shape)
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
