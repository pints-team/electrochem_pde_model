This is an example repository showing how to use both 
[PyBaMM](https://github.com/pybamm-team/PyBaMM) and 
[Pints](https://github.com/pints-team/pints) to perform inference on an complex PDE 
model in electrochemistry.

# Installation

## Prerequisites

This repository requires python3, which you can obtain on a debian-based Linux 
distribution using `apt`:

```bash
sudo apt install python3
```

We recommend using a python virtual environment to install the dependencies, which you 
can create using:

```bash
python3 -m venv env
```

You can then "activate" the environment using:

```bash
source env/bin/activate
```

Finally, install the dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
