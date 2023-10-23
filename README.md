# Dogs-vs-Cats-
## To install Tensorflow on Mac M1,M2
`coming from: https://stackoverflow.com/questions/54843623/pip-install-tensorflow-failed`

Download and install [Conda env](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh).

```python
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

Install the TensorFlow dependencies:

```python
conda install -c apple tensorflow-deps==2.6.0
```

Install base TensorFlow:

```python
python -m pip install tensorflow-macos
```

Install tensorflow-metal plugin

```python
python -m pip install tensorflow-metal
```
#### Create environment
`conda create -n competitors_env python=3.11` </br>
`conda activate competitors_env`</br>

#### Install requirements
`conda install pip` *(if you don't have pip already)* </br>
`pip install -r requirements.txt`</br>

## Get data
`python Tools/auto_get.py`

## Perform Grid Test
`python Grid_test.py`

