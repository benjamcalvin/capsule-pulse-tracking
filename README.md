
# Capsule Pulse Tracking

The included notebook explains how to setup the processing pipeline. Specifics for the exact processing implementation can be found in the `algo.py` file.

[](capsule_pulse.gif)

## Setup

This has been tested on Ubuntu 17.10, but should work generally on linux systems, and potentially windows or OSX with small changes.

Clone this repository.

Recommended setup:
1. Follow directions to install [pyenv](https://github.com/pyenv/pyenv)
2. Follow directions to install [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
3. At the terminal, create a virtual environment using python 3.7 called cpt:
```bash
        pyenv virtualenv install 3.7.0
        pyenv virtualenv 3.7.0 cpt
```
4. Install the necessary python libraries in your virtualenv
```bash
        pyenv activate cpt
        pip install python-opencv2==3.4.0 numpy==1.14.3 pandas==0.22.0
```
5. Setup a project directory with the following structure:
```
      capsule-pulse-tracking/
          code/
              notebooks/
                  example.ipynb
              algo.py
          videos/
              myvideo/
                  myvideo.mp4 (or avi, etc)
```
6. Run jupyter: `jupyter lab` and open the example notebook once it's in the above directory structure.
