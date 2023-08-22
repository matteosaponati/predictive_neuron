## a predictive learning rule in single neurons

**summary**

this is a repository for the paper:
<br/>"*Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule*"<br/>
M Saponati, M Vinck<br/>
Nature Communications 14, 4985 (2023). <br/>
https://doi.org/10.1038/s41467-023-40651-w

-------------------------

![](./imgs/fig_model.png)

-------------------------

**installation/dependencies**

The current version of the scripts has been tested with Python 3.8. All the dependencies are listed in the environment.yml file. 
The project has a pip-installable package. How to set it up:

- `git clone` the repository 
- `pip install -e . `

**structure**

this repo is structured as follows:

+ `./figures/`: contains the code necessary to reproduce all the figures in the paper
+ `./models/` contains the Python Class of the different models
+ `./scripts/` contains scripts to run the model on different types of inputs and network implementations
+ `./utils/` contains the Python modules for training and the helper functions for the analysis

+ `environment.yml` configuration file with all the dependencies listed
+ `setup.py` python script for installation with pip
-------------------------

 ## citation and credits
Saponati, M., Vinck, M. (2023).
**Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule**. Nature Communications, 14(1), 1-13.<br/>
```
@article{saponati2023sequence,
  title={Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule},
  author={Saponati, Matteo and Vinck, Martin},
  journal={Nature communications},
  volume={14},
  number={1},
  pages={1--13},
  year={2023},
  publisher={Nature Publishing Group}
}
