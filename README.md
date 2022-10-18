## Predictive plasticity at the single neuron level

**summary**

this is a repository for the paper:
<br/>"*Sequence anticipation and STDP emerge from a voltage-based predictive learning rule*"<br/>
M Saponati, M Vinck<br/>
(2021, BiorXiv) (under revision)<br/>
https://www.biorxiv.org/content/10.1101/2021.10.31.466667v1.full

-------------------------

![](./imgs/model_description.png)

-------------------------

**installation/dependencies**

The current version of the scripts have been tested with Python 3.8. All the dependencies are listed in the environment.yml file. 
The project has a pip-installable package. How to set it up:

- git clone the repository 
- cd ../predictive_neuron/
- pip install -e .

**structure**

this repo is structured as follows:

+ `./figures/`: contains the code necessary to reproduce all the figures in the paper

  + `/fig_model/` to reproduce the results in Fig 1
  + `/fig_sequences/` to reproduce the results in Fig 2
  + `/fig_nn_selforganization/` to reproduce the results in Fig 3
  + `/fig_stdp/` to reproduce the results in Fig 4
  
+ `./predictive_neuron/` contains the Python modules and the helper functions for the analysis
+ `./scripts/`

-------------------------
