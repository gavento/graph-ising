# Ising model computation on graphs

*Authors: Tomáš Gavenčiak <gavento@gmail.com>, Jan Kulveit <jk@ks.cz>*

A library and a suite of tools for simulating the nucleation of the Ising model on general graphs.
Includes forward-flux sampling for nucleation rate estimation and several plotting tools.

## Requirements

```bash
pip install numpy networkx cffi plotly attrs tqdm pytest
```

## Running

* `make`
* `make test` (optional)
* `./run_ffs.py -h`

## Data

Data samples are stored in a separate repository [gising-data](https://github.com/gavento/gising-data).

