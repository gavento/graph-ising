# Ising model computation on graphs

*Authors: Tomáš Gavenčiak <gavento@gmail.com>, Jan Kulveit <jk@ks.cz>*

A suite of tools for simulating the nucleation of the Ising model on general graphs.
Includes forward-flux sampling for nucleation rate estimation.

Requirements: `numpy networkx cffi plotly attrs tqdm pytest`

Running:

* `make`
* `make test` (optional)
* `./run_ffs.py -h`

## Features

* Graph generation (`gen_graph.py`)
* Estimate mid-point I based on P(L_B|L_I)=0.5
* Output cluster snapshots at ifaces
* TODO: Estimate error (prod of bernoulli)
* 

## Large tests

### Arxiv-ca-HepTh

9877 nodes, 25998 edges, average degree 5.264, 28339 triangles, mean clustering 0.471, degree associativity coeff 0.267.

#### Generated graphs




### 2D grid nucleation rate

With T=1.5, F=0.05, the nucleation rate of 2D grid is 2.78e-19, see [here, sec III.A](http://micro.stanford.edu/~caiwei/papers/Ryu10pre-Ising.pdf).

```bash
python ffsampling.py -T 1.5 -F 0.05 --grid 50 -s 100 --Is 400 --Imin 5 --Imax 1000 -c 50x50-T1.5-F0.05-s100-I400
```

Reproduced as (at 2973022ea65b73b85f39368e1b62663c75ecdffb):

Unnormed, Imin 5: 2.14e-19, 2.71e-19.

### 3D grid nucleation rate

With T=2.662, F=0.589, the nucleation rate of 3D grid is around 5.81e-10 or 4.10e-10, see [here, sec III.A](http://micro.stanford.edu/~caiwei/papers/Ryu10pre-Ising.pdf).

```bash
python ffsampling.py -T 2.662 -F 0.589 --grid3d 50 -s 100 --Imin 10 --Imax 400 --Is 100 -c Q50-T2.66-F0.59-s100-I100
```
Reproduced as (at 2973022ea65b73b85f39368e1b62663c75ecdffb):

