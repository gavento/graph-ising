* cising.c: Fast Ising model on graphs
* ising.py: Python wrapper IsingState and cffi interface to cising.c
* tp_ising.py: Demo with ThreadPoolExecutor

Python 2.x requirements: numpy, networkx, cffi, futures

Python 3.x requirements: numpy, networkx, cffi

Running:
* make
* python tp_ising.py


## Large tests

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
