from cffi import FFI

ffi = FFI()
ffi.cdef("""
    typedef uint64_t rand_t;
    typedef int32_t index_t;
    typedef int8_t spin_t;

    typedef struct {
	index_t n;
	spin_t *spins;
	double field;
	double T;
	index_t **neighbours; // -1 terminated neighbor list  
    } ising_state;

    uint32_t get_rand(rand_t *seed);
    double get_rand_01(rand_t *seed);
    void rand_perm(size_t n, index_t *result, rand_t *seed);

    index_t ising_mc_sweep(ising_state *s, rand_t *seed);
    index_t ising_max_cluster(ising_state *s, spin_t value, double edge_prob, rand_t *seed);
    """)
cising = ffi.dlopen('./cising.so')


