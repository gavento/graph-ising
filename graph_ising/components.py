from .graph_set import GraphSet
import tensorflow.compat.v2 as tf


class ComponentsMixin(GraphSet):

    def __init__(self, **kwargs):
        "Create variables for find_components and friends"
        super().__init__(**kwargs)

        self._add_var('C_comp_nums', tf.zeros([self.v_order], dtype=self.itype))
        self._add_var('C_iters', 0, dtype=self.itype)
        self._add_var('C_unfinished', 0.0, dtype=self.ftype)
        self._add_var('C_run', True)
        self._add_var('C_mean_max_sizes', tf.zeros([self.v_n], dtype=self.ftype))
        self.m_C_unfinished = self._add_mean_metric('components/unfinished')
        self.m_C_iterations = self._add_mean_metric('components/iterations')

    def find_components(self, node_mask=None, edge_mask=None, max_iters=16):
        """
        Return component numbers for every vertex as shape [n, order].
        Components are numbered 1 .. (n * max_order), comp 0 is used for masked and
        non-existing vertices.
        """

        max_iters = tf.identity(max_iters)
        initial_comp_nums = tf.range(1, self.order + 1, dtype=tf.int64)
        if node_mask is not None:
            node_mask = tf.cast(node_mask, self.itype)
            initial_comp_nums = initial_comp_nums * node_mask

        # Alias and init variables (NOTE: actually local variables)
        comp_nums = self.v_C_comp_nums
        comp_nums.assign(initial_comp_nums)
        iters = self.v_C_iters
        iters.assign(0)
        unfinished = self.v_C_unfinished
        unfinished.assign(0.0)
        run = self.v_C_run
        run.assign(True)

        while run:
            neigh_nums = self.max_neighbors(comp_nums, edge_weights=edge_mask)
            mask_neigh_nums = neigh_nums if node_mask is None else neigh_nums * node_mask
            new_comp_nums = tf.maximum(comp_nums, mask_neigh_nums)
            iters.assign_add(1)
            if tf.reduce_all(tf.equal(new_comp_nums, comp_nums)):
                run.assign(False)
            comp_nums.assign(new_comp_nums)
            if iters >= tf.cast(max_iters, self.itype):
                unfinished.assign(1.0)
                run.assign(False)

        self.m_C_unfinished.update_state(unfinished)
        self.m_C_iterations.update_state(tf.cast(iters, tf.float32))

        if node_mask is not None:
            comp_nums == comp_nums * node_mask
        return tf.identity(comp_nums)

    def largest_components(self, node_mask=None, edge_mask=None, max_iters=16):
        "Return the largest component size for every graph."
        comp_nums = self.find_components(node_mask=node_mask, edge_mask=edge_mask, max_iters=max_iters)
        K = tf.cast(self.v_order + 1, tf.int32)
        comp_sizes = tf.math.bincount(tf.cast(comp_nums, dtype=tf.int32), minlength=K, maxlength=K)
        comp_sizes = comp_sizes[1:]  # drop the 0-th component
        return tf.math.segment_max(comp_sizes, self.v_batch)

    def mean_largest_components(self, node_mask=None, edge_mask=None, drop_edges=0.5, samples=10, max_iters=16):
        """
        Mean largest components over several samples of differet edge-drops.

        Dropping is combined with `edge_mask` if provided.
        """
        samples = tf.identity(samples)
        drop_edges = tf.identity(drop_edges)
        # Alias and reset vars
        mean_max_sizes = self.v_C_mean_max_sizes
        mean_max_sizes.assign(tf.zeros([self.n], dtype=self.ftype))

        for i in range(samples):
            em = tf.random.uniform([self.v_size]) >= drop_edges
            if edge_mask is not None:
                em = em & edge_mask
            largest = self.largest_components(node_mask=node_mask, edge_mask=em, max_iters=max_iters)
            mean_max_sizes.assign_add(tf.cast(largest, self.ftype))
        return mean_max_sizes / tf.cast(samples, self.ftype)

