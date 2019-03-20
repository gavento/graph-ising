from .graph_set import GraphSet

class ComponentsMixin(GraphSet):

    def _create_vars(self):
        "Create variables for find_components and friends"
        self.v_FC_comp_nums = tf.Variable(tf.zeros(self.order, dtype=tf.int64))
        self.v_FC_iters = tf.Variable(0, dtype=tf.int64)
        self.v_FC_unfinished = tf.Variable(0.0, dtype=self.DTYPE)
        self.v_FC_run = tf.Variable(True)
        self.V_SLC_mean_max_sizes = tf.Variable(tf.zeros([self.n], dtype=self.DTYPE))

    def find_components(self, node_mask=None, edge_mask=None, max_iters=tf.constant(32)):
        """
        Return component numbers for every vertex as shape [n, order].
        Components are numbered 1 .. (n * max_order), comp 0 is used for masked and
        non-existing vertices.
        """

        max_iters = tf.identity(max_iters)
        initial_comp_nums = tf.range(1, self.order + 1, dtype=tf.int64)
        if node_mask is not None:
            initial_comp_nums = initial_comp_nums * node_mask

        # Alias and init variables
        comp_nums = self.v_FC_comp_nums
        comp_nums.assign(initial_comp_nums)
        iters = self.v_FC_iters
        iters.assign(0)
        unfinished = self.v_FC_unfinished
        unfinished.assign(0.0)
        run = self.v_FC_run
        run.assign(True)

        while run:
            neigh_nums = self.max_neighbors(comp_nums, edge_mask=edge_mask)
            mask_neigh_nums = neigh_nums if node_mask is None else neigh_nums * tf.cast(node_mask, tf.int64)
            new_comp_nums = tf.maximum(comp_nums, mask_neigh_nums)
            iters.assign_add(1)
            if tf.reduce_all(tf.equal(new_comp_nums, comp_nums)):
                run.assign(False)
            comp_nums.assign(new_comp_nums)
            if iters >= tf.cast(max_iters, tf.int64):
                unfinished.assign(1.0)
                run.assign(False)

        self.metric_components_unfinished.update_state(unfinished)
        self.metric_components_iterations.update_state(tf.cast(iters, tf.float32))

        if node_mask is not None:
            comp_nums == comp_nums * tf.cast(node_mask, tf.int64)
        return tf.identity(comp_nums)

    def largest_cluster(self, spins, positive_spin=True, drop_edges=None, max_iters=32):
        """
        Return the size of larges positive (res. negative) spin cluster for every graph.
        If given, `drop_edges`-fraction of edges is ignored.
        """
        K = self.n * self.max_order
        if positive_spin:
            node_mask = tf.cast(spins > 0.0, tf.int64)
        else:
            node_mask = tf.cast(spins < 0.0, tf.int64)
        if drop_edges is not None:
            lls = tf.math.log([[drop_edges, 1.0 - drop_edges]])
            edge_mask = tf.reshape(tf.random.categorical(lls, tf.cast(self.v_tot_size, tf.int32) * 2), (-1, ))
        else:
            edge_mask = None

        comp_nums = self.find_components(node_mask=node_mask, edge_mask=edge_mask, max_iters=max_iters)
        comp_nums = tf.reshape(comp_nums, [K])
        comp_sizes = tf.math.unsorted_segment_sum(tf.fill([K], 1), comp_nums, K + 1)
        comp_sizes = comp_sizes[1:]  # drop the 0-th component
        comp_sizes = tf.reshape(comp_sizes, [self.n, self.max_order])

        max_comp_sizes = tf.reduce_max(comp_sizes, axis=1)
        return max_comp_sizes

    def sampled_largest_cluster(self, spins, positive_spin=True, drop_edges=tf.constant(0.5), samples=tf.constant(10), max_iters=32):
        """
        Mean largest_cluster over several samples.
        """
        mean_max_sizes = self.V_SLC_mean_max_sizes
        mean_max_sizes.assign(tf.zeros([self.n], dtype=self.dtype))
        for i in range(samples):
            largest = self.largest_cluster(spins, positive_spin=positive_spin, drop_edges=drop_edges, max_iters=max_iters)
            mean_max_sizes.assign_add(tf.cast(largest, self.dtype))
        return mean_max_sizes / tf.cast(samples, self.dtype)
