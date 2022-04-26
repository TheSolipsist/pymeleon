"""
Rule search module
"""
# TODO: Come up with and implement way of using edges with order -1
#       Finish find_subgraphs

class RuleSearch:
    """
    Rule search object for graph transformation
    
    RuleSearch objects are callable and return a generator, yielding subgraphs that match a rule's input graph
    
    -- Parameters --
        pass
        
    -- Attributes --
        pass
        
    -- Methods --
        pass 
    """
    
    def __init__(self):
        pass

    def __call__(self, rule, target_graph):
        """
        Returns a generator yielding node_dicts that match a rule's input graph from a given graph

        -- Arguments --
            rule (pymeleon Rule): The rule for which to find subgraphs matching the input graph
            graph (networkx DiGraph): The graph from which to find subgraphs
            
        -- Returns --
            search_iter: Iterator yielding the node_dicts corresponding to the matching subgraphs
        """
        return self._find_subgraphs(rule, target_graph)

    def _check_node_subgraph_rec(self, node_input_graph, node):
        if node_input_graph.value in self._cur_funcs:
            if node.value != node_input_graph.value:
                return False
        elif not self._cur_constraints[node_input_graph](node.value): # Check if node_value satisfies the constraint
            return False
        if self._cur_input_graph.out_degree(node_input_graph) != self._cur_target_graph.out_degree(node):
            return False
        self._cur_node_dict[node_input_graph] = node
        for suc_input_node in self._cur_input_graph.successors(node_input_graph):
            order = self._cur_input_graph[node_input_graph][suc_input_node]["order"]
            target_order = -2 # -1 is used for cases where no order is required
            for suc_node in self._cur_target_graph.successors(node):
                if self._cur_target_graph[node][suc_node]["order"] == order:
                    target_order = order
                    break
            if target_order == -2:
                print("Order error detected - current subgraph will not be checked")
                return False
            if not self._check_node_subgraph_rec(suc_input_node, suc_node):
                return False
        return True

    def _check_node_subgraph(self, root_input_node, node):
        """
        Check if the argument node is the root of a matching connected subgraph
        """
        self._cur_node_dict = dict()
        return self._check_node_subgraph_rec(root_input_node, node)

    def _find_connected_matching_subgraphs(self):
        """
        Returns a list of all node_dicts corresponding to connected subgraphs that match the connected input_graph
        and its constraints in the target_graph
        """
        matching_subgraphs = []
        for node in self._cur_target_graph:
            if self._check_node_subgraph(self._cur_root_input_node, node):
                matching_subgraphs.append(self._cur_node_dict)
        return matching_subgraphs

    def _find_subgraphs(self, rule, target_graph):
        """
        Returns a generator yielding node_dicts that match a rule's input graph from a given graph

        Should only be called by __call__
        """
        self._cur_input_graph = rule._graph_in
        self._cur_constraints = rule._constraints
        self._cur_funcs = rule._funcs_in
        self._cur_target_graph = target_graph
        connected_subgraphs = dict()
        for node in self._cur_input_graph.successors("root_node"):
            # For any nodes that are not connected to the rest of the graph, find all matching subgraphs
            self._cur_root_input_node = node
            connected_subgraphs[node] = self._find_connected_matching_subgraphs()
        # Yield all combinations of node_dicts in the connected_subgraphs dict (1 from each matching_subgraphs list), added
        # in a single dict, that do not have any overlaps in nodes chosen