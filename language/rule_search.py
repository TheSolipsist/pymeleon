"""
Rule search module
"""
# TODO: Come up with and implement way of using edges with order -1
#       Remove root_node and instead use a list of all starting nodes
from itertools import product, chain

class RuleSearch:
    """
    Rule search object for graph transformation
    
    RuleSearch objects are callable and return a generator, yielding subgraphs that match a rule's input graph
    """
    
    def __init__(self) -> None:
        pass

    def __call__(self, rule, target_graph):
        """
        Returns a generator yielding transform_dicts that match a rule's input graph from a given graph

        -- Arguments --
            rule (pymeleon Rule): The rule for which to find subgraphs matching the input graph
            graph (networkx DiGraph): The graph from which to find subgraphs
            
        -- Returns --
            search_iter: Iterator yielding the transform_dicts corresponding to the matching subgraphs
        """
        return self._find_subgraphs(rule, target_graph)

    def _check_node_subgraph_rec(self, node_input_graph, node):
        for constraint in node_input_graph.constraints:
            if constraint not in node.constraints:
                return False
        self._cur_transform_dict[node_input_graph] = node
        for suc_input_node in self._cur_input_graph.successors(node_input_graph):
            order = self._cur_input_graph[node_input_graph][suc_input_node]["order"]
            target_order = "NOT FOUND" # -1 is used for cases where no order is required
            for suc_node in self._cur_target_graph.successors(node):
                if self._cur_target_graph[node][suc_node]["order"] == order:
                    target_order = order
                    break
            if target_order == "NOT FOUND":
                print("Order error detected - current subgraph will not be checked")
                return False
            if not self._check_node_subgraph_rec(suc_input_node, suc_node):
                return False
        return True

    def _check_node_subgraph(self, root_input_node, node):
        """
        Check if the argument node is the root of a matching connected subgraph
        """
        self._cur_transform_dict = dict()
        return self._check_node_subgraph_rec(root_input_node, node)

    def _find_connected_matching_subgraphs(self):
        """
        Returns a list of all transform_dicts corresponding to connected subgraphs that match the connected input_graph
        and its constraints in the target_graph
        """
        matching_subgraphs = []
        for node in tuple(self._cur_target_graph)[1:]:
            if self._check_node_subgraph(self._cur_root_input_node, node):
                matching_subgraphs.append(self._cur_transform_dict)
        return matching_subgraphs

    def _find_subgraphs(self, rule, target_graph):
        """
        Returns a generator yielding transform_dicts that match a rule's input graph from a given graph

        Should only be called by __call__
        """
        self._cur_input_graph = rule._graph_in
        self._cur_target_graph = target_graph
        connected_subgraphs = []
        for node in self._cur_input_graph.successors("root_node"):
            # For any nodes that are not connected to the rest of the graph, find all matching subgraphs
            self._cur_root_input_node = node
            connected_subgraphs.append(self._find_connected_matching_subgraphs())
        del self._cur_input_graph
        del self._cur_target_graph
        # For all combinations of transform_dicts in the connected subgraphs that do not have any overlaps in their chosen
        # nodes, combine the transform_dicts in a single transform_dict and yield it
        for transform_dict_combination in product(*connected_subgraphs):
            nodes = tuple(chain.from_iterable([transform_dict.values() for transform_dict in transform_dict_combination]))
            if len(nodes) == len(set(nodes)): # Checking for duplicates
                full_transform_dict = {k: v for d in transform_dict_combination for k, v in d.items()}
                yield full_transform_dict
