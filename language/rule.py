"""
Rule module for pymeleon
"""
from networkx import DiGraph
from itertools import product

class BadGraph(Exception):
    """
    Invalid graph input Exception
    """
    pass

def _replace_node(graph, old_node, new_node):
    """
    Replace old_node with new_node in the specified networkx DiGraph
    """
    nodes_in = tuple(graph.predecessors(old_node))
    nodes_out = tuple(graph.successors(old_node))
    graph.remove_node(old_node)
    graph.add_node(new_node)
    graph.add_edges_from(product(nodes_in, (new_node,)))
    graph.add_edges_from(product((new_node,), nodes_out))
    
class Rule:
    """
    Rule object for graph transformation
    
    -- Parameters --
        parser_obj_in: the parser object representing the graph that the rule can be applied to
        parser_obj_out: the parser object representing the graph after the application of the rule
    
    -- Attributes --
        node_map: dictionary mapping values from the generic input graph to the generic output graph
                  (the set of values of all objects, including variables and functions in the input
                  graph are the dict_keys and the nodes in the output graph corresponding to these values
                  are the dict_values)
        
    -- Methods --
        apply(graph): applies the rule to the specified graph
    """
    
    def __init__(self, parser_obj_in, parser_obj_out):
        self._parser_obj_in = parser_obj_in
        self._parser_obj_out = parser_obj_out
        
        self._graph_in = parser_obj_in.graph
        self._vars_in = parser_obj_in.variables_constants
        self._funcs_in = parser_obj_in.functions
        self._constraints = parser_obj_in.constraints
        
        self._graph_out = parser_obj_out.graph
        self._vars_out = parser_obj_out.variables_constants
        self._funcs_out= parser_obj_out.functions
        
        self._create_node_map()
        
    def _create_node_map(self):
        node_map = dict()
        self.node_map = node_map

    def apply(self, graph, transform_node_dict):
        """
        Apply the rule to the specified graph, returning the transformed graph

        Args:
            graph (networkx DiGraph): the graph to transform
            transform_node_map (dict): dictionary mapping each node from the generic input graph to the graph to be transformed
        
        Returns:
            transformed_graph (networkx DiGraph): the transformed graph
        """
        transformed_graph = DiGraph(self._graph_out)
        for input_node, old_node in self.node_map.items():
            new_node = transform_node_dict[input_node]
            _replace_node(transformed_graph, old_node, new_node)
        return transformed_graph
        

class RuleSearch:
    pass