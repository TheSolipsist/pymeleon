"""
Rule module for pymeleon
"""
from copy import deepcopy

class BadGraph(Exception):
    """
    Invalid graph input Exception
    """
    pass
    
class Rule:
    """
    Rule object for graph transformation
    
    -- Parameters --
        parser_obj_in: the parser object representing the graph that the rule can be applied to
        parser_obj_out: the parser object representing the graph after the application of the rule
    
    -- Attributes --
        node_map: dictionary mapping values from the generic input graph to the generic output graph
                  (the set of values of all objects, including variables and functions in the input
                  graph are the dict keys and the nodes in the output graph corresponding to these values
                  are the dict values)
        
    -- Methods --
        apply(graph): applies the rule to the specified graph
    """
    
    def __init__(self, parser_obj_in, parser_obj_out):
        parser_obj_in = deepcopy(parser_obj_in)
        parser_obj_out = deepcopy(parser_obj_out)

        self._parser_obj_in = parser_obj_in
        self._parser_obj_out = parser_obj_out
        
        self._graph_in = parser_obj_in.graph
        self._obj_in = parser_obj_in.variables_constants
        self._funcs_in = parser_obj_in.functions
        self._constraints = parser_obj_in.constraints
        
        self._graph_out = parser_obj_out.graph
        self._obj_out = parser_obj_out.variables_constants
        self._funcs_out= parser_obj_out.functions

        self._create_node_map()
        
    def _create_node_map(self):
        node_map = dict()
        common_obj = self._obj_in & self._obj_out
        unmatched_nodes = list(self._graph_out.nodes)
        unmatched_nodes.remove("root")
        for obj in common_obj:
            obj_nodes = []
            for node in unmatched_nodes:
                if node.value == obj:
                    obj_nodes.append(node)
            node_map[obj] = tuple(obj_nodes)
            for node in obj_nodes:
                unmatched_nodes.remove(node)
        self.node_map = node_map

    def apply(self, graph, transform_node_dict):
        """
        Apply the rule to the specified graph, returning the transformed graph

        -- Arguments --
            graph (networkx DiGraph): the graph to transformx
            transform_node_map (dict): dictionary mapping each node.value from the generic input graph to the graph to be transformed
        
        -- Returns --
            transformed_graph (networkx DiGraph): the transformed graph
        """
        node_map = self.node_map
        for node_value in node_map:
            for node in node_map[node_value]:
                node.value = transform_node_dict[node_value]
                self._graph_out.nodes[node]["name"] = node.value
        transformed_graph = deepcopy(self._graph_out)
        for node_value in node_map:
            for node in node_map[node_value]:
                node.value = node_value
                self._graph_out.nodes[node]["name"] = node.value
        return transformed_graph
        

class RuleSearch:
    pass