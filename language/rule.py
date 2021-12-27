"""
Rule module for pymeleon
"""
from networkx import DiGraph

class BadGraph(Exception):
    """
    Exception to raise in case of invalid graph input to rule application
    """
    pass
    
class Rule:
    """
    Rule object for graph transformation
    
    -- Parameters --
        parser_obj_in: the parser object representing the graph that the rule can be applied to
        parser_obj_out: the parser object representing the graph after the application of the rule
        
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
    
    def apply(self, graph):
        """
        Apply the rule to the specified graph, returning the transformed graph
        """
        def apply_rec(root_node, root_node_check):
            successors = list(graph.successors())
            successors_check = list(self._graph_in.successors(root_node_check))
            if root_node_check.value in self._funcs_in:
                if root_node_check.value != root_node.value or len(successors) != len(successors_check):
                    raise BadGraph
                
            else:
                pass
        
        new_graph = DiGraph()
        root_node = list(graph.successors('root'))
        root_node_check = list(self._graph_in.successors('root'))
        try:
            apply_rec(root_node, root_node_check)
        except BadGraph:
            print("Rule cannot be applied to the specified graph")
            
        return new_graph
