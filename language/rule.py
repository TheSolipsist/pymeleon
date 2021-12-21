"""
Rule module for pymeleon
"""

class Rule:
    """
    Rule object for graph transformation
    
    -- Parameters --
        parser_obj_in: the parser object representing the graph that the rule can be applied to
        parser_obj_out: the parser object representing the graph after the application of the rule
        
    -- Methods --
        check(graph): checks if this rule can be applied to the specified graph
        apply(graph): applies the rule to the specified graph
    """
    
    def __init__(self, parser_obj_in, parser_obj_out):
        self._parser_obj_in = parser_obj_in
        self._parser_obj_out = parser_obj_out
        self._graph_in = parser_obj_in.graph
        self._vars_in = parser_obj_in.variables_constants
        self._funcs_in = parser_obj_in.functions
    
    def check(self, graph):
        """
        Check if rule can be applied to the specified graph
        """
        def check_rec(root_node, root_node_check):
            
            if root_node_check.value in self._funcs_in:
                if  (
                    root_node_check.value != root_node.value or
                    len(list(graph.successors(root_node))) != len(list(graph.successors(root_node_check)))
                    ):
                    raise ValueError
                
            else:
                pass
        
        root_node = next(graph.successors('root'))
        root_node_check = next(self._graph_in.successors('root'))
        try:
            check_rec(root_node, root_node_check)
            return True
        except ValueError:
            print("Rule cannot be applied to the specified graph")
            return False
    
    def apply(self, graph):
        """
        Apply the rule to the specified graph, transforming it to the 
        """
        pass
