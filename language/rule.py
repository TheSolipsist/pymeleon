"""
Rule module for pymeleon
"""

class Rule:
    """
    Rule object for graph transformation
    
    -- Parameters --
        parser_obj_before: the parser object representing the graph that the rule can be applied to
        parser_obj_after: the parser object representing the graph after the application of the rule
        
    -- Methods --
        check(graph): checks if this rule can be applied to the specified graph
        apply(graph): applies the rule to the specified graph
    """
    
    def __init__(self, parser_obj_before, parser_obj_after):
        self._parser_obj_before = parser_obj_before
        self._parser_obj_after = parser_obj_after
    
    def check(self, graph):
        pass
    
    def apply(self, parser_obj):
        pass
