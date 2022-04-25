"""
Rule search module
"""

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
    
    def __call__(self, rule, graph):
        """
        Return a generator yielding subgraphs that match a rule's input graph from a given graph

        -- Arguments --
            rule (pymeleon Rule): The rule for which to find subgraphs matching the input graph
            graph (networkx DiGraph): The graph from which to find subgraphs
            
        -- Returns --
            search_iter (iterator): Iterator yielding the matching subgraphs
        """
        pass