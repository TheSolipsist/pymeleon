from language.rule_search import RuleSearch
import config

class PymLiz:
    """
    Object to apply rules on
    
    -- Parameters --
        parser_obj: the parser object representing the graph of the object to use
        initialize: specifies whether to automatically replace variable names with their values (default: True) 
    
    -- Methods --
        search(rule): Iterate through the possible subgraphs (in the form of transform_dicts) that match 
        a rule's input graph
        apply(rule, transform_dict): Apply the rule to the object, using the specific transform_dict (found
        by using search(rule))
    """
    def __init__(self, parser_obj, initialize=True) -> None:
        self._parser_obj = parser_obj
        self._variables_constants = parser_obj.variables_constants
        self._functions = parser_obj.functions
        self._graph = parser_obj.graph
        self._RuleSearch = RuleSearch()
        if initialize:
            self._variable_names_to_values()
    
    def _variable_names_to_values(self):
        # Python dictionaries are ordered since 3.7, so the first element will always be "root_node"
        local_vars = config.locals
        global_vars = config.globals
        variables_constants = self._variables_constants
        for node in tuple(self._graph.nodes)[1:]:
            node_value = node.value
            if node_value in variables_constants:
                if node_value in local_vars:
                    node.value = local_vars[node_value]
                elif node_value in global_vars:
                    node.value = global_vars[node_value]
                    print(f"WARNING: {node_value} is defined only in the global namespace: {node.value}")
                else:
                    raise NameError(f"name '{node_value}' is not defined")
                
    def search(self, rule):
        """
        Iterate through the possible subgraphs (in the form of transform_dicts) that match a rule's input graph
        """
        return self._RuleSearch(rule, self._graph)
    
    def apply(self, rule, transform_dict, inplace=False):
        """
        Apply the rule to the object, using the specific transform_dict (found by using search(rule)) and, if
        the rule is not to be applied to the graph inplace, return the transformed graph
        """
        if not inplace:
            return rule.apply(self._graph, transform_dict, deepcopy_graph=True)
        else:
            rule.apply(self._graph, transform_dict, deepcopy_graph=False)
            