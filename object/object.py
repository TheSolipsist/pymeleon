from language.rule_search import RuleSearch
from language.parser import Parser

class PymLiz:
    """
    Object to apply rules on
    
    -- Parameters --
        parser_obj: the parser object representing the graph of the object to use
        initialize: specifies whether to automatically replace variable names with their values (default: True) 
    
    -- Methods --
        search(rule): Iterates through the possible subgraphs (in the form of transform_dicts) that match 
        a rule's input graph
        apply(rule, transform_dict): Applies the rule to the object, using the specific transform_dict (found
        by using search(rule))
        view(): Returns the value represented by the object's graph
    """
    
    def __init__(self, viewer, parser_obj, constraint_types=None, modules=None, initialize=False) -> None:
        self._parser_obj = parser_obj
        self._variables_constants = parser_obj.variables_constants
        self._functions = parser_obj.functions
        self._graph = parser_obj.graph
        self._viewer = viewer
        self._RuleSearch = RuleSearch()
        
        if modules is None:
            modules = dict()
        self._modules = modules
        
        if constraint_types is None:
            constraint_types = dict()
        self._constraint_types = constraint_types
        self._find_satisfied_constraint_types(constraint_types, parser_obj.graph)
        
        if initialize:
            self._variable_names_to_values()
    
    def copy(self):
        return PymLiz(viewer=self._viewer, parser_obj=self._parser_obj, )
    
    def _find_satisfied_constraint_types(self, constraint_types, graph):
        for constraint_type, constraint in constraint_types.items():
            for node in tuple(graph.nodes)[1:]:
                if constraint(node.value):
                    node.constraints.add(constraint_type)
            
    def _variable_names_to_values(self):
        # Python dictionaries are ordered since 3.7, so the first element will always be "root_node"
        import config
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
    
    def _deparse_component(self, starting_node):
        """
        Deparses a connected component from the object's graph to an evaluatable expression
        """
        successors = tuple(self._graph.successors(starting_node))
        if successors:
            starting_node_value = starting_node.value
            if starting_node_value in Parser.SUPPORTED_OPERATORS_REVERSE:
                components = []
                for node in successors:
                    components.append(f"{self._deparse_component(node)}")
                return Parser.SUPPORTED_OPERATORS_REVERSE[starting_node_value].join(components)
            else:    
                expression = starting_node_value + "("
                for node in successors:
                    expression += self._deparse_component(node) + ","
                return expression[:-1] + ")"
        else:
            return str(starting_node.value)

    def search(self, rule):
        """
        Iterates through the possible subgraphs (in the form of transform_dicts) that match a rule's input graph
        """
        return self._RuleSearch(rule, self._graph)
    
    def apply(self, rule, transform_dict, inplace=False):
        """
        Applies the rule to the object, using the specific transform_dict (found by using search(rule)) and, if
        the rule is not to be applied to the graph inplace, return the transformed graph
        """
        if not inplace:
            return rule.apply(self._graph, transform_dict, deepcopy_graph=True)
        else:
            rule.apply(self._graph, transform_dict, deepcopy_graph=False)
    
    def view(self):
        return self._viewer.view(self)
        
    def run(self):
        """
        Returns the value represented by the object's graph
        """
        for obj, obj_name in self._modules.items():
            exec(f"import {obj} as {obj_name}")
        connected_components = list(self._graph.successors("root_node"))
        result = []
        for node in connected_components:
            deparsed_component = self._deparse_component(node)
            result.append(eval(deparsed_component))
        if len(result) == 1:
            return result[0]
        else:
            return None