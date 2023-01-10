from pymeleon.dsl.parser import Parser, PymLizParser, RuleParser
from pymeleon.dsl.rule import Rule
import ctypes


class PymLiz:
    """
    Object to apply rules on
    
    -- Parameters --
        viewer: The viewer which creates the object
        parser_obj: The parser object representing the graph of the object to use
        constraint_types: All constraint types to check for the object's components
        ext: All ext to be used when "running" the object (returning its value)
    
    -- Methods --
        search(rule): Iterates through the possible subgraphs (in the form of transform_dicts) that match 
        a rule's input graph
        apply(rule, transform_dict): Applies the rule to the object, using the specific transform_dict (found
        by using search(rule))
        view(): Calls the viewer's "view" method
    """

    def __init__(self, viewer, parser_obj: PymLizParser, constraint_types=None, ext=None) -> None:
        self._parser_obj = parser_obj
        self._graph = parser_obj.graph
        self._viewer = viewer
        if ext is None:
            ext = dict()
        self._ext = ext
        if constraint_types is None:
            constraint_types = dict()
        self._constraint_types = constraint_types
        self._find_satisfied_constraint_types(constraint_types, parser_obj.graph)

    def copy(self):
        """
        Returns a deep copy of the PymLiz object
        """
        return PymLiz(viewer=self._viewer,
                      parser_obj=self._parser_obj.copy(),
                      constraint_types=self._constraint_types.copy(),
                      ext=self._ext.copy())

    def get_graph(self):
        """
        Returns the object's graph
        """
        return self._graph

    def _find_satisfied_constraint_types(self, constraint_types, graph):
        for constraint_type, constraint in constraint_types.items():
            for node in tuple(graph.nodes)[1:]:
                if constraint(node.value):
                    node.constraints.add(constraint_type)

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
            return f"__pym_get_obj({id(starting_node.value)})"

    def apply(self, rule: Rule, transform_dict: dict, inplace=False):
        """
        Applies the rule to the object, using the specific transform_dict (found by using search(rule)) and, if
        the rule is not to be applied to the graph inplace, return the transformed graph
        """
        if not inplace:
            new_obj = self.copy()
            new_obj._graph = rule.apply(self._graph, transform_dict, deepcopy_graph=True)
            return new_obj
        else:
            rule.apply(self._graph, transform_dict, deepcopy_graph=False)

    def view(self, *args):
        return self._viewer.view(self, *args)

    def run(self):
        """
        Returns the value represented by the object's graph
        """
        def __pym_get_obj(obj_id):
            return ctypes.cast(obj_id, ctypes.py_object).value
        connected_components = list(self._graph.successors("root_node"))
        result = []
        for node in connected_components:
            deparsed_component = self._deparse_component(node)
            try:
                result.append(eval(deparsed_component, self._ext | {"__pym_get_obj": __pym_get_obj}))
            except Exception as e:
                raise Exception("Exception occurred during the execution of", deparsed_component)
        if len(result) == 1:
            return result[0]
        else:
            return result

    def __rshift__(self, right: RuleParser):
        return self.view(right)
