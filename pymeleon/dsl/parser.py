from typing import Callable
import networkx as nx

from pymeleon.dsl.rule import Rule


class Wrapper:
    """
    Wrapper around any Python object that holds a value
    """

    def __init__(self, value) -> None:
        self.value = value


class Node(Wrapper):
    """
    Node class for usage in the graph
    
    -- Parameters --
        value: The node's value
        constraints: List of constraints that the node satisfies
    """

    def __init__(self, value="empty_node", constraints=None) -> None:
        if constraints is None:
            constraints = set()
        self.value = value
        self.constraints = constraints

    def copy(self):
        """
        Returns a deep copy of the Node object
        """
        return Node(self.value, constraints=self.constraints.copy())


class Expression(Wrapper):
    pass


class ParsingError(Exception):
    pass


class Parser:
    """
    Abstract Parser class for graph generation from Python expressions
    """
    # The following operators should be written in reverse order of execution hierarchy (e.g. + is higher than *)
    SUPPORTED_OPERATORS = {
        "-": "__pymeleon_sub",
        "+": "__pymeleon_add",
        "@": "__pymeleon_matmul",
        "%": "__pymeleon_mod",
        "//": "__pymeleon_floordiv",
        "*": "__pymeleon_mul",
        "/": "__pymeleon_truediv",
        "^": "__pymeleon_power",
        "==": "__pymeleon_eq",
        ">": "__pymeleon_gt",
        ">=": "__pymeleon_ge",
        "<": "__pymeleon_lt",
        "<=": "__pymeleon_le",
        # ".": "__pymeleon_getattr"
    }
    SUPPORTED_OPERATORS_REVERSE = {v: k for k, v in SUPPORTED_OPERATORS.items()}

    def __init__(self, *args, constraints=None) -> None:
        raise NotImplementedError("Abstract Parser __init__ should not be called")


def add_constraints_to_nodes(graph, constraints):
    """
    Adds any constraints from the constraints dicts to their corresponding nodes
    Example: {"a": "isint"} -> node_a.constraints == set(..., "isint")
    """
    for node in tuple(graph.nodes)[1:]:  # Skipping "root_node"
        if node.value in constraints:
            for constraint in constraints[node.value]:
                node.constraints.add(constraint)


def parse_expression(expression, functions, variables_constants):
    """
    Parses the expression and returns a graph representation list
    """

    def disassemble_expression(expression, functions, variables_constants):
        """
        Populates the variables_constants and functions sets
        """
        for operator in Parser.SUPPORTED_OPERATORS:
            expression = expression.replace(operator, " ")
        expression = expression.replace(")", "").replace(",", " ").split(" ")
        for item in expression:
            if "(" in item[1:]:
                split_item = item.split("(")
                # Last item in split_item doesn't have a left bracket on its right, so it is not a function
                for func in split_item[:-1]:
                    if func:
                        functions.add(func)
                # Hopefully there are no naked operators or commas after a left bracket, so, since the last item
                # is not a function, it must be a variable or a constant
                variables_constants.add(split_item[-1])
            else:
                variables_constants.add(item)

    def operators_to_functions(expr_obj):
        """
        Changes an expression such that each operator becomes a function (e.g. "func(a+b,b*c)" => "func(add(a,b),mul(b,c))")
        """

        def custom_split(expr_obj, i):
            starting_position = i
            split_list, args_list, curr_str = [], [], ""
            while i < len(expr_obj.value):
                curr_char = expr_obj.value[i]
                if curr_char == operator:
                    split_list.append(curr_str)
                    curr_str = ""
                elif curr_char == "(":
                    next_char_position = i + 1
                    i = custom_split(expr_obj, next_char_position)
                    curr_str += expr_obj.value[next_char_position - 1:i + 1]
                elif curr_char == ",":
                    if split_list:
                        split_list.append(curr_str)
                        args_list.append(f"{operator_function}(" + ",".join(split_list) + ")")
                    else:
                        args_list.append(curr_str)
                    split_list, curr_str = [], ""
                elif curr_char == ")":
                    if split_list:
                        split_list.append(curr_str)
                        joined_str = f"{operator_function}(" + ",".join(split_list) + ")"
                        if expr_obj.value[starting_position - 2] not in Parser.SUPPORTED_OPERATORS:
                            args_list.append(joined_str)
                            new_expression = "(" + ",".join(args_list) + ")"
                        else:
                            new_expression = joined_str
                        expr_obj.value = expr_obj.value[:starting_position - 1] + new_expression + expr_obj.value[
                                                                                                   i + 1:]
                        # The next character to be checked should be the character after ")"
                        return starting_position - 2 + len(new_expression)
                    elif expr_obj.value[starting_position - 2] not in Parser.SUPPORTED_OPERATORS:
                        args_list.append(curr_str)
                        new_expression = "(" + ",".join(args_list) + ")"
                        expr_obj.value = expr_obj.value[:starting_position - 1] + new_expression + expr_obj.value[
                                                                                                   i + 1:]
                        return starting_position - 2 + len(new_expression)
                    else:
                        return i
                else:
                    curr_str += curr_char
                i += 1
            if split_list:
                split_list.append(curr_str)
                expr_obj.value = f"{operator_function}(" + ",".join(split_list) + ")"

        for operator in Parser.SUPPORTED_OPERATORS:
            if operator in expr_obj.value:
                operator_function = Parser.SUPPORTED_OPERATORS[operator]
                custom_split(expr_obj, 0)

    def add_from_brackets(expr_obj, bracket_list, bracket_i):
        if bracket_i > 0:
            str_to_add = expr_obj.value[:bracket_i]
            for item in str_to_add.split(","):
                if item:
                    bracket_list.append(item)
        expr_obj.value = expr_obj.value[bracket_i + 1:]

    def parse_brackets_rec(expr_obj):
        """
        Returns a list representation of functions and their arguments in an expression
        """
        bracket_contents = []
        while (l_bracket := expr_obj.value.find("(")) < (r_bracket := expr_obj.value.find(")")) and l_bracket != -1:
            add_from_brackets(expr_obj, bracket_contents, l_bracket)
            bracket_contents.append(parse_brackets_rec(expr_obj))
        if r_bracket != -1:
            add_from_brackets(expr_obj, bracket_contents, r_bracket)
        else:
            if expr_obj.value:
                bracket_contents.append(expr_obj.value)
        return bracket_contents

    expr_obj = Expression(expression)
    operators_to_functions(expr_obj)
    disassemble_expression(expr_obj.value, functions, variables_constants)
    brackets_list = parse_brackets_rec(expr_obj)
    return brackets_list, functions, variables_constants


def generate_graph(expression):
    """
    Generates and returns an expression's graph, functions and variables_constants
    """

    def generate_subgraph(root_node, arguments_list, functions):
        arg_iter = enumerate(arguments_list)
        functions_found = 0
        for i, item in arg_iter:
            item_node = Node(item)
            isfunction = item in functions
            graph.add_node(item_node)
            graph.add_edge(root_node, item_node, order=i - functions_found + 1)
            if isfunction:
                functions_found += 1
                next(arg_iter)
                generate_subgraph(item_node, arguments_list[i + 1], functions)

    graph = nx.DiGraph()
    functions, variables_constants = set(), set()
    graph_list, functions, variables_constants = parse_expression(expression, functions, variables_constants)
    graph.add_node("root_node")
    generate_subgraph("root_node", graph_list, functions)
    return graph, functions, variables_constants


def generate_graph_simple(args):
    """
    Generates the graph when a series of arbitrary objects is given
    """
    graph = nx.DiGraph()
    graph.add_node("root_node")
    for i, item in enumerate(args):
        graph.add_edge("root_node", Node(item), order=-1)
    return graph


class RuleParser(Parser):
    """
    Parser class for Rule components

    -- Parameters --
        *expressions: Unpacked list of strings containing Python expressions
        constraints: Dictionary mapping expression variables to constraints (e.g. type)

    -- Attributes --
        graph: The generated networkx graph
        variables_constants: Set containing all variables and constants in the expression
        functions: Set containing all functions in the expression
    """
    DEFAULT_SINGLE_INPUT_NAME = "_"
    
    def __init__(self, *args, constraints=None, constraints_func_dict=None) -> None:
        self.functions, self.variables_constants = set(), set()
        self._args = args
        component_graphs = []
        for expression in args:
            if not isinstance(expression, str):
                raise ParsingError("RuleParser requires str arguments")
            # Exponentiation is represented by "^", occurrences of "**" are replaced
            expression = expression.replace(" ", "").replace("**", "^")
            try:
                graph, functions, variables_constants = generate_graph(expression)
                self.functions |= functions
                self.variables_constants |= variables_constants
                component_graphs.append(graph)
            except ValueError:
                print("Inappropriate expression. Refer to the parser's requirements.")
        self.graph = nx.compose_all(component_graphs)
        if constraints is None:
            self.constraints = dict()
        else:
            self.constraints = self.fix_constraints(constraints)
            add_constraints_to_nodes(self.graph, self.constraints)
        if constraints_func_dict is None:
            constraints_func_dict = dict()
        self.constraints_func_dict = constraints_func_dict

    def __str__(self):
        return f"{', '.join(self._args)} with constraints {self.constraints}"

    def fix_constraints(self, constraints):
        """
        Returns a fixed constraints dict (each value should be an iterable yielding constraint types)
        """
        for node_value in constraints:
            node_constraints = constraints[node_value]
            if not isinstance(node_constraints, (list, tuple, set)):
                constraints[node_value] = (node_constraints,)
        return constraints

    def __or__(self, other):
        return Rule(self, other)


class PymLizParser(Parser):
    """
    Parser class for object creation
    
    -- Parameters --
        *args: Unpacked list of arbitrary objects
    
    -- Attributes --
        graph: networkx Digraph representing the object
    """

    def __init__(self, *args) -> None:
        self.args = args
        self.graph = generate_graph_simple(args)

    def copy(self):
        return PymLizParser(*self.args)


class Predicate:
    """
    Predicate class implementing DSL constraints
    """
    def __init__(self, name: str, func: Callable) -> None:
        if not isinstance(name, str):
            raise ParsingError("Predicate's name must be a string")
        if not isinstance(func, Callable):
            raise ParsingError("Predicate's function must be a Callable")
        self.type = {name: func}


def _get_constraint_name(constraint: str | type) -> str:
    """
    Get the constraint name representation of an object to use as a node's constraint
    """
    if isinstance(constraint, str):
        return constraint
    elif isinstance(constraint, type):
        return f"__pymeleon_constraint_name_{constraint.__module__}_{constraint.__name__}"
    else:
        raise ParsingError("Attempted to get constraint name for a non str or type object")
    

def _get_constraints_name_dict(constraints: dict):
    """
    Given a constraints dict, returns a dict that maps any [k: str, v: str|type] pair to a [k: str, v: str] pair
    """
    return {node_name: tuple(map(_get_constraint_name, constraint_types)) 
            for node_name, constraint_types in constraints.items()}
    
def _get_constraints_func_dict(constraints: dict):
    """
    Given a constraints dict, returns a dict that maps any [k: str, constraint_types: tuple[str|type]] pair to 
    a [constraint_type_name: str, lambda x: isinstance(x, constraint_type)] pair
    """
    return {_get_constraint_name(constraint_type): lambda x: isinstance(x, constraint_type)
            for constraint_types in constraints.values()
            for constraint_type in constraint_types
            if isinstance(constraint_type, type)}
    
    
def parse(*args, constraints: dict = None) -> RuleParser:
    """
    Parses a series of objects or expression for Rule creation
    
    -- Arguments --
        *args: Must be either a single dict or type argument, or an arbitrary number of str arguments followed by \
               a dict
    """
    if constraints is None:
        try:
            if isinstance(args[-1], dict):
                constraints = args[-1]
                args = args[:-1]
            else:
                constraints = dict()
        except IndexError:
            return RuleParser("")
    for k, v in constraints.items():
        if not isinstance(v, list | tuple | set):
            constraints[k] = (v,)
    if not all(map(lambda x: isinstance(x[0], str) and all(map(lambda y: isinstance(y, str | type), x[1])), 
                    constraints.items())):
        raise ParsingError("parse() constraints must be a dict['str', 'str'|'type']")
    if (not args) and constraints:
        return RuleParser(*constraints,
                          constraints=_get_constraints_name_dict(constraints),
                          constraints_func_dict=_get_constraints_func_dict(constraints))
    elif len(args) == 1 and not constraints:
        if isinstance(args[0], type):
            constraint_name = _get_constraint_name(args[0])
            return RuleParser(RuleParser.DEFAULT_SINGLE_INPUT_NAME, 
                              constraints={RuleParser.DEFAULT_SINGLE_INPUT_NAME: constraint_name},
                              constraints_func_dict={constraint_name: lambda x: isinstance(x, args[0])})
        raise ParsingError("When providing parse() with a single argument, it must be a type or a constraints dict")
    if (not all(map(lambda x: isinstance(x, str), args))):
        raise ParsingError("parse() takes an arbitrary number of str arguments and an optional dict mapping any \
                            of them to a type or str")
    return RuleParser(*args, 
                      constraints=_get_constraints_name_dict(constraints),
                      constraints_func_dict=_get_constraints_func_dict(constraints))
    