import networkx as nx


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
    }
    SUPPORTED_OPERATORS_REVERSE = {v: k for k, v in SUPPORTED_OPERATORS.items()}

    def __init__(self, *args, constraints=None) -> None:
        raise NotImplementedError("Abstract Parser __init__ should not be called")


def fix_constraints(constraints):
    """
    Returns a fixed constraints dict (each value should be an iterable yielding constraint types)
    """
    for node_value in constraints:
        node_constraints = constraints[node_value]
        if not isinstance(node_constraints, (list, tuple, set)):
            constraints[node_value] = (node_constraints,)
    return constraints


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
        graph.add_edge("root_node", Node(item), order=i + 1)
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

    def __init__(self, *args, constraints=None) -> None:
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
            self.constraints = fix_constraints(constraints)
            add_constraints_to_nodes(self.graph, self.constraints)


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


class GeneticParser(RuleParser):
    """
    Parser to use with Genetic (or other) viewers
    """

    def __init__(self, *args, constraints=None):
        super().__init__(*args, constraints=constraints)

    def get_graph(self) -> nx.DiGraph:
        return self.graph
