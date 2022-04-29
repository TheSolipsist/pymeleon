import networkx as nx

class Wrapper:
    """
    Wrapper around any Python object that holds a value
    """
    def __init__(self, value) -> None:
        self.value = value

class Node(Wrapper):
    """
    Node objects for usage in the graph
    """
    def copy(self):
        return Node(self.value)

class Expression(Wrapper):
    pass

class Parser:
    """
    Parser for graph generation from Python expressions
    
    -- Parameters --
        expression: String containing the Python expression
        constraints: Dictionary mapping expression variables to constraints (e.g. type)
    
    -- Attributes --
        graph: The generated networkx graph
        variables_constants: Set containing all variables and constants in the expression
        functions: Set containing all functions in the expression
    """
    # The following operators should be written in reverse order of execution hierarchy (e.g. + is higher than *)
    SUPPORTED_OPERATORS = {
        "-": "sub",
        "+": "add",
        "@": "matmul",
        "%": "mod",
        "//": "floordiv",
        "*": "mul",
        "/": "truediv",
        "^": "power"
    }
    
    def __init__(self, expression, constraints) -> None:
        # In order to avoid any parsing errors with "**" and "*", exponentiation is represented by "^"
        self._expression = expression.replace(" ", "").replace("**", "^")
        self.constraints = constraints
        self.variables_constants, self.functions = set(), set()
        self.graph = nx.DiGraph(name=self._expression)
        try:
            self._generate_graph()
        except ValueError:
            print("Inappropriate expression. Refer to the parser's requirements.")
    
    # --- Private methods and classes ---
        
    def _generate_graph(self):
        """
        Generates the expression's graph
        """
        def generate_subgraph(root_node, arguments_list):
            arg_iter = enumerate(arguments_list)
            functions_found = 0
            for i, item in arg_iter:
                item_node = Node(item)
                isfunction = item in self.functions
                self.graph.add_node(item_node)
                self.graph.add_edge(root_node, item_node, order=i - functions_found + 1)
                if isfunction:
                    functions_found += 1
                    next(arg_iter)
                    generate_subgraph(item_node, arguments_list[i + 1])

        if self._expression:
            graph_list = self._parse_expression()
            self.graph.add_node("root_node")
            generate_subgraph("root_node", graph_list)
        else:
            print("WARNING: Empty expression parsed")

    def _parse_expression(self):
        """
        Parses the expression and returns a graph representation list
        """
        def disassemble_expression(expression):
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
                            self.functions.add(func)
                    # Hopefully there are no naked operators or commas after a left bracket, so, since the last item
                    # is not a function, it must be a variable or a constant
                    self.variables_constants.add(split_item[-1])
                else:
                    self.variables_constants.add(item)
        
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
                            expr_obj.value = expr_obj.value[:starting_position - 1] + new_expression + expr_obj.value[i + 1:]
                            # The next character to be checked should be the character after ")"
                            return starting_position - 2 + len(new_expression)
                        elif expr_obj.value[starting_position - 2] not in Parser.SUPPORTED_OPERATORS:
                            args_list.append(curr_str)
                            new_expression = "(" + ",".join(args_list) + ")"
                            expr_obj.value = expr_obj.value[:starting_position - 1] + new_expression + expr_obj.value[i + 1:]
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
        
        expr_obj = Expression(self._expression)
        operators_to_functions(expr_obj)
        disassemble_expression(expr_obj.value)
        brackets_list = parse_brackets_rec(expr_obj)
        return brackets_list
        