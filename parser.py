import networkx as nx
import re

class Parser:
    """
    Parser for graph generation from Python expressions
    
    -- Parameters --
        expression: String containing the Python expression
        type_dict: Dictionary mapping expression variables to constraints (e.g. type)
    
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
        "//": "div_int",
        "/": "div",
        "*": "mul",
        "$": "power"
    }
    
    ORDERED_OPERATORS = [
        "-", "/", "//", "$", "%", "@"
    ]
    
    def __init__(self, expression, type_dict):
        # In order to avoid any parsing errors with "**" and "*", exponentiation is represented by "$"
        self._expression = expression.replace(" ", "").replace("**", "$")
        self._type_dict = type_dict
        self.variables_constants, self.functions = set(), set()
        self.graph = nx.DiGraph()
        try:
            self._generate_graph()
        except ValueError:
            print("Inappropriate expression. Refer to the parser's requirements.")
    
    # --- Private methods ---
    
    def _generate_graph(self):
        graph_list = self._parse_expression()
        print(graph_list)

    def _parse_expression(self):
        """Parse the expression given to the parser and return a graph representation list

        Returns:
            graph_list: 
        """
        
        class expression():
            """
            Wrapper around expression to be parsed, used to pass reference to expression object and change it.
            """
            def __init__(self, expression):
                self.value = expression
        
        def disassemble_expression(expression):
            for operator in self.SUPPORTED_OPERATORS:
                expression = expression.replace(operator, " ")
            expression = expression.replace(")", "").replace(",", " ").split(" ")
            for item in expression:
                if "(" in item[1:]:
                    split_item = item.split("(")
                    # Last item in split_item doesn't have a left bracket on its right, so it is not a function
                    for func in split_item[:-1]:
                        if func:
                            self.functions.add(func)
                    # Hopefully there are no naked operators or commas after a left bracket, so, since there
                    # is no function, there must be a variable or a constant
                    self.variables_constants.add(split_item[-1])
                else:
                    self.variables_constants.add(item)
        
        def operators_to_functions(expression):
            for operator in self.SUPPORTED_OPERATORS:
                pass


        def add_from_brackets(expr_obj, bracket_list, bracket_i):
            if bracket_i > 0:
                str_to_add = expr_obj.value[:bracket_i]
                for item in str_to_add.split(","):
                    if item:
                        bracket_list.append(item)
            expr_obj.value = expr_obj.value[bracket_i + 1:]
        
        def parse_brackets_rec(expr_obj):
            bracket_contents = []
            while (l_bracket := expr_obj.value.find("(")) < (r_bracket := expr_obj.value.find(")")) and l_bracket != -1:
                add_from_brackets(expr_obj, bracket_contents, l_bracket)
                bracket_contents.append(parse_brackets_rec(expr_obj))
            if r_bracket != -1:
                add_from_brackets(expr_obj, bracket_contents, r_bracket)
            else:
                bracket_contents.append(expr_obj.value)
            return bracket_contents

        def handle_functions(root_node, expression):
            for func in self.functions:
                

        disassemble_expression(self._expression)
        expr_obj = expression(self._expression)
        brackets_list = parse_brackets_rec(expr_obj)
        return brackets_list
        

