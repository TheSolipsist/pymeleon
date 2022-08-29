"""
DSL module for pymeleon
"""

from typing import Callable
from dsl.rule import Rule
from dsl.parser import Predicate

class DSLError(Exception):
    pass


DEFAULT_DSL_NAME = "DEFAULT_DSL__pym"

class DSL:
    """
    DSL class for Rule creation
    
    -- Parameters --
        *args: The first argument can be a str which will be used as the DSL's name. Every other argument \
               must be a Predicate or a Rule
    
    -- Attributes --
        rules: Rules that have been registered in this DSL
        types: Constraint types that have been registered in this DSL
    
    -- Methods --
        add_rules(*rules: *[Rule]): Registers the given rules in this DSL
        add_types(types: dict): Registers the given constraints (types) in this DSL
    
    Example:
        dsl = DSL(
                Predicate("int", int)
                Rule(parse("a", int), parse("float(a)", {float: "float"}))
              )
    """

    def __init__(self, *args, name: str = None):
        self.rules = list()
        self.types = dict()
        self.in_types = dict()
        self.out_types = dict()
        if name:
            if not isinstance(name, str):
                raise DSLError("The DSL constructor requires a 'str' for its 'name' keyword argument")
            self.name = name
        else:
            if isinstance(args[0], str):
                self.name = args[0]
                args = args[1:]
            else:
                self.name = DEFAULT_DSL_NAME
        for arg in args:
            if isinstance(arg, Rule):
                self._add_rule(arg)
                self.in_types |= arg._parser_obj_in.constraints_func_dict
                self.out_types |= arg._parser_obj_out.constraints_func_dict
            elif isinstance(arg, Predicate):
                self.add_types(arg.type)
            else:
                raise DSLError("The DSL constructor receives a str (the DSL's name) as an optional first argument \
                                and every other argument must be a Predicate or a Rule")
        self.types = self.in_types | self.out_types
            
    def _add_rule(self, rule: Rule):
        self.rules.append(rule)

    def _add_types(self, types: dict):
        self.types |= types

    def add_rules(self, *rules: Rule):
        for rule in rules:
            if not isinstance(rule, Rule):
                raise TypeError("can only register Rule objects in a DSL")
            self._add_rule(rule)

    def add_types(self, types: dict):
        for type_name, func in types.items():
            if not isinstance(type_name, str) or not isinstance(func, Callable):
                raise TypeError("add_types dict must be a str to function dict")
        self._add_types(types)
