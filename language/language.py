from language.rule import Rule

class LanguageError(Exception):
    pass

class Language:
    """
    Language class for Rule creation
    
    -- Parameters --
        pass
    
    -- Attributes --
        rules: Rules that have been registered in this language
        types: Constraint types that have been registered in this language
    
    -- Methods --
        register(*rules: *[Rule]): Registers the given rules in this language
        register(types: dict): Registers the given constraints (types) in this language
    """
    def __init__(self, rules=None, types=None):
        self.rules = list()
        self.types = dict()
        if rules is not None:
            self.add_rules(rules)
        if types is not None:
            self.add_types(types)
    
    def _add_rule(self, rule: Rule):
        self.rules.append(rule)
    
    def _add_types(self, types: dict):
        self.types.update(types)
    
    def add_rules(self, *rules: Rule):
        for rule in rules:
            if not isinstance(rule, Rule):
                raise TypeError("can only register Rule objects in a language")
            self._add_rule(rule)
    
    def add_types(self, types: dict):
        for type_name, func in types.items():
            if not isinstance(type_name, str) or not hasattr(func, "__call__"):
                raise TypeError("add_types dict must be a type_name to type_function dict") 
        self._add_types(types)
            
        