class LanguageError(Exception):
    pass

class Language:
    """
    Language object for the creation of rules
    
    -- Parameters --
        pass
    
    -- Attributes --
        rules: Rules that have been registered in this language
    
    -- Methods --
        register(rule: Rule): Registers the given rule in this language
    """
    def __init__(self):
        self.rules = []
    
    def _add_rule(self, rule):
        self.rules.append(rule)
    
    def register(self, input, output):
        if (not isinstance(input, (list, tuple, set)) or not isinstance(input[0], str) or not isinstance(input[1], dict)
            or not isinstance(output, (list, tuple, set)) or not isinstance(output[0], str) or not isinstance(output[1], dict)):
            raise ValueError("Registering a rule requires a (str, dict) tuple for the input and output")
        