from viewer.viewer import Viewer
from object.object import PymLiz
from language.parser import Parser
from language.language import Language
from random import choice

class BreakFromLoop(Exception):
    """
    Exception to break from an outer loop
    """

class RandomViewer(Viewer):
    """
    Random viewer class, implementing random selection and application of Rules
    
    -- Parameters --
        language(Language): the language object from which to find Rules
        
    -- Attributes --
        language(Language): the viewer's language
        
    -- Methods --
        blob(*args): creates and returns the PymLiz object
        view(): returns the object after having changed it according to the viewer's function
    """
    def __init__(self, language: Language, modules: dict=None):
        super().__init__(language)
        self.modules=modules
    
    def blob(self, *args):
        obj = PymLiz(self, Parser(*args, mode="PYMLIZ"), constraint_types=self.language.types, modules=self.modules)
        return obj
        
    def view(self, obj: PymLiz):
        rules = self.language.rules
        num_rules = len(rules)
        try:
            while True:
                i = 0
                for chosen_rule in rules:
                    transform_dicts = tuple(obj.search(chosen_rule))
                    if not transform_dicts:
                        i += 1
                        if i == num_rules:
                            print("No more rules to be applied, returning object")
                            raise BreakFromLoop
                        else:
                            continue
                    chosen_transform_dict = choice(transform_dicts)
                    obj.apply(chosen_rule, chosen_transform_dict, inplace=True)
                    print(obj.run())
        except BreakFromLoop:
            return obj.run()
        