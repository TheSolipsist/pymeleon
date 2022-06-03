from viewer.viewer import Viewer
from object.object import PymLiz
from language.parser import Parser
from language.language import Language
from random import choice
from utilities.util_funcs import save_graph

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
        for _ in range(100):
            chosen_rule = choice(rules)
            transform_dicts = tuple(obj.search(chosen_rule))
            if not transform_dicts:
                continue
            chosen_transform_dict = choice(transform_dicts)
            obj.apply(chosen_rule, chosen_transform_dict, inplace=True)
            save_graph(obj._graph, print=True)
            result = obj.run()
            if result is not None:
                return result
            