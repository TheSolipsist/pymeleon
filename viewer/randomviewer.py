from language.rule import Rule
from viewer.viewer import Viewer
from object.object import PymLiz
from language.parser import PymLizParser
from language.language import Language
from random import choice
from utilities.util_funcs import save_graph
from language.rule_search import RuleSearch

class RandomViewer(Viewer):
    """
    Random viewer class, implementing random selection and application of Rules
    
    -- Parameters --
        language(Language): The language object from which to find Rules
        
    -- Attributes --
        language(Language): The viewer's language
        
    -- Methods --
        blob(*args): Creates and returns the PymLiz object
        view(): Returns the object after having changed it according to the viewer's function
        search(rule, obj): Iterates through the possible subgraphs (in the form of transform_dicts) that match 
            a rule's input graph
    """
    def __init__(self, language: Language, modules: dict=None) -> None:
        super().__init__(language)
        self._RuleSearch = RuleSearch()
        self.modules=modules
    
    def blob(self, *args) -> PymLiz:
        """
        Creates and returns the PymLiz object
        """
        obj = PymLiz(self, PymLizParser(*args), constraint_types=self.language.types, modules=self.modules)
        return obj
        
    def view(self, obj: PymLiz):
        """
        Returns the object after having changed it according to the viewer's function
        """
        rules = self.language.rules
        for _ in range(100):
            chosen_rule = choice(rules)
            transform_dicts = tuple(self.search(chosen_rule, obj))
            if not transform_dicts:
                continue
            chosen_transform_dict = choice(transform_dicts)
            obj.apply(chosen_rule, chosen_transform_dict, inplace=True)
            save_graph(obj._graph, print=True)
            result = obj.run()
            if result is not None:
                return result
        
    def search(self, rule: Rule, obj: PymLiz):
        """
        Iterates through the possible subgraphs (in the form of transform_dicts) that match a rule's input graph
        """
        return self._RuleSearch(rule, obj._graph)
            