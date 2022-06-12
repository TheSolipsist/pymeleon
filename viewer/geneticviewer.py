from language.rule import Rule
from viewer.viewer import Viewer
from object.object import PymLiz
from language.parser import RuleParser, PymLizParser
from language.language import Language
from random import choice
from utilities.util_funcs import save_graph
from language.rule_search import RuleSearch

class GeneticViewer(Viewer):
    """
    Genetic viewer class, implementing genetic selection and application of Rules
    
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
    def __init__(self, language: Language, modules: dict=None):
        super().__init__(language)
        self._RuleSearch = RuleSearch()
        self.modules=modules
    
    def blob(self, *args):
        """
        Creates and returns the PymLiz object
        """
        obj = PymLiz(self, PymLizParser(*args), constraint_types=self.language.types, modules=self.modules)
        return obj
    
    def view(self, obj: PymLiz, parser_obj: RuleParser):
        """
        Returns the object after having changed it according to the viewer's function
        """
        rules = self.language.rules
        
    def search(self, rule: Rule, obj: PymLiz):
        """
        Iterates through the possible subgraphs (in the form of transform_dicts) that match a rule's input graph
        """
        return self._RuleSearch(rule, obj._graph)
    
    def fitness(self):
        pass