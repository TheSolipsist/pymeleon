from pymeleon.dsl.rule import Rule
from pymeleon.viewer.viewer import Viewer
from pymeleon.object.object import PymLiz
from pymeleon.dsl.parser import PymLizParser
from pymeleon.dsl.dsl import DSL
from random import choice
from pymeleon.utilities.util_funcs import save_graph
from pymeleon.dsl.rule_search import RuleSearch


class RandomViewer(Viewer):
    """
    Random viewer class, implementing random selection and application of Rules
    
    -- Parameters --
        DSL(DSL): The DSL object from which to find Rules
        
    -- Attributes --
        DSL(DSL): The viewer's DSL
        
    -- Methods --
        blob(*args): Creates and returns the PymLiz object
        view(): Returns the object after having changed it according to the viewer's function
        search(rule, obj): Iterates through the possible subgraphs (in the form of transform_dicts) that match 
            a rule's input graph
    """

    def __init__(self, dsl: DSL, ext: dict = None) -> None:
        self.dsl = dsl
        self._RuleSearch = RuleSearch()
        self.ext = ext

    def blob(self, *args) -> PymLiz:
        """
        Creates and returns the PymLiz object
        """
        obj = PymLiz(self, PymLizParser(*args), constraint_types=self.dsl.types, ext=self.ext)
        return obj

    def view(self, obj: PymLiz):
        """
        Returns the object after having changed it according to the viewer's function
        """
        rules = self.dsl.rules
        for _ in range(100):
            chosen_rule = choice(rules)
            transform_dicts = tuple(self.search(chosen_rule, obj))
            if not transform_dicts:
                continue
            chosen_transform_dict = choice(transform_dicts)
            obj.apply(chosen_rule, chosen_transform_dict, inplace=True)
            save_graph(obj._graph, print=True)
            result = obj.run()
            if not isinstance(result, list):
                return result

    def search(self, rule: Rule, obj: PymLiz):
        """
        Iterates through the possible subgraphs (in the form of transform_dicts) that match a rule's input graph
        """
        return self._RuleSearch(rule, obj._graph)
