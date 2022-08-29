from dsl.rule import Rule
from object.object import PymLiz
from dsl.dsl import DSL


class Viewer:
    """
    Abstract viewer class
    
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

    def __init__(self, DSL: DSL):
        self.DSL = DSL

    def blob(self, *args):
        raise NotImplementedError("'blob' method not implemented")

    def view(self, *args):
        raise NotImplementedError("'view' method not implemented")

    def search(self, rule: Rule, obj: PymLiz):
        raise NotImplementedError("'search' method not implemented")
