from language.rule import Rule
from object.object import PymLiz

class Viewer:
    """
    Abstract viewer class
    
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
    def __init__(self, language):
        self.language = language
    
    def blob(self, *args):
        raise NotImplementedError("'blob' method not implemented")
    
    def view(self, *args):
        raise NotImplementedError("'view' method not implemented")
    
    def search(self, rule: Rule, obj: PymLiz):
        raise NotImplementedError("'search' method not implemented")
    