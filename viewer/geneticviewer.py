from viewer.viewer import Viewer
from object.object import PymLiz
from language.parser import Parser
from language.language import Language
from random import choice
from utilities.util_funcs import save_graph

class GeneticViewer(Viewer):
    """
    Random viewer class, implementing genetic selection and application of Rules
    
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