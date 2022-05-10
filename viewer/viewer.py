class Viewer:
    """
    Abstract viewer class
    
    -- Parameters --
        language(Language): the language object from which to find Rules
        
    -- Attributes --
        language(Language): the viewer's language
        
    -- Methods --
        blob(*args): creates and returns the PymLiz object
        view(): returns the object after having changed it according to the viewer's function
    """
    def __init__(self, language):
        self.language = language
    
    def blob(self, *args):
        raise NotImplementedError("'blob' method not implemented")
    
    def view(self):
        raise NotImplementedError("'view' method not implemented")
    
