import torch


class NeuralNet:
    """
    Neural network implementation for usage with the Genetic Viewer as its fitness function
    
    -- Parameters --
        pass
        
    -- Methods --
        pass
    """

    def __init__(self, language):
        self.language = language
        training_data = self._generate_training_sample(language)

    def _generate_training_sample(self, language):
        pass
