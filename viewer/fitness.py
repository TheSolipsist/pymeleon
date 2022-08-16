"""
Fitness module for the genetic viewer
"""
# pymeleon modules
from language.language import Language
from neural_net.neural_net import NeuralNet, NeuralNetError
from neural_net.training_generation import TrainingGeneration, TrainingGenerationRandom
# networkx modules
from networkx import DiGraph
# Python Standard Library modules
import math


class Fitness:
    """
    ### Abstract fitness class for use with the GeneticViewer

    #### Methods:
        ``fitness_score(graph_before, graph_after, graph_final)``: Returns a fitness score (0-1) for a \
                (graph_before, graph_after, graph_final) sequence
    """
    def __init__(self, *args):
        pass
    
    def fitness_score(self, graph: DiGraph, target_graph: DiGraph) -> float:
        raise NotImplemented("'fitness_score' method not implemented")
    

class FitnessError(Exception):
    pass


class RecursionObject:
    """
    Object to be used for recursion (to pass less arguments in each recursion step)
    """

    def __init__(self):
        pass
    
    
class FitnessHeuristic(Fitness):
    """
    ### Heuristic fitness class for use with the GeneticViewer

    #### Methods:
        ``fitness_score(graph_before, graph_after, graph_final)``: Returns a fitness score (0-1) for a \
                (graph_before, graph_after, graph_final) sequence using a heuristic algorithm
    """
    def __init__(self):
        pass
        
    def _calculate_target_penalty(self, target_graph):
        return math.log(sum((target_graph.in_degree(node) ** 2 for node in target_graph)))

    def _check_graph_match_rec(self, wrapper_obj: RecursionObject, root_node, target_root_node):
        graph = wrapper_obj.graph
        target_graph = wrapper_obj.target_graph
        # Successor nodes are ordered by their "order" edge attribute in relation to their root node
        target_suc_nodes = sorted(list(target_graph.successors(target_root_node)),
                                  key=lambda node: target_graph[target_root_node][node]["order"])
        # If there are no more successor nodes in the target graph, we found everything we needed
        if not target_suc_nodes:
            return True
        suc_nodes = sorted(list(graph.successors(root_node)),
                           key=lambda node: graph[root_node][node]["order"])
        if len(suc_nodes) != len(target_suc_nodes):
            return False
        for suc_node, target_suc_node in zip(suc_nodes, target_suc_nodes):
            if (not target_suc_node.constraints.issubset(suc_node.constraints) or
                    not self._check_graph_match_rec(wrapper_obj, suc_node, target_suc_node)):
                return False
        return True

    def _calculate_regularized_score(self, graph, score, num_of_root_successors, target_penalty):
        score /= num_of_root_successors
        penalty = max(0, sum((graph.in_degree(node) ** 2 for node in graph)) - target_penalty)
        score -= penalty * self.penalty_coefficient
        return score
    
    def fitness_score(self, graph, target_graph, target_penalty):
        """
        Fitness function for the genetic algorithm
        
        Checks if the desired graph structure is found in each of the components of the graph. The score starts as
        1 if at least 1 component follows the desired graph structure (otherwise 0), gets divided by the number
        of connected components and is penalized by the total number of incoming edges squared for each node (times
        a parameter)
        """
        wrapper_obj = RecursionObject()
        wrapper_obj.graph = graph
        wrapper_obj.target_graph = target_graph
        score = 0
        root_successors = tuple(graph.successors("root_node"))
        target_root_successor = next(target_graph.successors("root_node"))
        for node in root_successors:
            if self._check_graph_match_rec(wrapper_obj, node, target_root_successor):
                score = 1
                break
        score = self._calculate_regularized_score(graph, score, len(root_successors), target_penalty)
        return score
    

class FitnessNeuralNet(Fitness):
    """
    ### Neural net fitness class for use with the GeneticViewer

    #### Parameters
        ``language``: Language to be used
        ``n_gen``: Number of consecutive rules to be applied to the initial graphs when generating
            the training data
        ``n_items``: Maximum number of items to create initial graphs from when generating the training data
        ``lr``: Learning rate
        ``num_epochs``: Number of epochs to iterate through while training the network
        ``batch_size``: The batch size to use while iterating through the training and testing data
        ``num_classes``: The number of labels for each data instance
        ``device_str``: The name of the device on which to train the model and make any predictions
        ``training_generation_class``: The class to use for training example generation
        
    #### Methods:
        ``fitness_score(graph, target_graph)``: Returns a fitness score (0-1) for a (graph_before, graph_after, \
                graph_final) sequence using the neural network's predictions, in which graph_before is \
                ``initial_graph``, graph_after is ``graph`` and graph_final is ``target_graph``
    """
    def __init__(self,
                 language: Language,
                 hyperparams: dict = None,
                 device_str: str = "cpu",
                 training_generation: str = "random"
                 ) -> None:
        self.initial_graph = None
        self.model = NeuralNet(language=language,
                               hyperparams=hyperparams,
                               device_str=device_str,
                               training_generation=training_generation)
    
    def fitness_score(self, graph: DiGraph, target_graph: DiGraph) -> float:
        """
        ### Returns the fitness score of the current graph

        #### Arguments:
            ``graph`` (DiGraph): The graph currently being examined
            ``target_graph`` (DiGraph): The final graph which we are trying to reach

        #### Returns:
            ``float``: Prediction (0-1) assessing how close ``graph`` is to ``target_graph``
        """
        try:
            return self.model.predict(graph, target_graph)
        except NeuralNetError:
            return -1
