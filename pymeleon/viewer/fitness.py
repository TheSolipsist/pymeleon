"""
Fitness module for the genetic viewer
"""
import networkx as nx
import numpy as np
# pymeleon modules
from pymeleon.dsl.dsl import DSL
from pymeleon.neural_net.neural_net import NeuralNet, NeuralNetError
# networkx modules
from networkx import DiGraph


class Fitness:
    """
    Abstract fitness class for use with the GeneticViewer

    Methods:
        fitness_score(graph, target_graph): Returns a float (0-1) assessing how close graph is to target_graph
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
    Heuristic fitness class for use with the GeneticViewer

    Methods:
        fitness_score(graph, targt_graph): Returns a fitness score (0-1)
    """
    def __init__(self, distr_std_prc: float = 0.02):
        """
        Heuristic fitness class for use with the GeneticViewer 

        Args:
            distr_std_prc (float, 0-1): The fraction of the fitness score to be the normal distribution's standard
                deviation, when calculating the noise to add to the fitness score. Defaults to 0.02.
        """
        self.distr_std_prc = distr_std_prc
        
    def _calculate_target_penalty(self, target_graph):
        return np.log(sum((target_graph.in_degree(node) ** 2 for node in target_graph)))

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
    
    def _find_possible_matches(self, graph: nx.DiGraph, target_graph: nx.DiGraph) -> nx.Graph:
        """
        Finds possible matches between the roots of graph and target_graph.

        Args:
            graph (nx.DiGraph): The graph currently being examined.
            target_graph (nx.DiGraph): The final graph which we are trying to reach.

        Returns:
            nx.Graph: Matching between nodes that match in graph and target_graph.
        """
        G = nx.Graph()
        for target_root_node in target_graph.successors("root_node"):
            G.add_node(target_root_node)
            for root_node in graph.successors("root_node"):
                G.add_node(root_node)
                if target_root_node.constraints.issubset(root_node.constraints):
                    G.add_edge(target_root_node, root_node)
        return G
    
    def fitness_score(self, graph: nx.DiGraph, target_graph: nx.DiGraph) -> float:
        """
        Heuristic fitness function for the genetic algorithm
        
        Arguments:
            graph (DiGraph): The graph currently being examined
            target_graph (DiGraph): The final graph which we are trying to reach

        Returns:
            float: Assesses how close graph is to target_graph. Calculated by counting the number of roots
                in graph whose constraints are a superset of the roots' constraints in target_graph. If target_graph has
                more roots than graph, the result is normalized accordingly.
        """
        G = self._find_possible_matches(graph, target_graph)
        matched_nodes = len(nx.bipartite.maximum_matching(G, top_nodes=graph.successors("root_node"))) // 2
        score = matched_nodes / max(graph.out_degree("root_node"), target_graph.out_degree("root_node"))
        score = np.random.normal(score, self.distr_std_prc)
        return score

class FitnessNeuralNet(Fitness):
    """
    Neural net fitness class for use with the GeneticViewer

    Parameters
        DSL: DSL to be used
        n_gen: Number of consecutive rules to be applied to the initial graphs when generating
            the training data
        n_items: Maximum number of items to create initial graphs from when generating the training data
        lr: Learning rate
        num_epochs: Number of epochs to iterate through while training the network
        batch_size: The batch size to use while iterating through the training and testing data
        num_classes: The number of labels for each data instance
        device_str: The name of the device on which to train the model and make any predictions
        training_generation_class: The class to use for training example generation
        
    Methods:
        fitness_score(graph, target_graph): Returns a fitness score (0-1) using the neural network's predictions
    """
    def __init__(self,
                 dsl: DSL,
                 hyperparams: dict = None,
                 device_str: str = "cpu",
                 training_generation: str = "random",
                 use_pretrained: bool = True
                 ) -> None:
        self.initial_graph = None
        self.model = NeuralNet(dsl=dsl,
                               hyperparams=hyperparams,
                               device_str=device_str,
                               training_generation=training_generation,
                               use_pretrained=use_pretrained)
    
    def fitness_score(self, graph: DiGraph, target_graph: DiGraph) -> float:
        """
        Returns the fitness score of the current graph

        Arguments:
            graph (DiGraph): The graph currently being examined
            target_graph (DiGraph): The final graph which we are trying to reach

        Returns:
            float: (0-1) value returned by the neural network, assessing how close graph is to target_graph 
        """
        try:
            return self.model.predict(graph, target_graph)
        except NeuralNetError:
            return -1
