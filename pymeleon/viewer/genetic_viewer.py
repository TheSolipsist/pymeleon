from pymeleon.dsl.rule import Rule
from pymeleon.viewer.viewer import Viewer
from pymeleon.object.object import PymLiz
from pymeleon.dsl.parser import Node, PymLizParser, RuleParser, parse
from pymeleon.dsl.dsl import DSL
from random import choice
from pymeleon.dsl.rule_search import RuleSearch
from pymeleon.viewer.fitness import FitnessHeuristic, FitnessNeuralNet
import networkx as nx
from pymeleon.neural_net.training_generation import get_top_nodes_graph
from pymeleon.utilities.util_funcs import save_graph
from typing import Any


class ViewerError(Exception):
    pass


def _find_possible_matches(graph: nx.DiGraph, target_graph: nx.DiGraph) -> nx.Graph:
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
            if _check_graph_match_rec(graph, target_graph, root_node, target_root_node):
                G.add_edge(target_root_node, root_node)
        if G.degree(target_root_node) == 0:
            return False
    return G


def _check_graph_match_rec(graph: nx.DiGraph, target_graph: nx.DiGraph, root_node: Node, target_root_node: Node) -> bool:
    if not target_root_node.constraints.issubset(root_node.constraints):
        return False
    # Successor nodes are ordered by their "order" edge attribute in relation to their root node
    target_suc_nodes = sorted(tuple(target_graph.successors(target_root_node)),
                              key=lambda node: target_graph[target_root_node][node]["order"])
    # If there are no more successor nodes in the target graph, we found everything we needed
    if not target_suc_nodes:
        return True
    suc_nodes = sorted(tuple(graph.successors(root_node)),
                       key=lambda node: graph[root_node][node]["order"])
    if len(suc_nodes) != len(target_suc_nodes):
        return False
    for suc_node, target_suc_node in zip(suc_nodes, target_suc_nodes):
        if (not _check_graph_match_rec(graph, target_graph, suc_node, target_suc_node)):
            return False
    return True


def _check_graph_match(graph: nx.DiGraph, target_graph: nx.DiGraph) -> bool:
    """
    Checks if graph "contains" target_graph, meaning that target_graph and graph are isomorphic and
    each node in target_graph has constraints that are a subset of those in graph.
    graph's nodes may have children that target_graph's nodes don't have, meaning that target_graph can be a
    top node representation of graph

    Returns:
        bool: Specifies whether graph contains target_graph
    """
    if graph.out_degree("root_node") != target_graph.out_degree("root_node"):
        return False
    G = _find_possible_matches(graph, target_graph)
    if not G:
        return False
    maximum_match = nx.bipartite.maximum_matching(G, top_nodes=graph.successors("root_node"))
    return len(maximum_match) == len(G)
    
class GeneticViewer(Viewer):
    """
    Genetic viewer class, implementing genetic selection and application of Rules

    -- Parameters --
        (optional) DSL(DSL): The DSL object from which to find Rules

    -- Attributes --
        DSL(DSL): The viewer's DSL

    -- Methods --
        blob(*args): Creates and returns the PymLiz object
        view(): Returns the object after having changed it according to the viewer's function
        search(rule, obj): Iterates through the possible subgraphs (in the form of transform_dicts) that match
            a rule's input graph
    """

    def __init__(self,
                 ext: set | list | dict = None,
                 dsl: DSL = None,
                 n_iter: int = 20,
                 n_gen: int = 15,
                 n_fittest: int = 20,
                 fitness: str = "neural_random",
                 device_str: str = "cpu",
                 use_pretrained: bool = True,
                 hyperparams: dict = None
                 ) -> None:
        self._RuleSearch = RuleSearch()
        if ext is None:
            ext = dict()
        if isinstance(ext, list) or isinstance(ext, set):
            ext = {external.__name__: external for external in ext}
        self.ext = ext
        self.n_iter = n_iter
        self.n_gen = n_gen
        self.n_fittest = n_fittest
        self.fitness_str = fitness
        self.device_str = device_str
        self.use_pretrained = use_pretrained
        self.hyperparams = hyperparams
        if dsl is not None:
            self.add_dsl(dsl)

    def add_dsl(self, dsl: DSL):
        self.dsl = dsl
        if self.fitness_str == "neural_random":
            self.fitness_obj = FitnessNeuralNet(dsl=dsl, 
                                                hyperparams=self.hyperparams, 
                                                device_str=self.device_str,
                                                training_generation="random", 
                                                use_pretrained=self.use_pretrained,)
        elif self.fitness_str == "neural_exhaustive":
            self.fitness_obj = FitnessNeuralNet(dsl=dsl, 
                                                hyperparams=self.hyperparams, 
                                                device_str=self.device_str, 
                                                training_generation="exhaustive", 
                                                use_pretrained=self.use_pretrained,)
        elif self.fitness_str == "heuristic":
            self.fitness_obj = FitnessHeuristic()
        else:
            raise ViewerError(f"Unsupported genetic viewer fitness mode: '{self.fitness_str}'. Supported modes: ('neural_random', 'neural_exhaustive', 'heuristic')")
        self.fitness = self.fitness_obj.fitness_score
        
    def blob(self, *args):
        """
        Creates and returns the PymLiz object
        """
        obj = PymLiz(self, PymLizParser(*args), constraint_types=self.dsl.types, ext=self.ext | self.dsl.ext)
        return obj

    def view(self, obj: PymLiz, parser_obj: Any):
        """
        Returns the object's output after having changed it according to the viewer's function
        """
        if not isinstance(parser_obj, RuleParser):
            parser_obj = parse(parser_obj)
        target_graph = parser_obj.graph
        rules = self.dsl.rules
        n_iter = self.n_iter
        n_fittest = self.n_fittest
        for i_iter in range(n_iter):
            obj_list = [obj.copy() for __ in range(n_fittest)]
            for i_gen in range(self.n_gen):
                for i in range(n_fittest):
                    current_obj = obj_list[i]
                    chosen_rule = choice(rules)
                    transform_dicts = tuple(self.search(chosen_rule, current_obj))
                    if not transform_dicts:
                        obj_copy = current_obj.copy()
                        obj_list.append(obj_copy)
                        continue
                    chosen_transform_dict = choice(transform_dicts)
                    new_obj = current_obj.apply(chosen_rule, chosen_transform_dict)
                    obj_list.append(new_obj)
                    if _check_graph_match(get_top_nodes_graph(new_obj.get_graph()), get_top_nodes_graph(target_graph)):
                        return (new_obj.run(), i_iter + 1)
                obj_list.sort(
                    key=lambda x: self.fitness(x.get_graph(), target_graph), reverse=True)
                del obj_list[n_fittest:]
        raise ViewerError("Desired object could not be generated")

    def search(self, rule: Rule, obj: PymLiz):
        """
        Iterates through the possible subgraphs (in the form of transform_dicts) that match a rule's input graph
        """
        return self._RuleSearch(rule, obj._graph)

    def __rrshift__(self, left: DSL):
        self.add_dsl(left)
        return self

    def __lshift__(self, right: DSL):
        self.add_dsl(right)
        return self

    def __call__(self, *args):
        return self.blob(*args)