"""
Module for training examples generation
"""
from language.language import Language
from language.parser import Node
from networkx import DiGraph
import itertools
from random import choice
from language.rule import Rule
from language.rule_search import RuleSearch
from itertools import combinations_with_replacement


def node_representation(node: Node, constraint_types: dict) -> list:
    """
    Returns a node representation

    A node is represented by a (len(language.types) + 1)-int vector of its order and constraint types
    """
    representation = []
    for constraint_type in constraint_types:
        representation.append(int(constraint_type in node.constraints))
    return representation


def dfs_component_representation_rec(graph: DiGraph, root_node: Node, constraint_types: dict,
                                     visited_nodes: list, representation: list) -> None:
    """
    DFS representation for connected components (recursive)

    WARNING: Assuming that all sibling nodes are connected to their predecessor with an "order" label that is valid
    (numbers that collectively comprise the list [1, 2, ..., num_siblings])
    """
    suc_nodes = sorted(graph.successors(root_node), key=lambda suc_node: graph[root_node][suc_node]["order"])
    for node in suc_nodes:
        if node in visited_nodes:
            continue
        visited_nodes.append(node)
        representation.append(graph[root_node][node]["order"])
        representation.extend(node_representation(node, constraint_types))
        dfs_component_representation_rec(graph, node, constraint_types, visited_nodes, representation)


def dfs_representation(graph: DiGraph, language: Language) -> list:
    """
    Returns a Depth First Search representation of the graph

    In order to assure uniqueness of each representation, a hash of the connected components of the graph is generated
    and the order in which the representation of each component is added to the final representation is dictated by the
    order of the aforementioned hashes
    """
    constraint_types = language.types
    components = []
    for node in graph.successors("root_node"):
        # Successors of the root_node are to be considered to have order "0"
        component_representation = [0] + node_representation(node, constraint_types)
        visited_nodes = []
        dfs_component_representation_rec(graph, node, constraint_types, visited_nodes, component_representation)
        components.append(component_representation)
    components.sort(key=lambda component: hash_graph_representation(component, len(constraint_types)))
    dfs_list = list(itertools.chain.from_iterable(components))
    return dfs_list


def hash_graph_representation(representation: list, num_constraint_types: int) -> int:
    """
    Returns a unique hash for a connected graph's representation
    """
    graph_hash = 0
    node_vector_length = num_constraint_types + 1  # Includes the "order" int
    base = max(representation) + node_vector_length
    for exponent in range(len(representation) // node_vector_length):
        for i in range(node_vector_length):
            graph_hash += (base ** exponent) * representation[exponent * node_vector_length + i] + 1
    return graph_hash


def rule_representation(rule: Rule, language_rules: list) -> list:
    """
    Returns a Rule representation

    A Rule is represented as a one-hot vector in the domain of a language's rules
    """
    representation = []
    for lang_rule in language_rules:
        representation.append(int(rule is lang_rule))
    return representation


def generate_graph_from_constraint_types(constraint_types: tuple[str]):
    """
    Generates a graph that is composed of a tuple of constraint types
    """
    graph = DiGraph()
    for constraint_type in constraint_types:
        # A node's constraints are a set of constraint types
        graph.add_edge("root_node", Node(constraints={constraint_type}), order=-1)
    return graph


def generate_initial_graph_list(constraint_types: dict):
    """
    Generates the initial graph list

    The initial graph list consists of all combinations (with replacement) of the constraint types
    defined in a language of length 1, 2, ..., len(constraint_types).
    """
    initial_graph_list = []
    for r in range(1, len(constraint_types) + 1):
        initial_graph_list.extend(map(generate_graph_from_constraint_types,
                                      combinations_with_replacement(constraint_types, r)))
    return initial_graph_list


def generate_training_examples(language: Language, n_gen: int) -> list:
    """
    Generates training examples for a given language
    """
    rule_search = RuleSearch()
    lang_rules = language.rules
    graph_list = generate_initial_graph_list(language.types)
    for i in range(n_gen):
        pass




