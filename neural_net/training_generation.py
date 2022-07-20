"""
Module for training examples generation
"""
from language.language import Language
from language.parser import Node
from networkx import DiGraph
import itertools
from random import choice, choices
from language.rule import Rule
from language.rule_search import RuleSearch
from itertools import combinations_with_replacement, permutations


class TrainingGenerationError(Exception):
    pass


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
    order of the hashes
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


def create_rule_representations(language_rules: list[Rule]) -> dict:
    """
    Creates the Rule representations dictionary

    A Rule is represented as a one-hot vector in the domain of a language's rules
    """
    representations = dict()
    for lang_rule_i in language_rules:
        rule_representation = []
        for lang_rule_to_check in language_rules:
            rule_representation.append(int(lang_rule_i is lang_rule_to_check))
        representations[lang_rule_i] = rule_representation
    return representations


def generate_graph_from_constraint_types(constraint_types: tuple[str]):
    """
    Generates a graph that is composed of a tuple of constraint types
    """
    graph = DiGraph()
    for constraint_type in constraint_types:
        # A node's constraints are a set of constraint types
        graph.add_edge("root_node", Node(constraints={constraint_type}), order=-1)
    return graph


def generate_initial_graph_list(constraint_types: dict, n_items: int = None):
    """
    Generates the initial graph list

    The initial graph list consists of all combinations (with replacement) of the constraint types
    of length 1, 2, ..., n_items.
    """
    initial_graph_list = []
    if n_items is None:
        n_items = len(constraint_types)
    for r in range(1, n_items + 1):
        initial_graph_list.extend(map(generate_graph_from_constraint_types,
                                      combinations_with_replacement(constraint_types, r)))
    return initial_graph_list



def apply_rules(graph: DiGraph, chosen_rules: list[Rule], rule_search: RuleSearch) -> list:
    """
    Applies a list of rules (with random transform_dicts) to a graph
    """
    sequence = [graph]
    for rule in chosen_rules:
        transform_dicts = tuple(rule_search(rule, graph))
        if not transform_dicts:
            continue
        chosen_transform_dict = choice(transform_dicts)
        graph = rule.apply(graph, chosen_transform_dict)
        sequence.append(graph)
    return sequence
    
    
def generate_sequence(initial_graph: DiGraph, chosen_rules: list[Rule], chosen_bad_rules: list[Rule],
                      rule_search: RuleSearch) -> tuple[list, list]:
    """
    Generates a (Graph_1, Graph_2, ..., Graph_final) positive sequence based on an initial graph and a sequence of rules,
    along with its negative sequence which results from the application of a different sequence of rules but has the same Graph_final

    If no rules were applied, returns None for the corresponding sequence
    """
    positive_sequence = apply_rules(initial_graph, chosen_rules, rule_search)
    negative_sequence = apply_rules(initial_graph, chosen_bad_rules, rule_search)
    negative_sequence[-1] = positive_sequence[-1]
    if len(positive_sequence) == 1:
        return (None, None)
    elif len(negative_sequence) == 1:
        return positive_sequence, None
    return positive_sequence, negative_sequence


def add_sequence_to_training_examples(sequence: list, training_examples: dict[str, list], language: Language, label: bool):
    """
    Adds a sequence of (Graph_1, Rule_1, Graph_2, ..., Graph_final) in the representation required to be used in a neural network to the
    training examples dictionary
    """
    graph_before_repr = dfs_representation(sequence[0], language)
    graph_final_repr = dfs_representation(sequence[-1], language)
    for i in range(1, len(sequence) - 1):
        graph_after_repr = dfs_representation(sequence[i], language)
        training_examples["graph_before"].append(graph_before_repr)
        training_examples["graph_after"].append(graph_after_repr)
        training_examples["graph_final"].append(graph_final_repr)
        training_examples["label"].append(label)
        graph_before_repr = graph_after_repr


def fix_length_training_examples(training_examples: dict[str, list]) -> dict[str, int]:
    """
    Fixes the graph representations in the training_examples to have a fixed length (equal to the length of the
    longest representation) by padding zeros to the right of each graph representation.

    -- Returns --
        max_length: The max length found in the training examples (required to prepare data for forward propagation in the ANN)
    """
    graphs_keys = ["graph_before", "graph_after", "graph_final"]
    max_len = float("-inf")
    for graphs_key in graphs_keys:
        max_len = max(len(max(training_examples[graphs_key], key=len)), max_len)
    for graphs_key in graphs_keys:
        for representation in training_examples[graphs_key]:
            needed_zeros_num = max_len - len(representation)
            representation.extend([0] * needed_zeros_num)
    return max_len


def remove_duplicates(data: list[list[int]], labels: list[int]):
    """
    Removes duplicates from training examples

    If labels are conflicted, keeps the training example as positive
    """
    indices_to_delete = []
    found_graphs = dict()
    for i, training_example in enumerate(data):
        graph_str = ''.join(map(str, training_example))
        label = labels[i]
        if graph_str in found_graphs:
            if label != found_graphs[graph_str]["label"]:
                # If both 1 and 0 have been found, keep the example as positive (1)
                first_occurrence_index = found_graphs[graph_str]["index"]
                labels[first_occurrence_index] = 1
            indices_to_delete.append(i)
        else:
            found_graphs[graph_str] = {"label": label,
                                       "index": i}
    for index in indices_to_delete[::-1]:
        # Deleting in reverse order to not shift the position of the actual elements of the rest of the indices
        del data[index]
        del labels[index]


def generate_training_examples(language: Language, n_gen: int = 5, n_items: int = None) -> tuple[list, list, dict]:
    """
    Generates training examples for a given language

    -- Arguments --
        language: The language for which to generate training examples
        n_gen: The number of rules to apply to each initial graph of constraint types while generating training
            sequences
        n_items: The range of items to use when generating the initial graphs of constraint types (graphs of
            range(n_items) nodes will be generated). If not given, defaults to len(language.constraint_types)

    -- Returns --
        data: The list containing the training examples
        labels: The list containing the training labels
        max_length: The max length for "graph_before", "graph_after", "graph_final" representations
    """
    rule_search = RuleSearch()
    rules = language.rules
    initial_graph_list = generate_initial_graph_list(language.types, n_items)
    training_examples = {"graph_before": [],    # Representations of the graphs before the Rule application
                         "graph_after": [],     # Representations of the graphs after the Rule application
                         "graph_final": [],     # Representations of the graphs after all Rule applications
                         "label": []}           # 1 if positive training example, 0 if negative
    
    for graph in initial_graph_list:
        chosen_rules = choices(rules, k=n_gen)
        chosen_negative_rules = choices(rules, k=n_gen)
        while chosen_negative_rules == chosen_rules:
            chosen_negative_rules = choices(rules, k=n_gen)
        positive_sequence, negative_sequence = generate_sequence(graph, chosen_rules, chosen_negative_rules,
                                                                 rule_search)
        if positive_sequence:
            add_sequence_to_training_examples(positive_sequence, training_examples, language, label=1)
        if negative_sequence:
            add_sequence_to_training_examples(negative_sequence, training_examples, language, label=0)
    if not training_examples["graph_before"]:
        raise TrainingGenerationError("Training examples could not be generated")
    max_length = fix_length_training_examples(training_examples)
    data = [graph_before + graph_after + graph_final for graph_before, graph_after, graph_final in
            zip(training_examples["graph_before"],
                training_examples["graph_after"],
                training_examples["graph_final"])]
    labels = training_examples["label"]
    remove_duplicates(data, labels)
    return data, labels, max_length
