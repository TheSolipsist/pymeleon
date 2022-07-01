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
from itertools import combinations_with_replacement


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


def generate_sequence(initial_graph: DiGraph, chosen_rules: list[Rule], rule_search: RuleSearch) -> list:
    """
    Generates a (Graph_1, Rule_1, Graph_2, Rule_2, ..., Graph_final) sequence based on an initial graph
    and a sequence of rules

    If no rules were applied, returns None
    """
    sequence = [initial_graph]
    graph = initial_graph
    for rule in chosen_rules:
        transform_dicts = tuple(rule_search(rule, graph))
        if not transform_dicts:
            continue
        chosen_transform_dict = choice(transform_dicts)
        graph = rule.apply(graph, chosen_transform_dict)
        sequence.extend((rule, graph))
    if len(sequence) > 1:
        return sequence


def add_sequence_to_training_examples(sequence: list, training_examples: dict[str, list], language: Language,
                                      rule_representations: dict[Rule, list[int]]):
    """
    Adds a sequence of (Graph, Rule, ..., Graph) in the representation required to be used in a neural network to the
    training examples dictionary
    """
    graphs_before = training_examples["graph_before"]
    rules = training_examples["rule"]
    graphs_after = training_examples["graph_after"]
    labels = training_examples["label"]
    final_graph = sequence[-1]
    for i in range(0, len(sequence) - 1, 2):
        graph_before = sequence[i]
        rule = sequence[i + 1]
        for lang_rule in language.rules:
            graphs_before.append(dfs_representation(graph_before, language))
            rules.append(rule_representations[lang_rule])
            graphs_after.append(dfs_representation(final_graph, language))
            if rule is lang_rule:
                labels.append(1)
            else:
                labels.append(0)


def fix_length_training_examples(training_examples: dict[str, list]) -> dict[str, int]:
    """
    Fixes the "before" and "after" graphs in the training_examples to have a fixed length (equal to the length of the
    longest representation) by padding zeros to the right of each graph representation.

    -- Returns --
        max_length_dict: Dictionary containing the max length found in the training examples for "graph_before",
        "graph_after" (this is used to prepare data for forward propagation in the ANN)
    """
    max_length_dict = dict()
    for graphs_key in ("graph_before", "graph_after"):
        graph_representations = training_examples[graphs_key]
        max_len = len(max(graph_representations, key=len))
        max_length_dict[graphs_key] = max_len
        for representation in graph_representations:
            needed_zeros_num = max_len - len(representation)
            representation.extend([0] * needed_zeros_num)
    return max_length_dict


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
        language: The language from which to find constraint types to generate training examples
        n_gen: The number of rules to apply to each initial graph of constraint types before generating training
            sequences
        n_items: The range of items to use when generating the initial graphs of constraint types (graphs of
            range(n_items) nodes will be generated). If not given, defaults to len(language.constraint_types)

    -- Returns --
        data: The list containing the training examples
        labels: The list containing the training labels
        max_length_dict: The dictionary containing the max length for "graph_before", "graph_after" representations
    """
    rule_search = RuleSearch()
    rules = language.rules
    rule_representations = create_rule_representations(rules)
    initial_graph_list = generate_initial_graph_list(language.types, n_items)
    training_examples = {"graph_before": [],    # Representations of the graphs before the Rule application
                         "rule": [],            # Rules that were applied
                         "graph_after": [],     # Representations of the graphs after the Rule application
                         "label": []}           # 1 if positive training example, 0 if negative
    for graph in initial_graph_list:
        chosen_rules = choices(rules, k=n_gen)
        sequence = generate_sequence(graph, chosen_rules, rule_search)
        if sequence:
            add_sequence_to_training_examples(sequence, training_examples, language, rule_representations)
    if not training_examples["graph_before"]:
        raise TrainingGenerationError("Training examples could not be generated")
    max_length_dict = fix_length_training_examples(training_examples)
    data = [graph_before + rule + graph_after for graph_before, rule, graph_after in
            zip(training_examples["graph_before"],
                training_examples["rule"],
                training_examples["graph_after"])]
    labels = training_examples["label"]
    remove_duplicates(data, labels)
    return data, labels, max_length_dict
