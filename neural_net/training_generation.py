"""
Module for training data generation
"""
from dsl.dsl import DSL
from dsl.parser import Node
from networkx import DiGraph
from random import choice, choices
from dsl.rule import Rule
from dsl.rule_search import RuleSearch
from itertools import combinations_with_replacement


class TrainingGenerationError(Exception):
    pass


def get_top_nodes_graph(graph: DiGraph) -> DiGraph:
    """
    Returns a graph changed so that it is composed only of its top nodes
    """
    top_nodes_graph = DiGraph()
    top_nodes_graph.add_node("root_node")
    for node in graph.successors("root_node"):
        top_nodes_graph.add_edge("root_node", node.copy(), order=-1)
    return top_nodes_graph


def generate_graph_from_constraint_types(constraint_types: tuple[str]):
    """
    Generates a graph that is composed of a tuple of constraint types
    """
    graph = DiGraph()
    for constraint_type in constraint_types:
        # A node's constraints are a set of constraint types
        graph.add_edge("root_node", Node(constraints={constraint_type}), order=-1)
    return graph


def generate_initial_graph_list(constraint_types: dict, n_items: int) -> list[DiGraph]:
    """
    Generates the initial graph list

    The initial graph list consists of all combinations (with replacement) of the constraint types
    of length 1, 2, ..., n_items.
    """
    initial_graph_list = []
    for r in range(1, n_items + 1):
        initial_graph_list.extend(map(generate_graph_from_constraint_types,
                                      combinations_with_replacement(constraint_types, r)))
    return initial_graph_list


def add_samp_to_data(data: list[tuple[DiGraph]],
                     graph_before: DiGraph,
                     graph_after: DiGraph,
                     graph_negative: DiGraph,
                     graph_final: DiGraph):
    """
    Adds a training sample to the training data
    """
    
    data.append(((graph_before, graph_final),
                 (graph_after, graph_final),
                 (graph_negative, graph_final)
                 ))


def negative_sample(rules: list[Rule],
                    rule_search: RuleSearch,
                    applied_rule: Rule,
                    graph_before: DiGraph,
                    graph_after: DiGraph,
                    graph_final: DiGraph):
    """
    Given a positive sample in the form of (graph_before, graph_after, graph_final), generates and returns
    a negative graph_after
    """
    for rule in rules:
        if rule is not applied_rule:
            transform_dicts = tuple(rule_search(rule, graph_before))
            if transform_dicts:
                return rule.apply(graph_before, choice(transform_dicts))
    return None
    

def add_sequence_to_training_data(sequence: list[DiGraph], 
                                  rules_sequence: list[Rule],
                                  data: dict[str, list[DiGraph]],
                                  DSL: DSL,
                                  rule_search: RuleSearch,
                                  add_simple: bool = True):
    """
    Adds a sequence of (Graph_1, Graph_2, ..., Graph_n_gen) to the training data
    
    -- Arguments --
        sequence: The (Graph_1, Graph_2, ..., Graph_n_gen) sequence
        rules_sequence: The (Rule_1, Rule_2, ..., Rule_[n_gen-1]) sequence
        data: The training data
        DSL: The DSL used
        add_simple: If True, for every training sample consisting of a (graph_before, graph_after, graph_final) \
            sequence, also add the (graph_before, graph_after, top_nodes_graph_final) sequence, in which the graph \
            top_nodes_graph_final consists of only the root nodes of graph_final (successor nodes of "root_node"). \
            For example, if graph_final is "f1(f3(a), b), f2(c,d)", then top_nodes_graph_final is "f1, f2".
    """
    rules = DSL.rules
    if add_simple:
        top_nodes_sequence = tuple(get_top_nodes_graph(graph) for graph in sequence)
    for i in range(1, len(sequence)):
        for j in range(i, len(sequence)):
            add_samp_to_data(data, 
                             sequence[i - 1], 
                             sequence[i], 
                             negative_sample(rules, rule_search, rules_sequence[i-1], 
                                             sequence[i - 1], sequence[i], sequence[j]),
                             sequence[j])
            if add_simple:
                add_samp_to_data(data, 
                                sequence[i - 1], 
                                sequence[i], 
                                negative_sample(rules, rule_search, rules_sequence[i-1], 
                                                sequence[i - 1], sequence[i], top_nodes_sequence[j]),
                                top_nodes_sequence[j])


class TrainingGeneration:
    """
    ### Abstract training generation class
    
    #### Abstract methods:
        ``generate_training_data(DSL)``: Generates training data
    """
    def __init__(self) -> None:
        pass
    
    def generate_training_data(self, DSL: DSL) -> tuple[list[list[int]], list[int], int]:
        raise NotImplementedError("'generate_training_data' method not implemented")
    

class TrainingGenerationRandom(TrainingGeneration):
    """
    ### Training generation class for quick generation of few samples through random choice of rules and \
        transformation dictionaries
        
    Args:
        n_gen: The number of rules to apply to each initial graph of constraint types while generating training \
            sequences
        n_items: The range of items to use when generating the initial graphs of constraint types (graphs of \
            range(n_items) nodes will be generated).
            
    Methods:
        ``generate_training_data(DSL, n_gen, items)``: Returns the training data 
    """
    def __init__(self, 
                 n_gen: int,
                 n_items: int
                 ) -> None:
        self.n_gen = n_gen
        self.n_items = n_items
    
    def apply_rules_random(self, 
                           graph: DiGraph, 
                           chosen_rules: list[Rule],
                           rule_search: RuleSearch) -> list:
        """
        Applies a list of rules (with random transform_dicts) to a graph
        """
        sequence = [graph]
        rules_sequence = []
        for rule in chosen_rules:
            transform_dicts = tuple(rule_search(rule, graph))
            if not transform_dicts:
                continue
            chosen_transform_dict = choice(transform_dicts)
            graph = rule.apply(graph, chosen_transform_dict)
            sequence.append(graph)
            rules_sequence.append(rule)
        return sequence, rules_sequence
        
    def generate_sequence_random(self, 
                                 initial_graph: DiGraph, 
                                 chosen_rules: list[Rule], 
                                 rule_search: RuleSearch) -> tuple[list, list]:
        """
        Generates a (Graph_1, Graph_2, ..., Graph_final) positive sequence based on an initial graph and a sequence of 
        chosen rules

        If no rules were applied, returns None
        """
        sequence, rules_sequence = self.apply_rules_random(initial_graph, chosen_rules, rule_search)
        if not rules_sequence:
            sequence, rules_sequence = None, None
        return sequence, rules_sequence
    
    def generate_training_data(self, DSL: DSL) -> tuple[list[tuple[DiGraph]]]:
        """
        Generates training data for a given DSL

        Args:
            DSL: The DSL for which to generate training data

        Returns:
            list[tuple[DiGraph]]: The training data, containing 4-tuples of DiGraphs (graph_before, graph_after, graph_negative, graph_final)
        """
        rule_search = RuleSearch()
        data = []   # The training data (each record is a list of 3 DiGraphs: the graph before the application of the Rule,
                    # the graph after the application of the Rule, and the graph after the application of multiple Rules
        initial_graph_list = generate_initial_graph_list(DSL.types, self.n_items)
        total_nodes = sum(graph.number_of_nodes() for graph in initial_graph_list)
        examined_nodes = 0
        NUM_BARS = 20
        for graph in initial_graph_list:
            num_bars_done = round((examined_nodes / total_nodes) * NUM_BARS)
            print(f"\rGenerating training examples: |{num_bars_done * '='}{(NUM_BARS - num_bars_done) * ' '}|", end="")
            examined_nodes += graph.number_of_nodes()
            chosen_rules = choices(DSL.rules, k=self.n_gen)
            sequence, rules_sequence = self.generate_sequence_random(graph, chosen_rules, rule_search)
            if sequence:
                add_sequence_to_training_data(sequence, rules_sequence, data, DSL, rule_search, add_simple=True)
        print(f"\r{' ' * 60}", end="")
        if not data:
            raise TrainingGenerationError("Training data could not be generated")
        return data


class TrainingGenerationExhaustive(TrainingGeneration):
    """
    ### Training generation class for exhaustive (and slow) generation of training data, utilizing all possible rule combinations \
        and all transform dicts for these rules

    Methods:
        ``generate_training_data(DSL, n_gen, items)``: Returns the training data 
    """
    def __init__(self, 
                 n_gen: int, 
                 n_items: int) -> None:
        self.n_gen = n_gen
        self.n_items = n_items
    
    def generate_training_data(self, DSL: DSL) -> tuple[list[tuple[DiGraph]], list[int]]:
        """
        Generates training data for a given DSL

        Args:
            DSL: The DSL for which to generate training data

        Returns:
            list[tuple[DiGraph]]: The training data, containing 3-tuples of DiGraphs
        """
        pass