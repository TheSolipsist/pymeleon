from language.rule import Rule
from viewer.viewer import Viewer
from object.object import PymLiz
from language.parser import RuleParser, PymLizParser, GeneticParser
from language.language import Language
from random import choice
from utilities.util_funcs import save_graph
from language.rule_search import RuleSearch

class RecursionObject:
    """
    Object to be used for recursion (to pass less arguments in each recursion step)
    """
    def __init__(self):
        pass

class GeneticViewer(Viewer):
    """
    Genetic viewer class, implementing genetic selection and application of Rules
    
    -- Parameters --
        language(Language): The language object from which to find Rules
        
    -- Attributes --
        language(Language): The viewer's language
        
    -- Methods --
        blob(*args): Creates and returns the PymLiz object
        view(): Returns the object after having changed it according to the viewer's function
        search(rule, obj): Iterates through the possible subgraphs (in the form of transform_dicts) that match 
            a rule's input graph
    """
    def __init__(self, language: Language, modules: dict=None, 
                 n_iter=10, n_generations=50, penalty_coefficient=0.1, n_fittest=1000):
        super().__init__(language)
        self._RuleSearch = RuleSearch()
        if modules is None:
            modules = dict()
        self.modules=modules
        self.n_iter = n_iter
        self.n_generations = n_generations
        self.penalty_coefficient = penalty_coefficient
        self.n_fittest = n_fittest
    
    def blob(self, *args):
        """
        Creates and returns the PymLiz object
        """
        obj = PymLiz(self, PymLizParser(*args), constraint_types=self.language.types, modules=self.modules)
        return obj
    
    def view(self, obj: PymLiz, parser_obj: GeneticParser):
        """
        Returns the object's output after having changed it according to the viewer's function
        """
        target_graph = parser_obj.get_graph()
        target_penalty = self._calculate_target_penalty(target_graph)
        rules = self.language.rules
        max_score = float("-inf")
        n_iter = self.n_iter
        n_fittest = self.n_fittest
        n_generations = self.n_generations
        for i_iter in range(n_iter):
            obj_list = [obj.copy() for __ in range(n_fittest)]
            for i_gen in range(n_generations):
                print(f"\rRunning: GeneticViewer.view() - Iteration {i_iter + 1}, Generation {i_gen + 1}  ", end = '')
                for i in range(n_fittest):
                    current_obj = obj_list[i]
                    chosen_rule = choice(rules)
                    transform_dicts = tuple(self.search(chosen_rule, current_obj))
                    if not transform_dicts:
                        obj_list.append(current_obj.copy())
                        continue
                    chosen_transform_dict = choice(transform_dicts)
                    new_obj = current_obj.apply(chosen_rule, chosen_transform_dict)
                    obj_list.append(new_obj)
                obj_list.sort(key=lambda obj: self.fitness(obj.get_graph(), target_graph, target_penalty), reverse=True)
                del obj_list[n_fittest:]
            current_best_obj = obj_list[0]
            current_best_score = self.fitness(current_best_obj.get_graph(), target_graph, target_penalty)
            if current_best_score > max_score:
                max_score = current_best_score
                best_obj = current_best_obj
        print()
        return best_obj.run()
                
        
    def search(self, rule: Rule, obj: PymLiz):
        """
        Iterates through the possible subgraphs (in the form of transform_dicts) that match a rule's input graph
        """
        return self._RuleSearch(rule, obj._graph)
    
    def _calculate_target_penalty(self, target_graph):
        return sum((target_graph.in_degree(node) ** 2 for node in target_graph))
        
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
            
    def fitness(self, graph, target_graph, target_penalty):
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
        