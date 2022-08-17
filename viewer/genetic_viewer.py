from language.rule import Rule
from viewer.viewer import Viewer
from object.object import PymLiz
from language.parser import PymLizParser, GeneticParser
from language.language import Language
from random import choice
from utilities.util_funcs import save_graph
from language.rule_search import RuleSearch
from viewer.fitness import FitnessHeuristic, FitnessNeuralNet

#TODO extend check_graph_match_rec for not connected 

def _check_graph_match_rec(graph, target_graph, root_node, target_root_node):
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
                not _check_graph_match_rec(graph, target_graph, suc_node, target_suc_node)):
            return False
    return True

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
    def __init__(self,
                 language: Language,
                 modules: dict = None,
                 n_iter: int = 10,
                 n_gen: int = 20,
                 n_fittest: int = 10,
                 fitness: str = "neural_random",
                 device_str: str = "cpu",
                 hyperparams: dict = None
                 ) -> None :
        super().__init__(language)
        self._RuleSearch = RuleSearch()
        if modules is None:
            modules = dict()
        self.modules = modules
        self.n_iter = n_iter
        self.n_gen = n_gen
        self.n_fittest = n_fittest
        if fitness == "neural_random":
            self.fitness_obj = FitnessNeuralNet(language, hyperparams, training_generation="random", device_str=device_str)
        elif fitness == "neural_exhaustive":
            self.fitness_obj = FitnessNeuralNet(language, hyperparams, training_generation="exhaustive", device_str=device_str)
        elif fitness == "heuristic":
            self.fitness_obj = FitnessHeuristic()
        self.fitness = self.fitness_obj.fitness_score

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
        rules = self.language.rules
        max_score = float("-inf")
        n_iter = self.n_iter
        n_fittest = self.n_fittest
        for i_iter in range(n_iter):
            obj_list = [obj.copy() for __ in range(n_fittest)]
            scores = {_obj: self.fitness(_obj.get_graph(), target_graph) for _obj in obj_list}
            for i_gen in range(self.n_gen):
                print(f"\rRunning: GeneticViewer.view() - Iteration {i_iter + 1}, Generation {i_gen + 1}  ", end="")
                for i in range(n_fittest):
                    current_obj = obj_list[i]
                    chosen_rule = choice(rules)
                    transform_dicts = tuple(self.search(chosen_rule, current_obj))
                    if not transform_dicts:
                        obj_copy = current_obj.copy()
                        obj_list.append(obj_copy)
                        scores[obj_copy] = scores[current_obj]
                        continue
                    chosen_transform_dict = choice(transform_dicts)
                    new_obj = current_obj.apply(chosen_rule, chosen_transform_dict)
                    obj_list.append(new_obj)
                    from neural_net.training_generation import get_top_nodes_graph
                    if _check_graph_match_rec(get_top_nodes_graph(new_obj.get_graph()), get_top_nodes_graph(target_graph), "root_node", "root_node"):
                        print()
                        return new_obj.run()
                    scores[new_obj] = self.fitness(new_obj.get_graph(), target_graph) - new_obj.get_graph().number_of_edges() ** 2 * (float(i_iter) / float(n_iter))
                obj_list.sort(key=scores.__getitem__, reverse=True)
                # for _obj in obj_list:
                #     print(scores[_obj])
                #     save_graph(_obj.get_graph(), print=True, show_constraints=True)
                del obj_list[n_fittest:]
                scores = {_obj: scores[_obj] for _obj in obj_list}
            current_best_obj = obj_list[0]
            current_best_score = scores[current_best_obj]
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

