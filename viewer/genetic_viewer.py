from language.rule import Rule
from viewer.viewer import Viewer
from object.object import PymLiz
from language.parser import PymLizParser, GeneticParser
from language.language import Language
from random import choice
from utilities.util_funcs import save_graph
from language.rule_search import RuleSearch
from viewer.fitness import FitnessHeuristic, FitnessNeuralNet


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
            for i_gen in range(self.n_gen):
                print(f"\rRunning: GeneticViewer.view() - Iteration {i_iter + 1}, Generation {i_gen + 1}  ", end="")
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
                obj_list.sort(key=lambda _obj: self.fitness(_obj.get_graph(),
                                                            target_graph), 
                              reverse=True)
                del obj_list[n_fittest:]
            current_best_obj = obj_list[0]
            current_best_score = self.fitness(current_best_obj.get_graph(),
                                              target_graph)
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

