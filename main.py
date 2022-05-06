from language.parser import Parser
from matplotlib import pyplot as plt
from language.rule import Rule
from object.object import PymLiz
import networkx as nx
from time import perf_counter
import numpy as np
from utilities.util_funcs import save_graph

constraint_types = {"islist": lambda x: isinstance(x, list),
                    "isnparray": lambda x: isinstance(x, np.ndarray),
                    "isint": lambda x: isinstance(x, int)}

convert_to_nparr = Rule(Parser("a", constraints={"a": "islist"}, mode="RULE"),
                       Parser("np.array(a)", constraints={"np.array": "isnparray"}, mode="RULE"))

dot_product = Rule(Parser("a", "b", constraints={"a": "isnparray", "b": "isnparray"}, mode="RULE"),
                   Parser("np.sum(a*b)", constraints={"np.sum": "isnparray"}, mode="RULE"))

multiply_by_2 = Rule(Parser("a", constraints={"a": "isnparray"}, mode="RULE"),
                     Parser("2 * a", mode="RULE"))
    
list1 = [1, 2, 3]
list2 = [2, 2, 2]
obj1 = PymLiz(Parser(list1, list2, mode="PYMLIZ"), constraint_types=constraint_types, modules_to_import={"numpy": "np"})

print(f"view 1: {obj1.view()}")
save_graph(obj1._graph, print=True)
for transform_dict in obj1.search(convert_to_nparr):
    obj1.apply(convert_to_nparr, transform_dict, inplace=True)
    save_graph(obj1._graph, print=True)
    print(f"view 2: {obj1.view()}")
for transform_dict in obj1.search(dot_product):
    obj1.apply(dot_product, transform_dict, inplace=True)
    save_graph(obj1._graph, print=True)
    print(f"view 3: {obj1.view()}")
    break
for transform_dict in obj1.search(multiply_by_2):
    obj1.apply(multiply_by_2, transform_dict, inplace=True)
    save_graph(obj1._graph, print=True)
    print(f"view 4: {obj1.view()}")
    break
