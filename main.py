from language.parser import Parser
from matplotlib import pyplot as plt
from language.rule import Rule
from viewer.randomviewer import RandomViewer
import networkx as nx
from time import perf_counter
import numpy as np
from utilities.util_funcs import save_graph
from language.language import Language

constraint_types = {"islist": lambda x: isinstance(x, list),
                    "isnparray": lambda x: isinstance(x, np.ndarray),
                    "isint": lambda x: isinstance(x, int)}

convert_to_nparr = Rule(Parser("a", constraints={"a": "islist"}, mode="RULE"),
                       Parser("np.array(a)", constraints={"np.array": "isnparray"}, mode="RULE"))

dot_product = Rule(Parser("a", "b", constraints={"a": "isnparray", "b": "isnparray"}, mode="RULE"),
                   Parser("np.sum(a*b)", constraints={"np.sum": "isnparray"}, mode="RULE"))

modules = {"numpy": "np"}
    
list1 = [1, 2, 3]
list2 = [2, 2, 2]

lang = Language()
lang.add_rules(convert_to_nparr, dot_product)
lang.add_types(constraint_types)
# Or we could say Language(rules=[convert_to_nparr, dot_product], types=constraint_types)

viewer = RandomViewer(lang, modules)
obj = viewer.blob(list1, list2)
print(obj.view())
