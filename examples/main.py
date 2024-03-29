from pymeleon.dsl.parser import RuleParser
from pymeleon.dsl.rule import Rule
from pymeleon.viewer.genetic_viewer import GeneticViewer
import numpy as np
from pymeleon.dsl.dsl import DSL


def float2int(x: float):
    return int(x) + 2
    
constraint_types = {"list": lambda x: isinstance(x, list),
                    "nparray": lambda x: isinstance(x, np.ndarray),
                    "float": lambda x: isinstance(x, float),
                    "int": lambda x: isinstance(x, int)}

convert_to_nparray = Rule(RuleParser("a", constraints={"a": "list"}),
                          RuleParser("np.array(a)", constraints={"np.array": "nparray"}))

dot_product = Rule(RuleParser("a", "b", constraints={"a": "nparray", "b": "nparray"}),
                   RuleParser("np.sum(a*b)", constraints={"np.sum": "float"}))

float_to_int = Rule(RuleParser("a", constraints={"a": "float"}),
                   RuleParser("float2int(a)", constraints={"float2int": "int"}))

ext = {"np": np,
       "float2int": float2int}

list1 = [1.3, 2, 3]
list2 = [2, 2.5, 15]

lang = DSL()
lang.add_rules(convert_to_nparray, dot_product, float_to_int)
lang.add_types(constraint_types)
# Or we could say DSL(rules=[convert_to_nparray, dot_product], types=constraint_types)

viewer = GeneticViewer(ext=ext, dsl=lang, hyperparams={"num_epochs": 1000}, device_str="cuda", use_pretrained=True)
obj = viewer.blob(list1, list2)
target_out = RuleParser("a", constraints={"a": "int"})
print(obj.view(target_out))
