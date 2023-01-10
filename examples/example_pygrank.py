from pymeleon.dsl.parser import RuleParser
from pymeleon.dsl.rule import Rule
from pymeleon.viewer.genetic_viewer import GeneticViewer
import pygrank as pg
import networkx as nx
from pymeleon.dsl.dsl import DSL

NOINPUT = lambda x: False

def list2dict(x: list):
    return {v: 1 for v in x}


constraint_types = {"Graph": lambda x: isinstance(x, nx.Graph),
                    "GraphSignal": lambda x: isinstance(x, pg.GraphSignal),
                    "Dict": lambda x: isinstance(x, dict),
                    "List": lambda x: isinstance(x, list),
                    "OUT": NOINPUT}

G = nx.Graph()
G.add_edge("a232", "b232")
x = ["a232"]

lang = DSL()
lang.add_rules(Rule(RuleParser("a", constraints={"a": "GraphSignal"}),
                    RuleParser("a.graph", constraints={"a.graph": "Graph"})),
               Rule(RuleParser("a", constraints={"a": "List"}),
                    RuleParser("list2dict(a)", constraints={"list2dict": "Dict"})),
               Rule(RuleParser("a", "b", constraints={"a": "Graph", "b": "Dict"}),
                    RuleParser("pg.to_signal(a, b)", constraints={"pg.to_signal": ("OUT", "GraphSignal")})),
               )
lang.add_types(constraint_types)

viewer = GeneticViewer({"list2dict": list2dict, "pg": pg}, lang)
obj = viewer.blob(G, x)
target_out = RuleParser("a", constraints={"a": ("OUT", "GraphSignal")})
ret = obj.view(target_out)
print(ret)
