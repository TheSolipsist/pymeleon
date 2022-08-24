from language.parser import GeneticParser, RuleParser
from language.rule import Rule
from viewer.genetic_viewer import GeneticViewer
import pygrank as pg
import networkx as nx
from language.language import Language

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

lang = Language()
lang.add_rules(Rule(RuleParser("a", constraints={"a": "GraphSignal"}),
                    RuleParser("a.graph", constraints={"a.graph": "Graph"})),
               Rule(RuleParser("a", constraints={"a": "List"}),
                    RuleParser("list2dict(a)", constraints={"list2dict": "Dict"})),
               Rule(RuleParser("a", "b", constraints={"a": "Graph", "b": "Dict"}),
                    RuleParser("pg.to_signal(a, b)", constraints={"pg.to_signal": ("OUT", "GraphSignal")})),
               )
lang.add_types(constraint_types)

viewer = GeneticViewer(lang, {"list2dict": list2dict, "pg": pg})
obj = viewer.blob(G, x)
target_out = GeneticParser("a", constraints={"a": ("OUT", "GraphSignal")})
print(obj.view(target_out))
