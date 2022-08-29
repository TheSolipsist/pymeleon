from time import perf_counter
from dsl.rule import Rule
from viewer.genetic_viewer import GeneticViewer
import pygrank as pg
import networkx as nx
from dsl.dsl import DSL
from dsl.parser import Predicate, parse

def list2dict(x: list):
    return {v: 1 for v in x}

def get_graph(x: nx.Graph):
    return x.graph

G = nx.Graph()
G.add_edge("node_a", "node_b")
x = ["node_a"]

viewer = DSL(
    Predicate("normalized", lambda x: pg.max(x) == 1),
    Rule(parse(pg.GraphSignal), parse({"get_graph(_)": nx.Graph})),
    Rule(parse(list), parse("list2dict(_)", {"list2dict": ("normalized", dict)})),
    Rule(parse({"a": nx.Graph, "b": dict}), parse("pg.to_signal(a, b)", {"pg.to_signal": ("normalized", "noinput", pg.GraphSignal)})),
) >> GeneticViewer({"list2dict": list2dict, "pg": pg, "get_graph": get_graph}, use_pretrained=False)

# result = viewer(G, x) >> parse({"a": "OUT", "b": "GraphSignal"})
# print(result)
