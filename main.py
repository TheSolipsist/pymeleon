from language.parser import Parser
from matplotlib import pyplot as plt
from language.rule import Rule
from object.object import PymLiz
import networkx as nx
import config
from time import perf_counter

expression_LHS = "a + b + c"
constraints_LHS = {"a": lambda x: isinstance(x, int),
                   "b": lambda x: isinstance(x, int),
                   "c": lambda x: isinstance(x, float),
                   "add": lambda x: x == "add"}
LHS_obj = Parser(expression_LHS, constraints_LHS)
RHS_obj = Parser("a * b", {})

Rule_Add2Int1Float_To_Mul2 = Rule(LHS_obj, RHS_obj)

n1 = 1
n2 = 23
n3 = 1.4
n4 = "SomeNode"

config.locals = locals()
config.globals = globals()
obj1 = PymLiz(Parser("foo(n1 + n2 + n3, n2 + n1 + n3, n4)", {}))
for i, transform_dict in enumerate(obj1.search(Rule_Add2Int1Float_To_Mul2)):
    transformed_graph = obj1.apply(Rule_Add2Int1Float_To_Mul2, transform_dict)

    to_draw_dg = transformed_graph
    for node in to_draw_dg.nodes:
        if node != "root_node":
            to_draw_dg.nodes[node]["name"] = node.value
        else:
            to_draw_dg.nodes[node]["name"] = "root"
    pos = nx.planar_layout(to_draw_dg)
    nx.draw(to_draw_dg, pos, node_size=400)
    nx.draw_networkx_labels(to_draw_dg, pos, labels=nx.get_node_attributes(to_draw_dg, "name"), font_size=7)
    nx.draw_networkx_edge_labels(to_draw_dg, pos, edge_labels=nx.get_edge_attributes(to_draw_dg, "order"), font_size=7)
    # plt.show()
    plt.savefig(f"graph{i}.png", dpi=600)
    plt.close()
