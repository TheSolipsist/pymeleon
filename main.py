from language.parser import Parser
from matplotlib import pyplot as plt
from language.rule import Rule
import networkx as nx

expression = "a + b * c"
parser_obj = Parser(expression, {})
DG = parser_obj.graph

myrule = Rule(Parser("a + b + c", {}), Parser("a * b", {}))
to_apply_rule = Parser("x + y + z", {})
transformed_dg = myrule.apply(to_apply_rule.graph, {"a": "x", "b": "y", "c": "z"})

pos = nx.planar_layout(transformed_dg)
nx.draw(transformed_dg, pos, node_size=400)
nx.draw_networkx_labels(transformed_dg, pos, labels=nx.get_node_attributes(transformed_dg, "name"), font_size=7)
nx.draw_networkx_edge_labels(transformed_dg, pos, edge_labels=nx.get_edge_attributes(transformed_dg, "order"), font_size=7)
plt.savefig("graph.png", dpi=600)
