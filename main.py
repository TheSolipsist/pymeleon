from language.parser import Parser
from matplotlib import pyplot as plt
import networkx as nx

expression = "a + func(b * (c + 2 * func2(z * c, a + b)), a + b) / b * 2 + 8"
parser_obj = Parser(expression, {})
DG = parser_obj.graph

pos = nx.spring_layout(DG)
nx.draw(DG, pos, node_size=1000)
nx.draw_networkx_labels(DG, pos, labels=nx.get_node_attributes(DG, "name"))
nx.draw_networkx_edge_labels(DG, pos, edge_labels=nx.get_edge_attributes(DG, "order"))
plt.savefig("graph.png")
