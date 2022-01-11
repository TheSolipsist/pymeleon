from language.parser import Parser
from matplotlib import pyplot as plt
import networkx as nx

expression = "a + b * c"
parser_obj = Parser(expression, {})
DG = parser_obj.graph

print(list(DG.successors('root'))[0].value)
# pos = nx.planar_layout(DG)
# nx.draw(DG, pos, node_size=100)
# nx.draw_networkx_labels(DG, pos, labels=nx.get_node_attributes(DG, "name"), font_size=3)
# nx.draw_networkx_edge_labels(DG, pos, edge_labels=nx.get_edge_attributes(DG, "order"), font_size=3)
# plt.savefig("graph.png", dpi=600)
