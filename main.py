from language.parser import Parser
from matplotlib import pyplot as plt
from language.rule import Rule
import networkx as nx
from time import perf_counter

LHS_obj = Parser("a + b + c", {})
RHS_obj = Parser("a * b", {})
myrule = Rule(LHS_obj, RHS_obj)
to_apply_rule = Parser("28 * (x + y + z + t) * m * n * n * pl", {})

LHS_nodes = dict()
for node in LHS_obj.graph.nodes:
    if node != "root_node":
        LHS_nodes[node.value] = node
apply_rule_nodes = dict()
for node in to_apply_rule.graph.nodes:
    if node != "root_node":
        apply_rule_nodes[node.value] = node        
transform_node_dict = dict()
transform_node_dict[LHS_nodes["add"]] = apply_rule_nodes["add"]
for val1, val2 in zip("abc", "xyz"):
    transform_node_dict[LHS_nodes[val1]] = apply_rule_nodes[val2]

t1 = perf_counter()
myrule.apply(to_apply_rule.graph, transform_node_dict, deepcopy_graph=False)
t2 = perf_counter()
print(t2-t1)

transformed_dg = to_apply_rule.graph

to_draw_dg = transformed_dg
for node in to_draw_dg.nodes:
    if node != "root_node":
        to_draw_dg.nodes[node]["name"] = node.value
    else:
        to_draw_dg.nodes[node]["name"] = "root"
pos = nx.planar_layout(to_draw_dg)
nx.draw(to_draw_dg, pos, node_size=400)
nx.draw_networkx_labels(to_draw_dg, pos, labels=nx.get_node_attributes(to_draw_dg, "name"), font_size=7)
nx.draw_networkx_edge_labels(to_draw_dg, pos, edge_labels=nx.get_edge_attributes(to_draw_dg, "order"), font_size=7)
plt.savefig("graph.png", dpi=600)
