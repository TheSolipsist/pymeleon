"""
Rule module
"""
from networkx import DiGraph
from collections import defaultdict

from pymeleon.dsl.parser import RuleParser


class BadGraph(Exception):
    """
    Invalid graph input Exception
    """
    pass


class Rule:
    """
    Rule class for graph transformation
    
    -- Parameters --
        parser_obj_in: The parser object representing the graph that the rule can be applied to
        parser_obj_out: The parser object representing the graph after the application of the rule
        other_node_dict: Dictionary mapping input to output nodes that are not inferrable by variable names
        ext: A dictionary of external dependencies that are gathered by DSLs and are eventually passed to viewers.
    
    -- Attributes --
        node_dict: Dictionary mapping nodes from the generic input graph to the generic output graph (1-n)
        reverse_node_dict: Dictionary mapping nodes from the generic output graph to the generic input graph (1-1)
        
    -- Methods --
        apply(graph): Applies the rule to the specified graph
    """
    def __init__(self, 
                 parser_obj_in: RuleParser, 
                 parser_obj_out: RuleParser, 
                 other_node_dict=None, 
                 operator_dict=None,
                 ext=None) -> None:
        self._parser_obj_in = parser_obj_in
        self._parser_obj_out = parser_obj_out

        self._graph_in = parser_obj_in.graph
        self._obj_in = parser_obj_in.variables_constants
        self._funcs_in = parser_obj_in.functions

        self._graph_out = parser_obj_out.graph
        self._obj_out = parser_obj_out.variables_constants
        self._funcs_out = parser_obj_out.functions

        self._create_node_dict(other_node_dict, operator_dict)

        self.ext = ext
    
    def __str__(self):
        return f"{str(self._parser_obj_in)}\nTRANSFORMS TO\n{str(self._parser_obj_out)}"

    # ------- PRIVATE METHODS START ------- #

    def _create_node_dict(self, other_node_dict, operator_dict):
        node_dict = dict()
        reverse_node_dict = dict()
        graph_in_node_dict = dict()
        if other_node_dict is not None:
            for node in other_node_dict:
                node_dict[node] = other_node_dict[node]
                for output_node in other_node_dict[node]:
                    reverse_node_dict[output_node] = node
        # Python dictionaries are ordered since 3.7, so the first element will always be "root_node"
        for node in tuple(self._graph_in.nodes)[1:]:
            graph_in_node_dict[node.value] = node
        common_obj = self._obj_in & self._obj_out
        unmatched_nodes = list(self._graph_out.nodes)[1:]
        for obj in common_obj:
            obj_nodes = []
            for node in unmatched_nodes:
                if node.value == obj:
                    reverse_node_dict[node] = graph_in_node_dict[obj]
                    obj_nodes.append(node)
            node_dict[graph_in_node_dict[obj]] = list(obj_nodes)
            for node in obj_nodes:
                unmatched_nodes.remove(node)
        self.node_dict = node_dict
        self.reverse_node_dict = reverse_node_dict

    def _copy_apply_graph(self, graph: DiGraph, transform_dict: dict):
        """
        Returns a deepcopy of the graph and the new transform_dict
        """
        nodes_copy_dict = {node: node.copy() for node in tuple(graph.nodes)[1:]} # Skipping "root_node"
        nodes_copy_dict["root_node"] = "root_node"
        graph_copy = DiGraph()
        graph_copy.add_node("root_node")
        for node, node_copy in nodes_copy_dict.items():
            graph_copy.add_node(node_copy)
            for suc_node in graph.successors(node):
                suc_node_copy = nodes_copy_dict[suc_node]
                graph_copy.add_edge(node_copy, suc_node_copy, order=graph[node][suc_node]["order"])
        transform_dict_copy = {node_in: nodes_copy_dict[node_graph] for node_in, node_graph in transform_dict.items()}
        return graph_copy, transform_dict_copy

    def _remove_mapped_edges_rec(self, node):
        cur_transform_dict = self._cur_transform_dict
        for successor_node in self._graph_in.successors(node):
            self._cur_graph.remove_edge(cur_transform_dict[node], cur_transform_dict[successor_node])
            self._remove_mapped_edges_rec(successor_node)

    def _copy_output_node(self, node):
        """
        Returns a copy of a node in the generic output graph with the value required for the application of the rule
        """
        reverse_node_dict = self.reverse_node_dict
        cur_transform_dict = self._cur_transform_dict
        if node in reverse_node_dict:
            node_to_copy = cur_transform_dict[reverse_node_dict[node]]
            node_copy = node_to_copy.copy()
            self._cur_node_dict[node_to_copy].append(node_copy)
        else:
            node_copy = node.copy()
        return node_copy

    def _add_output_graph_rec(self, root_node, root_node_copy):
        graph_out = self._graph_out
        for node in graph_out.successors(root_node):
            node_copy = self._copy_output_node(node)
            self._cur_graph.add_edge(root_node_copy, node_copy, order=graph_out.get_edge_data(root_node, node)["order"])
            self._add_output_graph_rec(node, node_copy)

    def _add_output_graph(self):
        """
        Adds a copy of the output graph to the currently under transformation graph and create the specific _cur_node_dict
        """
        self._cur_node_dict = defaultdict(list)
        for node in self._graph_out.successors("root_node"):
            node_copy = self._copy_output_node(node)
            if node not in self.reverse_node_dict:
                self._cur_graph.add_edge("root_node", node_copy, order=-1)
            self._add_output_graph_rec(node, node_copy)

    def _get_cur_node_dict_edges(self, cur_node_dict: dict, graph: DiGraph, reverse_transform_dict: dict):
        """
        Returns any edges between nodes that are keys in the cur_node_dict.
        Should be called after having removed the edges that make up the input graph from the graph to which the rule
        is currently being applied.
        
        -- Arguments --
            cur_node_dict(dict[Node, Node]): Dictionary mapping the nodes in the graph that match the input graph's 
                nodes to their corresponding nodes in the graph that match the output nodes. It is the self.node_dict 
                for the specific input and output graphs contained in the full graph
        """
        pass
                
        
    # ------- PRIVATE METHODS END ------- #

    def apply(self, graph, transform_dict, deepcopy_graph=True) -> DiGraph:
        """
        Applies the rule to the specified graph

        -- Arguments --
            graph (networkx DiGraph): The graph to transform
            transform_dict (dict): Dictionary mapping each node in the generic input graph to a node in the graph to be transformed
            deepcopy_graph (bool): Specifies whether the graph will be deepcopied and returned or transformed in place
        
        -- Returns --
            transformed_graph (networkx DiGraph): The transformed graph
        """
        if deepcopy_graph:
            graph, transform_dict = self._copy_apply_graph(graph, transform_dict)
        reverse_transform_dict = {v: k for k, v in transform_dict.items()}
        
        self._cur_graph = graph
        self._cur_transform_dict = transform_dict
        self._cur_reverse_transform_dict = reverse_transform_dict

        # Remove the edges that make up the structure of the generic input graph
        for in_node in self._graph_in.successors("root_node"):
            if graph.has_edge("root_node", transform_dict[in_node]):
                graph.remove_edge("root_node", transform_dict[in_node])
            self._remove_mapped_edges_rec(in_node)
        self._add_output_graph() 
        # cur_node_dict is the equivalent of node_dict for the specific input and output graphs of the graph to transform
        cur_node_dict = self._cur_node_dict

        # Remove the nodes that were transformed and add between the new output nodes and the rest of the graph any 
        # edges that existed between nodes that were transformed (and mapped to output nodes) and the rest of the graph
        for graph_node in cur_node_dict:
            out_nodes = cur_node_dict[graph_node]
            num_out_nodes = len(out_nodes)
            for pre_node in graph.predecessors(graph_node):
                if pre_node not in reverse_transform_dict:
                    cur_order = graph.get_edge_data(pre_node, graph_node)["order"]
                    # out_edges with data=True returns a tuple of (pre_node, suc_node, {attribute_keys: attribute_values})
                    # The following code ensures that order is preserved
                    for edge in graph.out_edges(pre_node, data=True):
                        if edge[2]["order"] > cur_order:
                            edge[2]["order"] += num_out_nodes
                    for i, node in enumerate(out_nodes):
                        graph.add_edge(pre_node, node, order=cur_order + i + 1)
            for suc_node in graph.successors(graph_node):
                if suc_node in cur_node_dict:
                    for node in out_nodes:
                        for suc_node_out in cur_node_dict[suc_node]:
                            graph.add_edge(node, suc_node_out, order=graph.get_edge_data(graph_node, suc_node)["order"])
                else:
                    for node in out_nodes:
                        graph.add_edge(node, suc_node, order=graph.get_edge_data(graph_node, suc_node)["order"])
        for graph_node in reverse_transform_dict:
            # Fix order for any sibling nodes and then remove the node
            for pre_node in graph.predecessors(graph_node):
                cur_order = graph.get_edge_data(pre_node, graph_node)["order"]
                for edge in graph.out_edges(pre_node, data=True):
                    if edge[2]["order"] > cur_order:
                        edge[2]["order"] -= 1
            for suc_node in graph.successors(graph_node):
                if graph.in_degree(suc_node) == 1:
                    graph.add_edge("root_node", suc_node, order=-1)
            graph.remove_node(graph_node)

        del self._cur_graph
        del self._cur_transform_dict
        del self._cur_reverse_transform_dict
        del self._cur_node_dict

        if deepcopy_graph:
            return graph
