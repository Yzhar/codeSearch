class Graph:

    def __init__(self):
        self.nodes = {}

    def enter_nodes(self, node_id):
        if node_id not in self.nodes.keys():
            self.nodes[node_id] = []

    def enter_edge(self, node_id_1, node_id_2, weight):
        if node_id_1 in self.nodes.keys() and node_id_2 in self.nodes.keys():
            self.nodes[node_id_1].append((node_id_2, weight))
            self.nodes[node_id_2].append((node_id_1, weight))

    def get_neighbors(self, node_id):
        output = []
        for node in self.nodes[node_id]:
            output.append(node[0])
        return output

    def get_weight(self, node_id_1, node_id_2):
        for node in self.nodes[node_id_1]:
            if node[0] == node_id_2:
                return node[1]
        return -1
    def next_vertcies(self, id):
        output = []
        for node_id in self.nodes[id]:
            output.append(node_id[0])
        return output


# test_graph = graph.Graph()
# test_graph.enter_nodes('1')
# test_graph.enter_nodes('2')
# test_graph.enter_nodes('3')
# test_graph.enter_nodes('4')
# test_graph.enter_nodes('5')
# test_graph.enter_nodes('6')
# test_graph.enter_nodes('7')
# test_graph.enter_nodes('8')
# test_graph.enter_nodes('9')
# test_graph.enter_nodes('10')
# test_graph.enter_nodes('11')
# test_graph.enter_nodes('12')
#
# test_graph.enter_edge('1', '2', 0.67)
# test_graph.enter_edge('2', '5', 0.64)
# test_graph.enter_edge('2', '6', 0.6)
# test_graph.enter_edge('5', '9', 0.9)
# test_graph.enter_edge('6', '10', 0.7)
# test_graph.enter_edge('10', '12', 0.75)
# test_graph.enter_edge('9', '12', 0.8)
#
# test_graph.enter_edge('1', '3', 0.65)
# test_graph.enter_edge('3', '7', 0.85)
#
# test_graph.enter_edge('1', '4', 0.7)
# test_graph.enter_edge('4', '8', 0.4)
# test_graph.enter_edge('8', '11', 0.9)
# test_graph.enter_edge('11', '7', 0.75)
