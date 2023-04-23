from graphgym.graph import Graph

# Create a simple graph
graph = Graph(directed=True)
graph.add_edge(0, 1)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(3, 4)
graph.add_edge(4, 0)

print(graph)
