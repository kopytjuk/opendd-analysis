# stuff used for copy paste into blog

# generate graph
traffic_lanes_graph = momepy.gdf_to_nx(trafficlanes, approach='primal', directed=True, multigraph=False)

# indetify nodes
start_nodes = [node for node, degree in traffic_lanes_graph.in_degree if degree == 0]
end_nodes = [node for node, degree in traffic_lanes_graph.out_degree if degree == 0]

# find possible paths
paths = list()
for i, (start, end) in enumerate(product(start_nodes, end_nodes)):
    
    # find path
    path = nx.shortest_path(traffic_lanes_graph, start, end)
    
    # convert to linestring
    path_as_ls = graph_path_to_linestring(traffic_lanes_graph, path)
    
    paths.append(DrivablePath(i, path_as_ls))