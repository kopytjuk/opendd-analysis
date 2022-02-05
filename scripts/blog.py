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



# Frenet reference path
@dataclass
class DiscreteReferencePath:
    """Represents a reference path by a collection of points sampled along the
    arc length of the curve.
    """
    
    points: np.ndarray  # in cartesian (metric) coordinates, as Nx3 (s, x, y)
    spatial_resolution: float  # approximate
    
    def to_frenet(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a cartesian coordinate to a Frenet coordinate.

        Args:
            pt (Tuple[float, float]): cartesian coordinate (x, y)

        Returns:
            Tuple[float, float]: frenet coordinate (s, d)
        """
        
        pt = np.array(pt)
        
        # compute distances to each sample of the ref.-path
        deltas = np.linalg.norm(self.points[:, 1:] - pt, axis=1)
        
        # take the index of the minium
        idx_min = np.argmin(deltas)
        
        # arc length
        s = self.points[idx_min, 0]
        # perpendicular distance
        d = deltas[idx_min]
        return s,  d
