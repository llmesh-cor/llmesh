"""
Dynamic routing algorithm for the MESH network
Implements self-healing and optimal path finding
"""

import heapq
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Route:
    """Represents a route through the network"""
    path: List[str]
    latency: float
    reliability: float
    cost: float

    @property
    def score(self) -> float:
        """Calculate route score (lower is better)"""
        return (self.latency * 0.4 + 
                (1 - self.reliability) * 0.4 + 
                self.cost * 0.2)


class DynamicRouter:
    """
    Implements dynamic routing for the MESH network
    Features: multi-path routing, self-healing, QoS awareness
    """

    def __init__(self):
        """Initialize the router"""
        self.routing_table: Dict[str, Dict[str, Route]] = {}
        self.node_metrics: Dict[str, Dict[str, float]] = {}
        self.path_cache: Dict[Tuple[str, str], List[Route]] = {}
        self.update_interval = 10.0
        self.last_update = 0

    def update_node_metrics(self, node_id: str, metrics: Dict[str, float]):
        """Update metrics for a node"""
        self.node_metrics[node_id] = {
            "latency": metrics.get("latency", 100.0),
            "reliability": metrics.get("reliability", 0.95),
            "load": metrics.get("load", 0.5),
            "stake": metrics.get("stake", 100.0),
            "timestamp": time.time()
        }

    def find_routes(self, source: str, target: str, 
                   model_name: str = None, k: int = 3) -> List[Route]:
        """
        Find k best routes from source to target
        Uses modified Dijkstra's algorithm with multiple metrics
        """

        # Check cache first
        cache_key = (source, target)
        if cache_key in self.path_cache:
            cached_routes = self.path_cache[cache_key]
            if cached_routes and time.time() - self.last_update < self.update_interval:
                return cached_routes[:k]

        # Calculate fresh routes
        routes = self._calculate_routes(source, target, model_name)

        # Cache results
        self.path_cache[cache_key] = routes
        self.last_update = time.time()

        return routes[:k]

    def handle_node_failure(self, failed_node: str):
        """Handle node failure by updating routing tables"""
        logger.warning(f"Handling failure of node {failed_node}")

        # Remove from metrics
        if failed_node in self.node_metrics:
            del self.node_metrics[failed_node]

        # Clear affected cache entries
        self.path_cache = {
            k: v for k, v in self.path_cache.items()
            if failed_node not in k
        }

        # Trigger immediate route recalculation for affected paths
        self._recalculate_affected_routes(failed_node)

    def _calculate_routes(self, source: str, target: str, 
                         model_name: Optional[str]) -> List[Route]:
        """Calculate all viable routes between nodes"""

        # Build graph from node metrics
        graph = self._build_graph()

        # Find paths using modified Yen's algorithm
        paths = self._find_k_shortest_paths(graph, source, target, k=10)

        # Convert paths to routes with metrics
        routes = []
        for path in paths:
            route = self._path_to_route(path, model_name)
            if route:
                routes.append(route)

        # Sort by score
        routes.sort(key=lambda r: r.score)

        return routes

    def _build_graph(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build network graph from node metrics"""
        graph: Dict[str, List[Tuple[str, float]]] = {}

        # In production, this would use actual network topology
        # For now, create a mesh topology based on metrics
        nodes = list(self.node_metrics.keys())

        for i, node1 in enumerate(nodes):
            graph[node1] = []
            for j, node2 in enumerate(nodes):
                if i != j:
                    # Calculate edge weight based on metrics
                    weight = self._calculate_edge_weight(node1, node2)
                    graph[node1].append((node2, weight))

        return graph

    def _calculate_edge_weight(self, node1: str, node2: str) -> float:
        """Calculate edge weight between two nodes"""
        metrics1 = self.node_metrics.get(node1, {})
        metrics2 = self.node_metrics.get(node2, {})

        # Combine metrics for edge weight
        latency = (metrics1.get("latency", 100) + metrics2.get("latency", 100)) / 2
        reliability = (metrics1.get("reliability", 0.9) * metrics2.get("reliability", 0.9))
        load = (metrics1.get("load", 0.5) + metrics2.get("load", 0.5)) / 2

        # Weight formula
        weight = latency * (1 + load) / reliability

        return weight

    def _find_k_shortest_paths(self, graph: Dict[str, List[Tuple[str, float]]], 
                              source: str, target: str, k: int) -> List[List[str]]:
        """Find k shortest paths using Yen's algorithm"""
        paths = []

        # Find shortest path first
        shortest = self._dijkstra(graph, source, target)
        if not shortest:
            return []

        paths.append(shortest)

        # Find k-1 more paths
        for i in range(1, k):
            for j in range(len(paths[i-1]) - 1):
                # Create graph copy without used edges
                graph_copy = self._remove_edge(graph.copy(), paths[i-1][j], paths[i-1][j+1])

                # Find alternative path
                alt_path = self._dijkstra(graph_copy, paths[i-1][j], target)
                if alt_path:
                    full_path = paths[i-1][:j] + alt_path
                    if full_path not in paths:
                        paths.append(full_path)

            if len(paths) == i:
                break  # No more paths found

        return paths

    def _dijkstra(self, graph: Dict[str, List[Tuple[str, float]]], 
                  source: str, target: str) -> Optional[List[str]]:
        """Dijkstra's shortest path algorithm"""
        distances = {node: float('inf') for node in graph}
        distances[source] = 0
        previous = {}
        pq = [(0, source)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current == target:
                # Reconstruct path
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(source)
                return list(reversed(path))

            if current in visited:
                continue

            visited.add(current)

            for neighbor, weight in graph.get(current, []):
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))

        return None

    def _remove_edge(self, graph: Dict[str, List[Tuple[str, float]]], 
                     node1: str, node2: str) -> Dict[str, List[Tuple[str, float]]]:
        """Remove edge from graph"""
        if node1 in graph:
            graph[node1] = [(n, w) for n, w in graph[node1] if n != node2]
        return graph

    def _path_to_route(self, path: List[str], model_name: Optional[str]) -> Optional[Route]:
        """Convert path to route with metrics"""
        if len(path) < 2:
            return None

        total_latency = 0
        total_reliability = 1.0
        total_cost = 0

        for i in range(len(path) - 1):
            metrics = self.node_metrics.get(path[i], {})
            total_latency += metrics.get("latency", 100)
            total_reliability *= metrics.get("reliability", 0.9)

            # Add model-specific costs
            if model_name and i == len(path) - 2:  # Target node
                total_cost += 0.1  # Base inference cost

        return Route(
            path=path,
            latency=total_latency,
            reliability=total_reliability,
            cost=total_cost
        )

    def _recalculate_affected_routes(self, failed_node: str):
        """Recalculate routes affected by node failure"""
        # In production, this would trigger immediate recalculation
        # of all routes passing through the failed node
        logger.info(f"Recalculating routes affected by {failed_node} failure")
