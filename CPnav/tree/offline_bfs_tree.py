import heapq
import math
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple

def world_to_grid_00(pos: Tuple[float, float], grid_spacing_m: float) -> Tuple[int, int]:
    """
    Map world (x, z) in meters -> 0-based grid cell (row, col).
    Simple tiling: floor(coord / grid_spacing_m).
    """
    x, y = float(pos[0]), float(pos[1])
    if grid_spacing_m <= 0:
        raise ValueError("grid_spacing_m must be > 0")
    col = math.floor(x / grid_spacing_m)
    row = math.floor(y / grid_spacing_m)
    return int(row), int(col)


Node = Tuple[int, int]  # (row, col)

def widest_path_nx(
    G: nx.DiGraph,
    start: Node,
    goal: Node,
    score_attr: str = "score",
) -> Tuple[List[Node], Optional[float]]:
    """
    Widest path in a directed graph:
    maximize the minimum edge score along the path from start to goal.

    Returns:
      path (list of nodes), best_min_score (float) or ([], None) if unreachable.
    """
    if start == goal:
        return [start], 1.0  # matches your old convention

    if start not in G or goal not in G:
        return [], None

    # best_cap[v] = best (max) bottleneck capacity found so far for v
    best_cap: Dict[Node, float] = {start: math.inf}
    parent: Dict[Node, Node] = {}

    # max-heap via negative
    pq: List[Tuple[float, Node]] = [(-best_cap[start], start)]

    while pq:
        neg_cap, u = heapq.heappop(pq)
        cap_u = -neg_cap

        # stale entry
        if cap_u < best_cap.get(u, -math.inf):
            continue

        if u == goal:
            break

        for v, edata in G[u].items():  # neighbors + edge attribute dict
            score = edata.get(score_attr, None)
            if score is None:
                continue  # or treat as 0.0

            cap_v = min(cap_u, float(score))
            if cap_v > best_cap.get(v, -math.inf):
                best_cap[v] = cap_v
                parent[v] = u
                heapq.heappush(pq, (-cap_v, v))

    if goal not in best_cap:
        return [], None

    # Reconstruct path
    path: List[Node] = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()

    return path, best_cap[goal]


class OfflineGridTree:
    def __init__(self, gs: float):
        self.G = nx.DiGraph()
        self.gs = gs
        self.explored_grid = set()
        self.goals = set()
        self.start = None

    def initialize(self, obs):
        position = obs["agent_state"].position
        self.start = world_to_grid_00((position[0], position[2]), self.gs)
    
    def grid_explored(self, xz: Tuple[float, float]):
        xz_grid = world_to_grid_00(xz, self.gs)
        return xz_grid in self.explored_grid

    def add_edge(
            self,
            from_xz: Tuple[float, float], 
            to_xz: Tuple[float, float], 
            score: float
        ):
        from_xz_grid = world_to_grid_00(from_xz, self.gs)
        to_xz_grid = world_to_grid_00(to_xz, self.gs)

        self.explored_grid.add(from_xz_grid)
        self.explored_grid.add(to_xz_grid)

        if from_xz_grid == to_xz_grid:
            return

        if self.G.has_edge(from_xz_grid, to_xz_grid):
            prev_score = self.G[from_xz_grid][to_xz_grid]["score"]
            self.G[from_xz_grid][to_xz_grid]["score"] = max(score, prev_score)

        else:
            self.G.add_edge(from_xz_grid, to_xz_grid, score=score)
    
    def add_goal(self, goal_xz):
        goal  = world_to_grid_00(goal_xz,  self.gs)
        self.goals.add(goal)

    def is_goal(self, xz):
        grid = world_to_grid_00(xz, self.gs)
        return (grid in self.goals)
    
    def get_best_path_by_score(self):
        best_score = 0
        best_path = []
        assert self.goals is not None
        for goal in self.goals:
            path, score = widest_path_nx(
                self.G, 
                self.start,
                goal
            )
            if score > best_score:
                best_score = score
                best_path = path
        assert best_score is not None
        return best_score, best_path

def visualize_tree(tree: OfflineGridTree, filename: str = "offline_tree_debug.png"):
    """
    Visualizes the OfflineGridTree structure, saving it to a PNG file.
    - Nodes are placed physically based on (row, col).
    - Edges show scores.
    - Start (Green) and Goals (Gold) are highlighted.
    """
    if tree.G.number_of_nodes() == 0:
        print("[Visualizer] Graph is empty. Nothing to plot.")
        return

    plt.figure(figsize=(12, 10))
    
    # 1. Setup Position Layout
    # Mapping (row, col) -> (x, y) for plotting.
    # We use (col, -row) so that the grid visual matches matrix indexing 
    # (row 0 at top, row N at bottom).
    pos = {n: (n[1], -n[0]) for n in tree.G.nodes()}
    
    # If start is defined but not in graph (isolated), add it to pos for plotting
    if tree.start and tree.start not in pos:
        pos[tree.start] = (tree.start[1], -tree.start[0])

    # 2. Categorize Nodes
    goals_in_graph = [n for n in tree.goals if n in tree.G]
    start_node = [tree.start] if tree.start else []
    
    # Regular nodes are those that are not start or goals
    regular_nodes = [
        n for n in tree.G.nodes() 
        if n not in tree.goals and n != tree.start
    ]

    # 3. Draw Nodes
    # Draw Regular
    nx.draw_networkx_nodes(
        tree.G, pos, 
        nodelist=regular_nodes, 
        node_color='lightblue', 
        node_size=500, 
        alpha=0.8
    )

    # Draw Start
    if start_node:
        nx.draw_networkx_nodes(
            tree.G, pos, 
            nodelist=start_node, 
            node_color='lightgreen', 
            node_size=600, 
            node_shape='s', # Square for start
            label='Start'
        )

    # Draw Goals
    if goals_in_graph:
        nx.draw_networkx_nodes(
            tree.G, pos, 
            nodelist=goals_in_graph, 
            node_color='gold', 
            node_size=600, 
            node_shape='*', # Star for goal
            label='Goal'
        )

    # 4. Draw Labels (Node IDs)
    # Label format: "(row,col)"
    node_labels = {n: str(n) for n in tree.G.nodes()}
    if tree.start and tree.start not in node_labels:
        node_labels[tree.start] = str(tree.start)
        
    nx.draw_networkx_labels(tree.G, pos, labels=node_labels, font_size=8)

    # 5. Draw Edges
    nx.draw_networkx_edges(
        tree.G, pos, 
        arrowstyle='->', 
        arrowsize=20, 
        edge_color='gray', 
        alpha=0.5
    )

    # 6. Draw Edge Scores
    edge_labels = nx.get_edge_attributes(tree.G, 'score')
    # Format float scores to 2 decimal places
    formatted_edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    
    nx.draw_networkx_edge_labels(
        tree.G, pos, 
        edge_labels=formatted_edge_labels, 
        font_color='red', 
        font_size=9
    )

    # Final Polish
    plt.title(f"Offline Tree Visualization\nGrid Spacing: {tree.gs}m")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.axis('off') # Turn off axis numbers
    plt.tight_layout()
    
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[Visualizer] Tree saved to {filename}")

