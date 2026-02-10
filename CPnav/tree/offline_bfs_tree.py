import heapq
import math
import networkx as nx
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

