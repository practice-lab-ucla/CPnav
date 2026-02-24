from __future__ import annotations

import math
import heapq
import networkx as nx
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

from ..env.actions import PolarAction, TeleopAction


StepId = int
ActionId = int

# ----------------------------
# Data classes
# ----------------------------

@dataclass(frozen=True)
class Pose2D:
    """2D pose in a global/world frame (meters, radians)."""
    x: float
    y: float
    yaw: float  # radians


@dataclass
class ActionProposal:
    """
    A candidate action from a node.
    index: discrete id (0 reserved for "backtrack"/turnaround in your old code)
    r: forward distance in meters
    theta: delta yaw in radians (relative turn then forward)
    score/adjusted: optional scoring fields (you may later remove 'adjusted')
    """
    index: int
    r: float
    theta: float
    score: Optional[float] = None
    adjusted: Optional[float] = None


@dataclass
class NodeData:
    """
    Node (state) stored in the exploration graph.

    actions: all proposals generated at this node (including index 0 if you keep it)
    unexplored: set of action indices not yet tried from this node
    gsv: global semantic value (optional)
    """
    step: StepId
    pose: Pose2D
    actions: List[ActionProposal] = field(default_factory=list)
    unexplored: Set[ActionId] = field(default_factory=set)
    gsv: Optional[float] = None


# ----------------------------
# Utility
# ----------------------------

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ----------------------------
# Exploration graph
# ----------------------------

class ExplorationTree:
    """
    Directed tree stored as nx.DiGraph: parent -> child.
    Includes a Global Priority Queue for actions.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        # Heap storing: (-score, -step_id, action_id)
        # We use negative score for Max-Heap behavior.
        # We use negative step_id to break ties by Recency (newer steps first).
        self.action_pq: List[Tuple[float, int, int]] = [] 

    # --------- internal helper ---------

    def _node(self, step_id: StepId) -> NodeData:
        """Return NodeData stored at node step_id. Raises KeyError if missing."""
        return self.G.nodes[step_id]["data"]

    def debug_print_node(self, step_id: StepId) -> None:
        """
        Pretty-print all data stored in a node for debugging.
        """
        if step_id not in self.G:
            print(f"[debug_print_node] step_id={step_id} NOT IN GRAPH")
            return

        node = self._node(step_id)

        print("\n" + "=" * 80)
        print(f"Node step_id = {step_id}")
        print("-" * 80)

        # Pose
        p = node.pose
        print("Pose:")
        print(f"  x   = {p.x:.3f}")
        print(f"  y   = {p.y:.3f}")
        print(f"  yaw = {p.yaw:.3f} rad")

        # Global semantic value
        print(f"\nGlobal Semantic Value (gsv): {node.gsv}")

        # Actions
        print("\nActions:")
        if not node.actions:
            print("  (no actions stored)")
        else:
            for a in node.actions:
                status = "UNEXPLORED" if a.index in node.unexplored else "TRIED"
                print(
                    f"  - idx={a.index:2d} | "
                    f"r={a.r:.3f} | "
                    f"theta={a.theta:.3f} | "
                    f"score={a.score} | "
                    f"adjusted={a.adjusted} | "
                    f"{status}"
                )

        # Unexplored set
        print("\nUnexplored action indices:")
        if node.unexplored:
            print(f"  {sorted(node.unexplored)}")
        else:
            print("  (none)")

        # Tree structure
        parent = self.get_parent(step_id)
        children = list(self.G.successors(step_id))

        print("\nTree links:")
        print(f"  Parent   : {parent}")
        print(f"  Children : {children if children else '(none)'}")
        print("=" * 80 + "\n")

    # --------- graph recording ---------
    def add_step(
        self,
        step_id: StepId,
        pose: Pose2D,
        actions: Optional[List[ActionProposal]] = None,
        gsv: Optional[float] = None,
        scores: Optional[float] = None,
        *,
        exclude_action0: bool = False,
    ) -> None:
        """
        Add or overwrite a node.
        """
        actions = actions or []

        if (not exclude_action0):
            # add action zero
            action_turnaround = ActionProposal(index=0, r=0, theta=0)
            actions.insert(0, action_turnaround)

        idxs = [a.index for a in actions]

        data = NodeData(step=step_id, pose=pose, actions=actions, gsv=gsv)
        data.unexplored = set(idxs)

        self.G.add_node(step_id, data=data)
        
        # If scores are provided immediately, fill them in (which populates PQ)
        if (scores is not None) and (gsv is not None):
            self.fill_in_scores(step_id, gsv, scores)


    def link_parent(self, parent_id: StepId, child_id: StepId) -> None:
        """
        Add a directed tree edge parent -> child.
        """
        self.G.add_edge(parent_id, child_id, kind="tree")

    # --------- per-node exploration state ---------

    def mark_tried(self, step_id: StepId, action_id: ActionId):
        """
        Mark an action as tried (remove from unexplored set).
        
        NOTE: This does NOT explicitly remove the item from self.action_pq.
        We use 'lazy deletion': the item stays in the PQ, but when popped later,
        we check if it is still in node.unexplored. If not, we discard it.
        """
        if step_id in self.G:
            self._node(step_id).unexplored.discard(action_id)

    def get_action_score(
        self,
        step_id: StepId,
        action_id: ActionId,
        default: float = 0.0,
    ) -> float:
        """
        Get the score for (step_id, action_id).
        """
        node = self._node(step_id)

        # action_id is 1..N; list index is 0..N-1
        idx = int(action_id)
        if idx < 0 or idx >= len(node.actions):
            raise IndexError(
                f"get_action_score: step={step_id} action_id={action_id} "
                f"(idx={idx}) out of range, len(actions)={len(node.actions)}"
            )

        a = node.actions[idx]

        if a.adjusted is not None:
            return float(a.adjusted)

        if a.score is not None:
            return float(a.score)

        return float(default)

    def get_unexplored_actions(self, step_id: StepId) -> List[ActionId]:
        """
        Return sorted unexplored action indices for a node.
        """
        return sorted(self._node(step_id).unexplored)

    def find_node_with_unexplored_actions(self, start_step_id: StepId) -> Optional[StepId]:
        """
        Recency-biased scan: from start_step_id down to 0.
        """
        for step in range(start_step_id, -1, -1):
            if step in self.G and len(self._node(step).unexplored) > 0:
                return step
        return None

    # --------- tree navigation helpers ---------

    def get_parent(self, step_id: StepId) -> Optional[StepId]:
        """
        Get the unique tree parent of step_id.
        """
        preds = list(self.G.predecessors(step_id))
        if len(preds) > 1:
            raise RuntimeError(f"Tree can only have one parent! node={step_id} preds={preds}")
        return preds[0] if preds else None

    def path_from_a_to_b(self, a: StepId, b: StepId) -> Tuple[List[StepId]]:
        """
        Compute the unique undirected path between a and b in a directed tree.
        """
        if a not in self.G or b not in self.G:
            raise KeyError(f"a and/or b not in graph: a={a in self.G}, b={b in self.G}")

        # 1) ancestors of a
        ancestors_a: Dict[Any, int] = {}
        path_to_root_a: List[Any] = []
        cur = a
        dist = 0
        while cur is not None:
            ancestors_a[cur] = dist
            path_to_root_a.append(cur)
            cur = self.get_parent(cur)
            dist += 1

        # 2) walk b up to LCA
        path_b_up: List[Any] = []
        cur = b
        while cur is not None and cur not in ancestors_a:
            path_b_up.append(cur)
            cur = self.get_parent(cur)

        if cur is None:
            raise ValueError(f"Nodes {a} and {b} are not connected in this tree")

        lca = cur
        a_to_lca = path_to_root_a[: ancestors_a[lca] + 1]  # includes lca
        lca_to_b = list(reversed(path_b_up))               # excludes lca

        return a_to_lca, lca_to_b
    
    def get_path_length(self, node_a: StepId, node_b: StepId) -> float:
        """
        Calculates the total Euclidean distance along the tree path from node_a to node_b.
        Used for calculating the 'cost' of a teleport/backtrack action.
        """
        if node_a == node_b:
            return 0.0
            
        # Get the sequence of nodes [start, ..., lca, ..., end]
        path_a_to_lca, lca_to_b = self.path_from_a_to_b(node_a, node_b)
        
        # path_a_to_lca includes LCA at the end
        # lca_to_b starts with the child of LCA
        full_path = path_a_to_lca + lca_to_b
        
        total_dist = 0.0
        for i in range(len(full_path) - 1):
            p1 = self._node(full_path[i]).pose
            p2 = self._node(full_path[i+1]).pose
            
            # Euclidean distance between consecutive nodes in the path
            dist = math.hypot(p2.x - p1.x, p2.y - p1.y)
            total_dist += dist
            
        return total_dist

    # --------- path -> MOVE_POLAR primitives ---------

    def path_to_move_delta(self, path: List[StepId], in_reverse: bool=False) -> List[Tuple[float, float]]:
        """
        Convert a node-path [s0, s1, ..., sk] into a sequence of MOVE_POLAR commands.
        """
        if len(path) < 2:
            return []

        cmds: List[Tuple[float, float]] = []
        cur_pose = self._node(path[0]).pose

        for i in range(1, len(path)):
            next_pose = self._node(path[i]).pose

            dx = next_pose.x - cur_pose.x
            dy = next_pose.y - cur_pose.y

            # heading to the next node in world frame
            dtheta = wrap_to_pi(next_pose.yaw - cur_pose.yaw)

            # straight-line distance
            r = math.hypot(dx, dy)
            if not in_reverse:
                cmds.append((r, dtheta))
            else:
                cmds.append((-r, dtheta))

            cur_pose = Pose2D(x=next_pose.x, y=next_pose.y, yaw=next_pose.yaw)

        return cmds

    def _action_score(self, node: NodeData, a: ActionProposal) -> float:
        if a.adjusted is not None:
            return float(a.adjusted)
        if a.score is not None:
            return float(a.score)
        return 0.0

    def select_next_action_at(
        self,
        step_id: StepId,
        score_threshold: float = 0.0,
        exclude_action0: bool = True,
        score_fn: Optional[Callable[[NodeData, ActionProposal], float]] = None,
    ) -> Optional[ActionId]:
        """
        Pick the best UNEXPLORED action at a specific node (Local Choice).
        """
        node = self._node(step_id)
        if score_fn is None:
            score_fn = lambda n, a: self._action_score(n, a)

        best: Optional[Tuple[int, float]] = None  # (idx, score)
        for a in node.actions:
            idx = int(a.index)
            if exclude_action0 and idx == 0:
                continue
            if idx not in node.unexplored:
                continue
            sc = float(score_fn(node, a))
            if sc < score_threshold:
                continue
            if best is None or sc > best[1]:
                best = (idx, sc)

        return best[0] if best else None

    # --------- Priority Queue Logic ---------

    def get_best_global_action(self, exclude_action0=True) -> Optional[Tuple[StepId, ActionId]]:
        """
        Pops the highest priority action from the Global Priority Queue.
        Checks validity (lazy deletion) against the 'unexplored' set.
        
        Returns:
            (step_id, action_id) or None if PQ is empty/exhausted.
        """
        while self.action_pq:
            # Pop: (-score, -step_id, action_id)
            neg_score, neg_step, action_id = heapq.heappop(self.action_pq)
            step_id = -neg_step

            # Verify node still exists
            if step_id not in self.G:
                continue

            if exclude_action0 and (action_id == 0):
                continue
            
            node = self._node(step_id)

            # Verify action is still unexplored (Lazy Deletion check)
            if action_id in node.unexplored:
                return step_id, action_id, -neg_score
                
        return None

    def fill_in_scores(self, step_id: StepId, gsv: float, action_scores: List[float], normalize_actions=True) -> None:
        """
        Fills per-action score, adjusted = score * gsv, AND pushes to Global PQ.
        """
        node = self._node(step_id)
        node.gsv = float(gsv)

        N = len(node.actions)
        M = len(action_scores)

        # --- decide whether GPT generated the correct action ---
        valid_fill = M == N
        scores = []
        
        if valid_fill:
            for s in action_scores:
                try:
                    v = float(s)
                except Exception:
                    v = 0.0
                if not math.isfinite(v):
                    v = 0.0
                scores.append(v)
        else:
            # if not, fill the score as 1/N
            scores = [1/N for _ in range(N)]

        for i, a in enumerate(node.actions):
            s = scores[i]
            a.score = s
            a.adjusted = s * node.gsv
            
            # Add to PQ if currently unexplored
            if a.index in node.unexplored:
                # Use adjusted score for priority, fallback to raw score
                priority = a.adjusted
                
                # Push: (-priority, -step_id, action_index)
                # 1. Negative priority => Max Heap
                # 2. Negative step_id => Tie-break: Prefer newer steps (Recency)
                heapq.heappush(self.action_pq, (-priority, -step_id, a.index))

    @staticmethod
    def to_action_proposal(sorted_actions):
        """
        GPT action indices are 1..N (0 reserved for backtrack).
        """
        proposals = []
        for i, a in enumerate(sorted_actions):
            proposals.append(ActionProposal(index=i+1, r=float(a.r), theta=float(a.theta)))
        return proposals

    def local_backtrack_once(
        self,
        step_id: StepId,
        *,
        score_threshold: float = 0.0,
        exclude_action0: bool = False,
        score_fn: Optional[Callable[[NodeData, ActionProposal], float]] = None,) -> Optional[Tuple[int, int]]:
        """
        Local backtrack: go to parent, pick best child.
        """
        parent = self.get_parent(step_id)
        if parent is None:
            return None

        unexpl = self.get_unexplored_actions(parent)
        if not unexpl:
            return None

        next_action = self.select_next_action_at(parent, score_threshold, exclude_action0, score_fn)
        if next_action is None:
            return None

        return parent, next_action

    def local_backtrack(
        self,
        step_id: StepId,
        *,
        score_threshold: float = 0.0,
        exclude_action0: bool = False,
        score_fn: Optional[Callable[[NodeData, ActionProposal], float]] = None,
    ) -> Optional[Tuple[int, int]]:
        
        # We start by looking at the parent of the current step
        current_step = step_id
        
        while True:
            result = self.local_backtrack_once(
                current_step, 
                score_threshold=score_threshold, 
                exclude_action0=exclude_action0, 
                score_fn=score_fn
            )
            
            if result is None:
                # We can't go higher, or the parent is empty. 
                # Let's try to find the best action AT the current node instead.
                # We MUST exclude_action0 here, otherwise we might loop infinitely.
                fallback_action = self.select_next_action_at(
                    current_step, 
                    score_threshold=score_threshold, 
                    exclude_action0=True, # Force True to ensure we progress
                    score_fn=score_fn
                )
                
                if fallback_action is None:
                    return None
                
                # Return the current node and the best action found there
                return current_step, fallback_action
            
            parent, next_action = result
            
            if (not exclude_action0) and (next_action == 0):
                current_step = parent
                continue
                
            return parent, next_action
    
    def global_backtrack(
        self,
        step_id: StepId,
        *,
        score_threshold: float = 0.0,
        exclude_action0: bool = True,
        score_fn: Optional[Callable[[NodeData, ActionProposal], float]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Returns (target_step_id, action_id) for global backtrack.
        Uses recency-first strategy to find the first ancestor with valid options.
        """
        
        target_step = self.select_global_target_recency_first(
            current_step_id=step_id,
            score_threshold=score_threshold,
            exclude_action0=exclude_action0,
            score_fn=score_fn
        )

        if target_step is None:
            # Nowhere left to explore in the entire history
            return None

        # Now pick the specific action at that node
        target_action_id = self.select_next_action_at(
            target_step, 
            score_threshold, 
            exclude_action0, 
            score_fn
        )

        if target_action_id is None:
            # This handles the rare edge case where a node was selected, 
            # but for some reason select_next_action_at disagrees.
            # Ideally, we should loop and try finding the *next* target_step, 
            # but returning None is safe behavior.
            return None
        
        return (target_step, target_action_id)
    
    def select_global_target_recency_first(
        self,
        current_step_id: StepId,
        *,
        score_threshold: float = 0.0,
        exclude_action0: bool = True,
        score_fn: Optional[Callable[[NodeData, ActionProposal], float]] = None,
    ) -> Optional[StepId]:
        """
        Recency-first global target selection (matches your friend's behavior):
        - scan steps newest -> oldest
        - return the FIRST step that has at least one valid unexplored action (passing threshold)

        Output:
        target step_id or None.
        """
        if score_fn is None:
            score_fn = lambda n, a: self._action_score(n, a)

        for step in range(current_step_id, -1, -1):
            if step not in self.G:
                continue
            node = self._node(step)

            # quick pre-check: any unexplored at all?
            if not node.unexplored:
                continue

            # confirm at least one valid unexplored action passing threshold
            for a in node.actions:
                idx = int(a.index)
                if exclude_action0 and idx == 0:
                    continue
                if idx not in node.unexplored:
                    continue
                if float(score_fn(node, a)) >= score_threshold:
                    return step

        return None

    def get_node_position(self, step_id: StepId):
        pose = self._node(step_id).pose
        x, z, yaw = pose.x, pose.y, pose.yaw
        return TeleopAction(x, z, yaw)

    def get_node_action(self, step_id: StepId, action_id: ActionId):
        node = self._node(step_id)
        idx = int(action_id)
        if idx < 0 or idx >= len(node.actions):
            raise IndexError(f"get_node_action invalid index: {idx}")
        a = node.actions[idx]
        return PolarAction(r=float(a.r), theta=float(a.theta))
