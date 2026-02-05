from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

import math
import networkx as nx

from .sim_wrapper import PolarAction, TeleopAction


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

    Key capabilities:
      - record nodes (pose + action proposals)
      - parent links (tree edges)
      - track unexplored action indices per node
      - find a previously visited node with remaining unexplored actions
      - compute an undirected path between two nodes in the tree
      - convert a node-path into MOVE_POLAR primitives (r, theta)
    """

    def __init__(self):
        self.G = nx.DiGraph()

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
        initialize_unexplored: bool = True,
        exclude_action0: bool = True,
    ) -> None:
        """
        Add or overwrite a node.

        Inputs:
          step_id: node id (usually your step index)
          pose: Pose2D at that step
          actions: list of ActionProposal produced at that step
          gsv: optional global semantic value for this step
          initialize_unexplored: if True, fill unexplored from actions
          exclude_action0: if True, do not include index==0 in unexplored

        Output:
          None (mutates graph)
        """
        actions = actions or []
        data = NodeData(step=step_id, pose=pose, actions=actions, gsv=gsv)

        if initialize_unexplored:
            idxs = [a.index for a in actions]
            if exclude_action0:
                idxs = [i for i in idxs if i != 0]
            data.unexplored = set(idxs)

        self.G.add_node(step_id, data=data)
        if (scores is not None) and (gsv is not None):
            self.fill_in_scores(step_id, gsv, scores)


    def link_parent(self, parent_id: StepId, child_id: StepId) -> None:
        """
        Add a directed tree edge parent -> child.

        Inputs:
          parent_id: existing node id (or will be auto-created if missing)
          child_id: existing node id (or will be auto-created if missing)

        Output:
          None (mutates graph)

        Notes:
          NetworkX will auto-add missing nodes, but they won't have NodeData.
          In practice, call add_step() for both before linking.
        """
        self.G.add_edge(parent_id, child_id, kind="tree")

    # --------- per-node exploration state ---------

    def mark_tried(self, step_id: StepId, action_id: ActionId):
        """
        Mark an action as tried (remove from unexplored set).

        Inputs:
          step_id: node where action was tried
          action_id: discrete action index tried

        Output:
          None
        """
        self._node(step_id).unexplored.discard(action_id)

    def get_action_score(
        self,
        step_id: StepId,
        action_id: ActionId,
        default: float = 0.0,
    ) -> float:
        """
        Get the score for (step_id, action_id).

        Priority:
          1) adjusted score (if prefer_adjusted=True and available)
          2) raw score
          3) default

        Inputs:
          step_id: node id
          action_id: discrete action index (1..N)
          prefer_adjusted: if True, use adjusted when available
          default: fallback value if neither score exists

        Output:
          score (float)
        """
        node = self._node(step_id)

        # action_id is 1..N; list index is 0..N-1
        idx = int(action_id) - 1
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

        Inputs:
          step_id: node id

        Output:
          sorted list of action indices (possibly empty)
        """
        return sorted(self._node(step_id).unexplored)

    def find_node_with_unexplored_actions(self, start_step_id: StepId) -> Optional[StepId]:
        """
        Recency-biased scan: from start_step_id down to 0, find first node
        that still has unexplored actions.

        Inputs:
          start_step_id: where to start scanning backward (e.g., current step)

        Output:
          step_id of the first node found with unexplored actions, or None.
        """
        for step in range(start_step_id, -1, -1):
            if step in self.G and len(self._node(step).unexplored) > 0:
                return step
        return None

    # --------- tree navigation helpers ---------

    def get_parent(self, step_id: StepId) -> Optional[StepId]:
        """
        Get the unique tree parent of step_id.

        Inputs:
          step_id: node id

        Output:
          parent id, or None if root

        Raises:
          RuntimeError if more than one parent exists (violates tree assumption).
        """
        preds = list(self.G.predecessors(step_id))
        if len(preds) > 1:
            raise RuntimeError(f"Tree can only have one parent! node={step_id} preds={preds}")
        return preds[0] if preds else None

    def path_from_a_to_b(self, a: StepId, b: StepId) -> Tuple[List[StepId]]:
        """
        Compute the unique undirected path between a and b in a directed tree.

        Inputs:
          a, b: node ids

        Output:
          list of node ids [a, ..., b]

        Raises:
          ValueError if a and b are not connected within the tree.
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

    # --------- path -> MOVE_POLAR primitives ---------

    def path_to_move_delta(self, path: List[StepId], in_reverse: bool=False) -> List[Tuple[float, float]]:
        """
        Convert a node-path [s0, s1, ..., sk] into a sequence of MOVE_POLAR commands.

        Input:
          path: list of StepId, consecutive nodes in the TREE path
          in_reverse: bool, if true the car first move backward and rotate

        Output:
          commands: list of (r, theta) where:
            - theta is delta-yaw to face the next node
            - r is straight-line distance to the next node
          Length = len(path) - 1

        Assumptions:
          - Pose2D is in a consistent world frame.
          - MOVE_POLAR executes: rotate(theta) then translate_forward(r).
          - We assume perfect execution (no drift), updating yaw to the travel heading each hop.
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

            # Update our "executed" pose estimate after this primitive
            # We assume we end up exactly at next_pose position and yaw aligned with desired heading.
            cur_pose = Pose2D(x=next_pose.x, y=next_pose.y, yaw=next_pose.yaw)

        return cmds

    def _action_score(self, node: NodeData, a: ActionProposal) -> float:
        """
        Default score used for action selection.
        Prefers `adjusted` if present, otherwise falls back to `score`, otherwise 0.
        You can replace this with anything.
        """
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
        Pick the best UNEXPLORED action at a given node.

        Inputs:
        step_id: node to expand
        score_threshold: ignore actions with score <= threshold
        exclude_action0: ignore index==0 if True
        score_fn: custom scoring (node, action) -> float

        Output:
        action_id (index) of best unexplored action, or None if none available.
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

    def fill_in_scores(self, step_id: StepId, gsv: float, action_scores: List[float]) -> None:
        """
        Fills per-action score and adjusted = score * gsv for actions 1..N at this node.

        Expected shapes:
        - If GPT includes action 0: len(action_scores) == N + 1  (index 0 is rewind)
        - If GPT excludes action 0: len(action_scores) == N
        We will robustly adapt if GPT returns too many / too few scores.

        NOTE: We do NOT normalize by default (your prompt says don't force sum to 1).
        """
        node = self._node(step_id)
        node.gsv = float(gsv)

        N = len(node.actions)
        M = len(action_scores)

        # --- sanitize scores (float + finite) ---
        scores = []
        for s in action_scores:
            try:
                v = float(s)
            except Exception:
                v = 0.0
            if not math.isfinite(v):
                v = 0.0
            scores.append(v)

        # --- decide whether action 0 is present ---
        # Prefer the common case: M == N+1 => includes action 0
        has_action0: Optional[bool] = None
        if M == N + 1:
            has_action0 = True
        elif M == N:
            has_action0 = False
        else:
            # Heuristic: if M > N, treat first as action0 and take next N
            # If M < N, assume no action0 and pad.
            has_action0 = (M > N)

        # if has_action0:
        #     # action 0 exists at index 0
        #     node.rewind_score = scores[0]  # optional; only if you want to store it
        #     scores_1_to_N = scores[1:1 + N]
        # else:
        scores_1_to_N = scores[:N]

        # --- pad or clip to exactly N ---
        if len(scores_1_to_N) < N:
            scores_1_to_N = scores_1_to_N + [0.0] * (N - len(scores_1_to_N))
        elif len(scores_1_to_N) > N:
            scores_1_to_N = scores_1_to_N[:N]

        # --- assign ---
        for i, a in enumerate(node.actions):
            s = scores_1_to_N[i]
            a.score = s
            a.adjusted = s * node.gsv

    @staticmethod
    def to_action_proposal(sorted_actions):
        """
        GPT action indices are 1..N (0 reserved for backtrack).
        Build ActionProposal list that matches those indices.
        """
        proposals = []
        for i, a in enumerate(sorted_actions):
            proposals.append(ActionProposal(index=i+1, r=float(a.r), theta=float(a.theta)))
        return proposals

    def local_backtrack(
        self,
        step_id: StepId,
        *,
        score_threshold: float = 0.0,
        exclude_action0: bool = True,
        score_fn: Optional[Callable[[NodeData, ActionProposal], float]] = None,) -> Optional[Tuple[int, int]]:
        """
        Returns (target_step_id, action_id) for local backtrack:
        go to parent, choose the unexplored action with the highest 
        score that's above threshold.
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
    
    def global_backtrack(
        self,
        step_id: StepId,
        *,
        score_threshold: float = 0.0,
        exclude_action0: bool = True,
        score_fn: Optional[Callable[[NodeData, ActionProposal], float]] = None,) -> Optional[Tuple[int, int]]:
        """
        Returns (target_step_id, action_id) for global backtrack:
        go to parent, choose the unexplored action with the highest 
        score that's above threshold.
        """

        target_step = self.select_global_target_recency_first(step_id, score_threshold=0.0)
        if target_step is None:
            # nowhere left to explore
            return None

        target_action_id = self.select_next_action_at(target_step, score_threshold, exclude_action0, score_fn)

        if target_action_id is None:
            # should be rare since select_global_target... ensured at least one, but be safe
            return None
        
        return (target_step, target_action_id)

    def get_node_position(self, step_id: StepId):
        pose = self._node(step_id).pose
        x, z, yaw = pose.x, pose.y, pose.yaw
        return TeleopAction(x, z, yaw)

    def get_node_action(self, step_id: StepId, action_id: ActionId):
        node = self._node(step_id)

        # action_id is 1..N; list index is 0..N-1
        idx = int(action_id) - 1

        if idx < 0 or idx >= len(node.actions):
            raise IndexError(
                f"get_node_action: step={step_id} action_id={action_id} "
                f"(idx={idx}) out of range, len(actions)={len(node.actions)}"
            )

        a = node.actions[idx]
        return PolarAction(r=float(a.r), theta=float(a.theta))
