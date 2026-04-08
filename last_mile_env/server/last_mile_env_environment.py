import random
import logging
from uuid import uuid4
from typing import List, Dict, Optional, Any
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LastMileEnv")

try:
    from ..models import (
        LastMileAction,
        LastMileObservation,
        Vehicle,
        Order,
        VehicleStatus,
        ActionType,
    )
    from ..tasks import BaseGrader
except ImportError:
    from models import (
        LastMileAction,
        LastMileObservation,
        Vehicle,
        Order,
        VehicleStatus,
        ActionType,
    )
    from tasks import BaseGrader


class LastMileEnvironment(Environment):
    """
    LMLC Environment: A non-stationary logistics simulator for Santacruz, Mumbai.
    8-node directed graph with dynamic traffic multipliers, fuel system,
    and explicit pickup/delivery actions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Graph topology ──────────────────────────────────────────────────
    # 8-node Santacruz network with bypass routes.
    #
    #  (7) Vile_Parle_Link ──→ (2) Linking_Road
    #    ↑                           ↓  ↑
    #  (0) Station ──→ (1) SV_Road ──→ (2) ──→ (3) Juhu_Tara
    #    ↓                ↑   ↓
    #  (4) Vakola ──→ (5) Kalina ──→ (6) BKC_Connector
    #                                  ↓
    #                                (3) Juhu_Tara
    #
    # Main corridor: 0→1→2→3  (short but traffic-heavy on edges touching node 1)
    # Bypass south : 0→4→5→6→3  (longer, less traffic)
    # Bypass north : 0→7→2→3  (medium length)

    NODES = {
        0: "Station",
        1: "SV_Road",
        2: "Linking_Road",
        3: "Juhu_Tara",
        4: "Vakola",
        5: "Kalina",
        6: "BKC_Connector",
        7: "Vile_Parle_Link",
    }

    # Directed adjacency list  — { from: [ {to, base_time}, … ] }
    ADJ_LIST: Dict[int, List[Dict[str, Any]]] = {
        0: [{"to": 1, "base_time": 4}, {"to": 4, "base_time": 5}, {"to": 7, "base_time": 6}],
        1: [{"to": 0, "base_time": 4}, {"to": 2, "base_time": 3}, {"to": 5, "base_time": 5}],
        2: [{"to": 1, "base_time": 3}, {"to": 3, "base_time": 5}, {"to": 7, "base_time": 6}],
        3: [{"to": 2, "base_time": 5}, {"to": 6, "base_time": 4}],
        4: [{"to": 0, "base_time": 5}, {"to": 5, "base_time": 3}],
        5: [{"to": 4, "base_time": 3}, {"to": 1, "base_time": 5}, {"to": 6, "base_time": 4}],
        6: [{"to": 5, "base_time": 4}, {"to": 3, "base_time": 5}],
        7: [{"to": 0, "base_time": 6}, {"to": 2, "base_time": 4}],
    }

    # Edges that are on the "main corridor" — higher traffic spike probability
    MAIN_CORRIDOR_EDGES = {"0_1", "1_0", "1_2", "2_1"}

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_steps = 200
        self.nodes = dict(self.NODES)
        self.adj_list = {k: [dict(e) for e in v] for k, v in self.ADJ_LIST.items()}
        self.vehicles: List[Vehicle] = []
        self.orders: List[Order] = []
        self.traffic_multipliers: Dict[str, float] = {}
        self.traffic_config: str = "dynamic_medium"
        self.grader = BaseGrader()
        self._rng = random.Random(42)
        self._init_traffic()

    # ── Traffic ──────────────────────────────────────────────────────────

    def _init_traffic(self):
        for u, edges in self.adj_list.items():
            for edge in edges:
                self.traffic_multipliers[f"{u}_{edge['to']}"] = 1.0

    def _update_traffic(self):
        """Update traffic based on the current traffic_config profile."""
        cfg = self.traffic_config

        if cfg == "static_low":
            # No changes — traffic stays at 1.0
            return

        for key in self.traffic_multipliers:
            old_val = self.traffic_multipliers[key]
            is_main = key in self.MAIN_CORRIDOR_EDGES

            if cfg == "extreme_stochastic":
                # Aggressive spikes, especially on main corridor
                spike_prob = 0.40 if is_main else 0.20
                if self._rng.random() < spike_prob:
                    if is_main:
                        self.traffic_multipliers[key] = self._rng.uniform(3.0, 8.0)
                    else:
                        self.traffic_multipliers[key] = self._rng.uniform(1.5, 4.0)
                    logger.info(
                        f"Traffic spike [{cfg}]: {key} changed from "
                        f"{old_val:.2f} to {self.traffic_multipliers[key]:.2f}"
                    )
                else:
                    # Slow decay
                    self.traffic_multipliers[key] = max(
                        1.0, self.traffic_multipliers[key] * 0.92
                    )
            else:
                # dynamic_medium (default)
                spike_prob = 0.15 if is_main else 0.05
                if self._rng.random() < spike_prob:
                    if is_main:
                        self.traffic_multipliers[key] = self._rng.uniform(1.5, 3.0)
                    else:
                        self.traffic_multipliers[key] = self._rng.uniform(1.2, 2.0)
                    logger.info(
                        f"Traffic spike [{cfg}]: {key} changed from "
                        f"{old_val:.2f} to {self.traffic_multipliers[key]:.2f}"
                    )
                else:
                    self.traffic_multipliers[key] = max(
                        1.0, self.traffic_multipliers[key] * 0.9
                    )

    # ── Vehicle movement ─────────────────────────────────────────────────

    def _move_vehicles(self):
        for v in self.vehicles:
            if v.status == VehicleStatus.BROKEN:
                continue

            if v.status == VehicleStatus.MOVING and v.destination_node is not None:
                # Calculate travel time on first tick
                if v.time_to_arrival <= 0:
                    edge_key = f"{v.location_node}_{v.destination_node}"
                    traffic = self.traffic_multipliers.get(edge_key, 1.0)
                    base = next(
                        (
                            e["base_time"]
                            for e in self.adj_list.get(v.location_node, [])
                            if e["to"] == v.destination_node
                        ),
                        5,
                    )
                    v.time_to_arrival = base * traffic
                    logger.info(
                        f"Vehicle {v.id} started moving {v.location_node}→"
                        f"{v.destination_node}. ETR: {v.time_to_arrival:.1f} "
                        f"(base={base}, traffic={traffic:.2f})"
                    )

                # Burn fuel while moving
                v.fuel -= 1.0 * self.traffic_multipliers.get(
                    f"{v.location_node}_{v.destination_node}", 1.0
                )

                # Progress: 1 step = 1 time-unit
                v.time_to_arrival -= 1.0

                if v.time_to_arrival <= 0:
                    logger.info(f"Vehicle {v.id} arrived at node {v.destination_node}")
                    v.time_to_arrival = 0.0
                    v.location_node = v.destination_node
                    v.destination_node = None
                    v.status = VehicleStatus.IDLE

            elif v.status == VehicleStatus.IDLE:
                # Idle vehicles still burn a small amount of fuel
                v.fuel -= 0.5

            # Check for fuel exhaustion
            if v.fuel <= 0:
                v.fuel = 0.0
                v.status = VehicleStatus.BROKEN
                v.destination_node = None
                v.time_to_arrival = 0.0
                logger.warning(f"Vehicle {v.id} ran out of fuel and is BROKEN!")

    # ── Adjacency check ──────────────────────────────────────────────────

    def _is_adjacent(self, from_node: int, to_node: int) -> bool:
        return any(
            e["to"] == to_node for e in self.adj_list.get(from_node, [])
        )

    # ── Episode checks ───────────────────────────────────────────────────

    def _check_all_delivered(self) -> bool:
        return all(o.status == "delivered" for o in self.orders) and len(self.orders) > 0

    # ── Core API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> LastMileObservation:
        """Reset the environment to initial state."""
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self.vehicles = []
        self.orders = []
        self.traffic_config = "dynamic_medium"
        self._rng = random.Random(seed or 42)
        self._init_traffic()
        return self._get_obs(reward=0.0)

    def step(self, action: LastMileAction) -> LastMileObservation:
        self._state.step_count += 1
        logger.info(
            f"Step {self._state.step_count}: action={action.action_type} "
            f"vehicle={action.vehicle_id} target={action.target_node} "
            f"order={action.order_id}"
        )
        self._update_traffic()

        # --- Apply Action ---
        reward = -0.1  # per-step operational cost

        vehicle = next((v for v in self.vehicles if v.id == action.vehicle_id), None)
        if vehicle is None:
            logger.error(f"Vehicle {action.vehicle_id} not found")
        elif vehicle.status == VehicleStatus.BROKEN:
            logger.warning(f"Vehicle {vehicle.id} is BROKEN — ignoring action")
            reward -= 1.0
        else:
            # ── ASSIGN ───────────────────────────────────────────────
            if action.action_type == ActionType.ASSIGN and action.target_node is not None:
                if vehicle.status == VehicleStatus.IDLE:
                    if self._is_adjacent(vehicle.location_node, action.target_node):
                        logger.info(f"ASSIGN {vehicle.id} → node {action.target_node}")
                        vehicle.destination_node = action.target_node
                        vehicle.status = VehicleStatus.MOVING
                        vehicle.time_to_arrival = 0
                    else:
                        logger.warning(
                            f"ASSIGN rejected: {action.target_node} not adjacent "
                            f"to {vehicle.location_node}"
                        )
                        reward -= 0.5  # penalty for invalid action
                else:
                    logger.warning(f"ASSIGN rejected: {vehicle.id} is {vehicle.status}")

            # ── REROUTE ──────────────────────────────────────────────
            elif action.action_type == ActionType.REROUTE and action.target_node is not None:
                if vehicle.status == VehicleStatus.MOVING:
                    if self._is_adjacent(vehicle.location_node, action.target_node):
                        if vehicle.destination_node != action.target_node:
                            logger.info(f"REROUTE {vehicle.id} → node {action.target_node}")
                            vehicle.destination_node = action.target_node
                            vehicle.time_to_arrival = 0
                            reward -= 2.0  # rerouting penalty
                    else:
                        logger.warning(
                            f"REROUTE rejected: {action.target_node} not adjacent "
                            f"to {vehicle.location_node}"
                        )
                elif vehicle.status == VehicleStatus.IDLE:
                    if self._is_adjacent(vehicle.location_node, action.target_node):
                        logger.info(f"REROUTE (idle) {vehicle.id} → node {action.target_node}")
                        vehicle.destination_node = action.target_node
                        vehicle.status = VehicleStatus.MOVING
                        vehicle.time_to_arrival = 0
                    else:
                        logger.warning(
                            f"REROUTE rejected: {action.target_node} not adjacent "
                            f"to {vehicle.location_node}"
                        )

            # ── WAIT ─────────────────────────────────────────────────
            elif action.action_type == ActionType.WAIT:
                logger.info(f"WAIT — Vehicle {vehicle.id} holds position")
                if vehicle.status == VehicleStatus.IDLE:
                    vehicle.destination_node = None

            # ── PICKUP ───────────────────────────────────────────────
            elif action.action_type == ActionType.PICKUP:
                if vehicle.status != VehicleStatus.IDLE:
                    logger.warning(f"PICKUP rejected: {vehicle.id} is {vehicle.status}")
                    reward -= 0.5
                elif action.order_id is None:
                    logger.warning("PICKUP rejected: no order_id specified")
                    reward -= 0.5
                else:
                    order = next((o for o in self.orders if o.id == action.order_id), None)
                    if order is None:
                        logger.warning(f"PICKUP rejected: order {action.order_id} not found")
                        reward -= 0.5
                    elif order.status != "queued":
                        logger.warning(
                            f"PICKUP rejected: order {order.id} status is {order.status}"
                        )
                        reward -= 0.5
                    elif order.pickup_node != vehicle.location_node:
                        logger.warning(
                            f"PICKUP rejected: vehicle at {vehicle.location_node}, "
                            f"order pickup at {order.pickup_node}"
                        )
                        reward -= 0.5
                    elif len(vehicle.current_load) >= vehicle.capacity:
                        logger.warning(f"PICKUP rejected: vehicle {vehicle.id} at capacity")
                        reward -= 0.5
                    else:
                        order.status = "assigned"
                        vehicle.current_load.append(order.id)
                        reward += 2.0  # pickup bonus
                        logger.info(
                            f"PICKUP: {vehicle.id} loaded order {order.id} "
                            f"at node {vehicle.location_node}"
                        )

            # ── DELIVER ──────────────────────────────────────────────
            elif action.action_type == ActionType.DELIVER:
                if vehicle.status != VehicleStatus.IDLE:
                    logger.warning(f"DELIVER rejected: {vehicle.id} is {vehicle.status}")
                    reward -= 0.5
                elif action.order_id is None:
                    logger.warning("DELIVER rejected: no order_id specified")
                    reward -= 0.5
                else:
                    order = next((o for o in self.orders if o.id == action.order_id), None)
                    if order is None:
                        logger.warning(f"DELIVER rejected: order {action.order_id} not found")
                        reward -= 0.5
                    elif order.status != "assigned":
                        logger.warning(
                            f"DELIVER rejected: order {order.id} status is {order.status}"
                        )
                        reward -= 0.5
                    elif order.id not in vehicle.current_load:
                        logger.warning(
                            f"DELIVER rejected: order {order.id} not in vehicle {vehicle.id}"
                        )
                        reward -= 0.5
                    elif order.dropoff_node != vehicle.location_node:
                        logger.warning(
                            f"DELIVER rejected: vehicle at {vehicle.location_node}, "
                            f"order dropoff at {order.dropoff_node}"
                        )
                        reward -= 0.5
                    else:
                        order.status = "delivered"
                        vehicle.current_load.remove(order.id)
                        # Bonus scaled by priority; late deliveries get less
                        on_time = self._state.step_count <= order.deadline
                        bonus = 10.0 * order.priority if on_time else 3.0
                        reward += bonus
                        logger.info(
                            f"DELIVER: {vehicle.id} delivered order {order.id} "
                            f"at node {vehicle.location_node} "
                            f"({'on-time' if on_time else 'LATE'}, +{bonus:.1f})"
                        )

        # ── Move vehicles / burn fuel ────────────────────────────────
        self._move_vehicles()

        # ── Mark late orders ─────────────────────────────────────────
        for o in self.orders:
            if o.status in ("queued", "assigned") and self._state.step_count > o.deadline:
                if o.status != "late":
                    logger.warning(f"Order {o.id} is past deadline ({o.deadline})!")

        # ── Episode termination ──────────────────────────────────────
        done = False
        metadata: Dict[str, Any] = {}
        if self._check_all_delivered():
            done = True
            score = self.grader.calculate_score(
                self.vehicles, self.orders, self._state.step_count
            )
            metadata["final_score"] = score
            reward += 20.0
            logger.info(f"All orders delivered! Final score: {score}")
        elif self._state.step_count >= self.max_steps:
            done = True
            score = self.grader.calculate_score(
                self.vehicles, self.orders, self._state.step_count
            )
            metadata["final_score"] = score
            logger.info(f"Max steps reached. Final score: {score}")
        # Check if all vehicles broken and undelivered orders remain
        elif all(v.status == VehicleStatus.BROKEN for v in self.vehicles):
            remaining = [o for o in self.orders if o.status != "delivered"]
            if remaining:
                done = True
                score = self.grader.calculate_score(
                    self.vehicles, self.orders, self._state.step_count
                )
                metadata["final_score"] = score
                reward -= 10.0
                logger.info(f"All vehicles broken! Final score: {score}")

        obs = self._get_obs(reward=reward, done=done, metadata=metadata)
        return obs

    def _get_obs(
        self,
        reward: float,
        done: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LastMileObservation:
        if done is None:
            done = self._state.step_count >= self.max_steps

        return LastMileObservation(
            timestep=self._state.step_count,
            vehicles=self.vehicles,
            active_orders=self.orders,
            traffic_map=self.traffic_multipliers,
            graph_adjacency=self.adj_list,
            done=done,
            reward=reward,
            metadata=metadata or {},
        )

    @property
    def state(self) -> State:
        return self._state