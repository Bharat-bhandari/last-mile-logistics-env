import random
from uuid import uuid4
from typing import List, Dict, Optional, Any
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

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
    4-node directed graph with dynamic traffic multipliers.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_steps = 200
        self.nodes = {0: "Station", 1: "SV_Road", 2: "Linking_Road", 3: "Juhu_Tara"}
        self.adj_list = {
            0: [{"to": 1, "base_time": 5}],
            1: [{"to": 0, "base_time": 5}, {"to": 2, "base_time": 4}],
            2: [{"to": 1, "base_time": 4}, {"to": 3, "base_time": 7}],
            3: [{"to": 2, "base_time": 7}],
        }
        self.vehicles: List[Vehicle] = []
        self.orders: List[Order] = []
        self.traffic_multipliers: Dict[str, float] = {}
        self.grader = BaseGrader()
        self._rng = random.Random(42)
        self._init_traffic()

    def _init_traffic(self):
        for u, edges in self.adj_list.items():
            for edge in edges:
                self.traffic_multipliers[f"{u}_{edge['to']}"] = 1.0

    def _update_traffic(self):
        for key in self.traffic_multipliers:
            if self._rng.random() < 0.10:
                if "1" in key:
                    self.traffic_multipliers[key] = self._rng.uniform(2.0, 5.0)
                else:
                    self.traffic_multipliers[key] = self._rng.uniform(1.2, 2.5)
            else:
                self.traffic_multipliers[key] = max(
                    1.0, self.traffic_multipliers[key] * 0.9
                )

    def _move_vehicles(self):
        for v in self.vehicles:
            if v.status == VehicleStatus.MOVING and v.destination_node is not None:
                # If just started moving, set initial travel time
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

                # Progress: 1 step = 1 time-unit
                v.time_to_arrival -= 1.0

                if v.time_to_arrival <= 0:
                    v.time_to_arrival = 0.0
                    v.location_node = v.destination_node
                    v.destination_node = None
                    v.status = VehicleStatus.IDLE

    def _check_all_delivered(self) -> bool:
        """Return True if every order has been delivered."""
        return all(o.status == "delivered" for o in self.orders) and len(self.orders) > 0

    def _is_adjacent(self, from_node: int, to_node: int) -> bool:
        """Check if two nodes are directly connected."""
        return any(
            e["to"] == to_node for e in self.adj_list.get(from_node, [])
        )

    def reset(self) -> LastMileObservation:
        """Reset the environment to initial state."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.vehicles = []
        self.orders = []
        self._rng = random.Random(42)
        self._init_traffic()
        return self._get_obs(reward=0.0)

    def step(self, action: LastMileAction) -> LastMileObservation:
        self._state.step_count += 1
        self._update_traffic()

        # --- Apply Action ---
        reward = -0.1  # per-step operational cost

        vehicle = next((v for v in self.vehicles if v.id == action.vehicle_id), None)
        if vehicle is not None:
            if action.action_type == ActionType.ASSIGN and action.target_node is not None:
                # Assign: start moving vehicle to target node (must be adjacent)
                if vehicle.status != VehicleStatus.MOVING:
                    if self._is_adjacent(vehicle.location_node, action.target_node):
                        vehicle.destination_node = action.target_node
                        vehicle.status = VehicleStatus.MOVING
                        vehicle.time_to_arrival = 0  # will be computed in _move_vehicles

            elif action.action_type == ActionType.REROUTE and action.target_node is not None:
                # Reroute: redirect a moving vehicle to a new adjacent node
                if vehicle.status == VehicleStatus.MOVING:
                    if self._is_adjacent(vehicle.location_node, action.target_node):
                        vehicle.destination_node = action.target_node
                        vehicle.time_to_arrival = 0  # recompute travel time
                        reward -= 2.0  # rerouting penalty
                elif vehicle.status == VehicleStatus.IDLE:
                    # If idle, treat reroute as assign
                    if self._is_adjacent(vehicle.location_node, action.target_node):
                        vehicle.destination_node = action.target_node
                        vehicle.status = VehicleStatus.MOVING
                        vehicle.time_to_arrival = 0

            elif action.action_type == ActionType.WAIT:
                # Wait: keep vehicle idle at current location
                if vehicle.status != VehicleStatus.MOVING:
                    vehicle.status = VehicleStatus.IDLE
                    vehicle.destination_node = None

        # --- Move Vehicles ---
        self._move_vehicles()

        # --- Order Lifecycle ---
        for o in self.orders:
            if o.status == "queued":
                # Auto-pickup: if any vehicle is at pickup node and has capacity
                for v in self.vehicles:
                    if (
                        v.location_node == o.pickup_node
                        and v.status == VehicleStatus.IDLE
                        and len(v.current_load) < v.capacity
                    ):
                        o.status = "assigned"
                        v.current_load.append(o.id)
                        break

            elif o.status == "assigned":
                # Auto-deliver: if carrying vehicle is at dropoff node
                for v in self.vehicles:
                    if (
                        v.location_node == o.dropoff_node
                        and o.id in v.current_load
                    ):
                        o.status = "delivered"
                        v.current_load.remove(o.id)
                        # Reward: on-time vs late
                        if self._state.step_count <= o.deadline:
                            reward += 20.0
                        else:
                            reward += 5.0
                        break

        # --- Mark late orders that exceeded deadline but weren't delivered ---
        for o in self.orders:
            if o.status in ("queued", "assigned") and self._state.step_count > o.deadline:
                o.status = "late"
                reward -= 10.0

        # --- Determine done ---
        all_terminal = all(o.status in ("delivered", "late") for o in self.orders)
        done = (
            (all_terminal and len(self.orders) > 0)
            or self._state.step_count >= self.max_steps
        )

        obs = self._get_obs(reward=reward, done=done)

        # If episode ended, compute final grader score and add to metadata
        if done:
            final_score = self.grader.calculate_score(
                self.vehicles, self.orders, self._state.step_count
            )
            obs.metadata["final_score"] = final_score
            obs.metadata["grader"] = "BaseGrader"

        return obs

    def _get_obs(self, reward: float, done: Optional[bool] = None) -> LastMileObservation:
        if done is None:
            done = self._state.step_count >= self.max_steps

        return LastMileObservation(
            timestep=self._state.step_count,
            vehicles=self.vehicles,
            active_orders=self.orders,
            traffic_map=self.traffic_multipliers,
            done=done,
            reward=reward,
            metadata={},
        )

    @property
    def state(self) -> State:
        return self._state