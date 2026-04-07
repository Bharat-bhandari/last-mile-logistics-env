from typing import List, Dict, Any
from .models import Order, Vehicle, VehicleStatus


class BaseGrader:
    """Deterministic scorer for Logistics Tasks."""

    def calculate_score(
        self,
        vehicles: List[Vehicle],
        orders: List[Order],
        total_steps: int,
    ) -> float:
        if not orders:
            return 0.0

        delivered_count = sum(1 for o in orders if o.status == "delivered")
        on_time_count = sum(
            1 for o in orders if o.status == "delivered" and o.deadline >= total_steps
        )

        # Weighted Score: 70% Delivery Success, 30% On-Time Efficiency
        score = (0.7 * (delivered_count / len(orders))) + (
            0.3 * (on_time_count / len(orders))
        )
        return round(min(max(score, 0.0), 1.0), 2)


class Task1Easy:
    """Scenario: Single delivery, low traffic, generous deadline."""

    SEED = 100

    def get_init_state(self) -> Dict[str, Any]:
        return {
            "vehicles": [
                Vehicle(
                    id="v1",
                    location_node=0,
                    capacity=5,
                    current_load=[],
                    status=VehicleStatus.IDLE,
                )
            ],
            "orders": [
                Order(
                    id="easy_1",
                    pickup_node=0,
                    dropoff_node=2,
                    deadline=100,
                    priority=1,
                    status="queued",
                )
            ],
            "traffic_config": "static_low",
            "seed": self.SEED,
        }


class Task2Medium:
    """Scenario: Multiple orders, dynamic traffic spikes on SV Road (Node 1)."""

    SEED = 200

    def get_init_state(self) -> Dict[str, Any]:
        return {
            "vehicles": [
                Vehicle(
                    id="v1",
                    location_node=0,
                    capacity=5,
                    current_load=[],
                    status=VehicleStatus.IDLE,
                ),
                Vehicle(
                    id="v2",
                    location_node=2,
                    capacity=5,
                    current_load=[],
                    status=VehicleStatus.IDLE,
                ),
            ],
            "orders": [
                Order(
                    id="med_1",
                    pickup_node=0,
                    dropoff_node=3,
                    deadline=60,
                    priority=2,
                    status="queued",
                ),
                Order(
                    id="med_2",
                    pickup_node=1,
                    dropoff_node=0,
                    deadline=80,
                    priority=3,
                    status="queued",
                ),
            ],
            "traffic_config": "dynamic_medium",
            "seed": self.SEED,
        }


class Task3Hard:
    """Scenario: High-pressure logistics. Tight deadlines and non-stationary traffic."""

    SEED = 300

    def get_init_state(self) -> Dict[str, Any]:
        return {
            "vehicles": [
                Vehicle(
                    id="v1",
                    location_node=0,
                    capacity=10,
                    current_load=[],
                    status=VehicleStatus.IDLE,
                )
            ],
            "orders": [
                Order(
                    id="h1",
                    pickup_node=0,
                    dropoff_node=3,
                    deadline=30,
                    priority=3,
                    status="queued",
                ),
                Order(
                    id="h2",
                    pickup_node=3,
                    dropoff_node=1,
                    deadline=50,
                    priority=2,
                    status="queued",
                ),
                Order(
                    id="h3",
                    pickup_node=2,
                    dropoff_node=0,
                    deadline=70,
                    priority=1,
                    status="queued",
                ),
            ],
            "traffic_config": "extreme_stochastic",
            "seed": self.SEED,
        }