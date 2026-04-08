import logging
from typing import List, Dict, Any
try:
    from .models import Order, Vehicle, VehicleStatus
except ImportError:
    from models import Order, Vehicle, VehicleStatus

logger = logging.getLogger("LastMileTasks")


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
        final_score = round(min(max(score, 0.0), 1.0), 2)
        logger.info(
            f"Final score: {final_score} "
            f"(Delivered: {delivered_count}/{len(orders)}, "
            f"On-time: {on_time_count}/{len(orders)}, Steps: {total_steps})"
        )
        return final_score


# =============================================================================
# Tasks — using 8-node Santacruz graph
#
#  Nodes:
#    0: Station          4: Vakola
#    1: SV_Road          5: Kalina
#    2: Linking_Road     6: BKC_Connector
#    3: Juhu_Tara        7: Vile_Parle_Link
#
#  Main corridor (traffic-heavy): 0→1→2→3
#  South bypass: 0→4→5→6→3
#  North bypass: 0→7→2→3
# =============================================================================


class Task1Easy:
    """Scenario: Single delivery, low traffic, generous deadline."""

    SEED = 100

    def get_init_state(self) -> Dict[str, Any]:
        logger.info("Initializing Easy Task")
        return {
            "vehicles": [
                Vehicle(
                    id="v1",
                    location_node=0,
                    capacity=5,
                    current_load=[],
                    status=VehicleStatus.IDLE,
                    fuel=100.0,
                )
            ],
            "orders": [
                Order(
                    id="easy_1",
                    pickup_node=0,
                    dropoff_node=3,
                    deadline=80,
                    priority=1,
                    status="queued",
                )
            ],
            "traffic_config": "static_low",
            "seed": self.SEED,
        }


class Task2Medium:
    """Scenario: Multiple orders, two vehicles, dynamic traffic on SV Road."""

    SEED = 200

    def get_init_state(self) -> Dict[str, Any]:
        logger.info("Initializing Medium Task")
        return {
            "vehicles": [
                Vehicle(
                    id="v1",
                    location_node=0,
                    capacity=5,
                    current_load=[],
                    status=VehicleStatus.IDLE,
                    fuel=100.0,
                ),
                Vehicle(
                    id="v2",
                    location_node=3,
                    capacity=5,
                    current_load=[],
                    status=VehicleStatus.IDLE,
                    fuel=100.0,
                ),
            ],
            "orders": [
                Order(
                    id="med_1",
                    pickup_node=0,
                    dropoff_node=3,
                    deadline=50,
                    priority=2,
                    status="queued",
                ),
                Order(
                    id="med_2",
                    pickup_node=3,
                    dropoff_node=0,
                    deadline=60,
                    priority=3,
                    status="queued",
                ),
            ],
            "traffic_config": "dynamic_medium",
            "seed": self.SEED,
        }


class Task3Hard:
    """
    Scenario: High-pressure logistics.
    - Single vehicle, 3 orders across the map.
    - Tight deadlines that are ONLY achievable if the agent avoids the
      congested main corridor (0→1→2→3) and uses bypasses.
    - extreme_stochastic traffic: 40% spike chance on SV Road edges (@3-8x).
    - Fuel constraint forces efficiency — no room for aimless exploration.
    """

    SEED = 300

    def get_init_state(self) -> Dict[str, Any]:
        logger.info("Initializing Hard Task")
        return {
            "vehicles": [
                Vehicle(
                    id="v1",
                    location_node=0,
                    capacity=10,
                    current_load=[],
                    status=VehicleStatus.IDLE,
                    fuel=80.0,  # Tighter fuel budget
                )
            ],
            "orders": [
                # Order 1: Station → Juhu_Tara (must cross the map)
                # Deadline is tight: main corridor @ 1x takes ~12 steps minimum,
                # but with 3-8x traffic on SV Road it can balloon to 30+
                Order(
                    id="h1",
                    pickup_node=0,
                    dropoff_node=3,
                    deadline=25,
                    priority=3,
                    status="queued",
                ),
                # Order 2: BKC_Connector → Station (reverse direction, south bypass)
                Order(
                    id="h2",
                    pickup_node=6,
                    dropoff_node=0,
                    deadline=50,
                    priority=2,
                    status="queued",
                ),
                # Order 3: Vile_Parle_Link → Kalina (cross-map lateral)
                Order(
                    id="h3",
                    pickup_node=7,
                    dropoff_node=5,
                    deadline=65,
                    priority=1,
                    status="queued",
                ),
            ],
            "traffic_config": "extreme_stochastic",
            "seed": self.SEED,
        }