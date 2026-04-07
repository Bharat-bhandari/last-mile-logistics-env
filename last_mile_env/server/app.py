import os
import random
from typing import Optional, Dict, Any
import uvicorn
import argparse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync' or 'pip install openenv'"
    ) from e

try:
    from ..models import LastMileAction, LastMileObservation
    from .last_mile_env_environment import LastMileEnvironment
    from ..tasks import Task1Easy, Task2Medium, Task3Hard
except (ImportError, ModuleNotFoundError):
    from models import LastMileAction, LastMileObservation
    from server.last_mile_env_environment import LastMileEnvironment
    from tasks import Task1Easy, Task2Medium, Task3Hard

# Task Registry
TASK_REGISTRY = {
    "easy": Task1Easy(),
    "medium": Task2Medium(),
    "hard": Task3Hard(),
}


class ManagedLastMileEnvironment(LastMileEnvironment):
    """
    Extends LastMileEnvironment to support task-based resets.
    Reads task_id from the options dict passed to reset, falling back to
    the LMLC_TASK environment variable, then defaulting to 'easy'.
    """

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Any:
        # Base reset: clears vehicles, orders, resets state
        super().reset()

        # Determine task: options dict > env var > default
        task_name = "easy"
        if options and "task_id" in options:
            task_name = options["task_id"]
        else:
            task_name = os.getenv("LMLC_TASK", "easy").lower()

        task = TASK_REGISTRY.get(task_name, TASK_REGISTRY["easy"])
        scenario = task.get_init_state()

        # Apply scenario
        self.vehicles = scenario["vehicles"]
        self.orders = scenario["orders"]

        # Seed RNG deterministically for this task
        seed = scenario.get("seed", 42)
        self._rng = random.Random(seed)

        # Apply traffic config
        traffic_config = scenario.get("traffic_config", "dynamic_medium")
        if traffic_config == "static_low":
            for k in self.traffic_multipliers:
                self.traffic_multipliers[k] = 1.0
        # Other configs start at 1.0 and evolve via _update_traffic()

        return self._get_obs(reward=0.0)


# Create the OpenEnv FastAPI application instance
app = create_app(
    ManagedLastMileEnvironment,
    LastMileAction,
    LastMileObservation,
    env_name="last_mile_env",
    max_concurrent_envs=5,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for running the server locally."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LMLC Environment Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    main(host=args.host, port=args.port)