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

    The reset() signature matches OpenEnv's Environment ABC so that the
    framework's _get_valid_kwargs correctly passes all client-supplied
    kwargs (including task_id) through to this method.
    """

    def reset(self, seed=None, episode_id=None, **kwargs) -> Any:
        # Base reset: clears vehicles, orders, resets state
        super().reset(seed=seed, episode_id=episode_id, **kwargs)

        # Determine task: kwargs > env var > default
        task_name = kwargs.get("task_id", os.getenv("LMLC_TASK", "easy")).lower()

        task = TASK_REGISTRY.get(task_name, TASK_REGISTRY["easy"])
        scenario = task.get_init_state()

        # Apply scenario
        self.vehicles = scenario["vehicles"]
        self.orders = scenario["orders"]

        # Seed RNG deterministically for this task
        task_seed = scenario.get("seed", seed or 42)
        self._rng = random.Random(task_seed)

        # Apply traffic config
        self.traffic_config = scenario.get("traffic_config", "dynamic_medium")
        if self.traffic_config == "static_low":
            for k in self.traffic_multipliers:
                self.traffic_multipliers[k] = 1.0
        # Other configs start at 1.0 and evolve via _update_traffic()

        import logging
        logging.getLogger("LastMileEnv").info(
            f"Reset complete — task={task_name}, "
            f"vehicles={len(self.vehicles)}, orders={len(self.orders)}, "
            f"traffic_config={self.traffic_config}"
        )

        return self._get_obs(reward=0.0)


# Create the OpenEnv FastAPI application instance
app = create_app(
    ManagedLastMileEnvironment,
    LastMileAction,
    LastMileObservation,
    env_name="last_mile_env",
    max_concurrent_envs=5,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for running the server locally."""
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()