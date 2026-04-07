from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import LastMileAction, LastMileObservation

class LastMileEnv(EnvClient[LastMileAction, LastMileObservation, State]):
    def _step_payload(self, action: LastMileAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[LastMileObservation]:
        obs_data = payload.get("observation", {})

        # The OpenEnv framework's serialize_observation() strips `done`,
        # `reward`, and `metadata` from the observation sub-dict and places
        # them at the envelope level.  Re-inject so LastMileObservation
        # validation succeeds.
        obs_data.setdefault("done", payload.get("done", False))
        obs_data.setdefault("reward", payload.get("reward", 0.0))

        observation = LastMileObservation(**obs_data)
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )