import logging
from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import LastMileAction, LastMileObservation

logger = logging.getLogger("LastMileClient")

class LastMileEnv(EnvClient[LastMileAction, LastMileObservation, State]):
    def _step_payload(self, action: LastMileAction) -> Dict:
        logger.info(f"Preparing payload for action: {action.action_type if action else 'None'}")
        # If action is None, return a valid dictionary that matches the model
        if action is None:
            return {
                "vehicle_id": "v1", 
                "action_type": "wait", # Ensure this matches your Enum string value
                "target_node": None,
                "order_id": None
            }
        
        # If it's a Pydantic model, use model_dump
        if hasattr(action, "model_dump"):
            return action.model_dump()
            
        # If it's already a dict, return it
        return action

    def _parse_result(self, payload: Dict) -> StepResult[LastMileObservation]:
        logger.info("Parsing step result from server")
        obs_data = payload.get("observation", {})

        # The OpenEnv framework's serialize_observation() strips `done`,
        # `reward`, and `metadata` from the observation sub-dict and places
        # them at the envelope level.  Re-inject so LastMileObservation
        # validation succeeds.
        obs_data.setdefault("done", payload.get("done", False))
        obs_data.setdefault("reward", payload.get("reward", 0.0))
        obs_data.setdefault("metadata", payload.get("metadata", {}))

        observation = LastMileObservation(**obs_data)
        
        logger.info(f"Result parsed: done={payload.get('done')}, reward={payload.get('reward')}")
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