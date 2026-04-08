import asyncio
import os
import textwrap
import json
from typing import List, Optional, Any

from openai import OpenAI
from dotenv import load_dotenv
from last_mile_env import LastMileEnv, LastMileAction

# Configuration
load_dotenv()

# Environment Variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("LMLC_TASK", "easy")
BENCHMARK = "last_mile_logistics_controller"

# Hyperparameters
MAX_STEPS = 200
TEMPERATURE = 0.2

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a state-aware logistics dispatcher in the Santacruz LMLC environment.

    **Graph (directed edges with base travel times):**
    - 0 (Station) → 1 (SV_Road) [5 min]
    - 1 (SV_Road) → 0 (Station) [5 min]
    - 1 (SV_Road) → 2 (Linking_Road) [4 min]
    - 2 (Linking_Road) → 1 (SV_Road) [4 min]
    - 2 (Linking_Road) → 3 (Juhu_Tara) [7 min]
    - 3 (Juhu_Tara) → 2 (Linking_Road) [7 min]

    STRICT DECISION POLICY (FOLLOW EXACTLY):

    Step 1: Check vehicle status

    * If status == "moving":
      → You MUST output:
      {"type": "WAIT", "payload": {}}
      → Do NOT assign or reroute

    Step 2: If status == "idle":

    * If vehicle is at pickup node of an undelivered order:
      → assign next hop toward drop node

    Step 3: Multi-hop routing rule

    * You CANNOT jump directly to destination
    * Example: 0 → 2 requires:
      Step A: assign to 1
      Step B: wait until arrival
      Step C: assign to 2

    Step 4: Reroute rule

    * Only reroute if traffic multiplier > 3.0
    * Otherwise NEVER reroute

    Step 5: Anti-loop rule (CRITICAL)

    * Do NOT repeat ASSIGN if vehicle is already moving
    * Repeated ASSIGN = invalid behavior

    Goal:
    Complete delivery for the active order with minimum steps.

    ---

    Before outputting action:

    * Check: Is vehicle moving?
    * If YES → WAIT
    * If NO → ASSIGN next valid hop

    ---

    Output ONLY valid JSON:
    {
    "type": "...",
    "payload": {...}
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, result: Optional[Any] = None) -> None:
    print(f"\n--- STEP {step} ---", flush=True)
    print(f"Action: {action}", flush=True)
    print(f"Reward: {reward:.2f}", flush=True)
    print(f"Done: {str(done).lower()}", flush=True)
    
    if result and hasattr(result, "observation"):
        obs = result.observation
        print("Vehicles Status:", flush=True)
        for v in obs.vehicles:
            dest = f" -> {v.destination_node}" if v.destination_node is not None else ""
            etr = f" (ETR: {v.time_to_arrival:.1f})" if v.status == "moving" else ""
            print(f"  - {v.id}: {v.location_node}{dest} [{v.status}]{etr}", flush=True)
        
        print("Active Orders:", flush=True)
        for o in obs.active_orders:
            print(f"  - {o.id}: {o.pickup_node}->{o.dropoff_node} [{o.status}] Priority: {o.priority} Deadline: {o.deadline}", flush=True)
        
        high_traffic = {k: v for k, v in obs.traffic_map.items() if v > 1.5}
        if high_traffic:
            print("High Traffic:", flush=True)
            for edge, mult in high_traffic.items():
                print(f"  - {edge}: {mult:.2f}x", flush=True)
    print("-" * 15, flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

def get_agent_action(client: OpenAI, obs_json: str) -> LastMileAction:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current State:\n{obs_json}"},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        
        print(f"\nDEBUG LLM Action Decision:", flush=True)
        print(f"Response: {content}", flush=True) 
        
        data = json.loads(content)
        
        # Ensure data is a dictionary before unpacking
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object")
            
        if "type" in data and "payload" in data:
            action_type = data.get("type", "wait").lower()
            payload = data.get("payload", {})
            target_node = payload.get("target_node")
            if target_node is None:
                target_node = payload.get("next_hop")
            if target_node is None:
                target_node = payload.get("next_node")
            if target_node is None:
                target_node = payload.get("destination_node")
                
            return LastMileAction(
                vehicle_id=payload.get("vehicle_id", "v1"),
                action_type=action_type,
                target_node=target_node,
                order_id=payload.get("order_id")
            )
            
        return LastMileAction(**data)

    except Exception as e:
        # CRITICAL: Print the error so you know if it's a Credit Limit issue
        print(f"[ERROR] Agent Action Failed: {e}", flush=True)
        
        # Fallback: Explicitly use None for optional fields to avoid Server validation errors
        return LastMileAction(
            vehicle_id="v1", 
            action_type="wait", 
            target_node=None, 
            order_id=None
        )

async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY not found in environment.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with LastMileEnv(base_url="http://localhost:8000") as env:
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

        try:
            result = await env.reset()

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_json = result.observation.model_dump_json()
                action = get_agent_action(client, obs_json)

                # Add a small delay to prevent Rate Limits
                await asyncio.sleep(1)

                result = await env.step(action)

                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action.action_type,
                    reward=reward,
                    done=result.done,
                    result=result,
                )

                if result.done:
                    # Extract grader score from metadata if available
                    if hasattr(result, "observation") and result.observation.metadata:
                        score = result.observation.metadata.get("final_score", 0.0)
                        success = score > 0.4
                    break

            # Fallback scoring if grader score not available
            if score == 0.0 and rewards:
                total_reward = sum(rewards)
                score = min(max(total_reward / 50.0, 0.0), 1.0)
                success = score > 0.4

        except Exception as e:
            print(f"[ERROR] Inference loop failed: {e}", flush=True)
        finally:
            log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass