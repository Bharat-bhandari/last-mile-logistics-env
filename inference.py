import asyncio
import os
import textwrap
import json
from typing import List, Optional

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
    You are a Senior Logistics Dispatcher for Santacruz, Mumbai.

    **Graph (directed edges with base travel times):**
    - 0 (Station) → 1 (SV_Road) [5 min]
    - 1 (SV_Road) → 0 (Station) [5 min]
    - 1 (SV_Road) → 2 (Linking_Road) [4 min]
    - 2 (Linking_Road) → 1 (SV_Road) [4 min]
    - 2 (Linking_Road) → 3 (Juhu_Tara) [7 min]
    - 3 (Juhu_Tara) → 2 (Linking_Road) [7 min]

    Actual travel time = base_time × traffic_multiplier for that edge.
    SV_Road (Node 1) edges frequently spike to 2x-5x multiplier.

    **Goal:** Deliver all orders before their deadlines.
    - On-time delivery: +20 reward
    - Late delivery: +5 reward
    - Order exceeding deadline undelivered: -10 penalty
    - Each step costs -0.1
    - Rerouting costs -2.0

    **Actions (one per step):**
    - "assign": Move idle vehicle to adjacent node. Requires vehicle_id, target_node.
    - "reroute": Redirect a moving vehicle to a different adjacent node. Requires vehicle_id, target_node.
    - "wait": Keep vehicle idle. Requires vehicle_id.

    **Strategy hints:**
    - Vehicles can only move to ADJACENT nodes (one hop at a time).
    - Plan multi-hop routes by issuing assign/reroute one hop at a time.
    - Avoid SV_Road (Node 1) when its traffic multiplier is high.
    - Pick up orders by having a vehicle arrive at the pickup_node.
    - Deliver orders by having a vehicle arrive at the dropoff_node while carrying the order.
    - Prioritize high-priority and tight-deadline orders.

    Reply ONLY with a JSON object:
    {
      "vehicle_id": "v1",
      "action_type": "assign" | "reroute" | "wait",
      "target_node": <int or null>,
      "order_id": "<str or null>"
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
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
        data = json.loads(content)
        return LastMileAction(**data)
    except Exception:
        # Fallback to WAIT if LLM fails or returns invalid JSON
        return LastMileAction(vehicle_id="v1", action_type="wait")


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

                result = await env.step(action)

                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action.action_type,
                    reward=reward,
                    done=result.done,
                    error=None,
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
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass