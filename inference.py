"""
inference.py – LLM-based agent for the Last-Mile Logistics Controller (LMLC).

Uses the OpenAI SDK to query an LLM at each environment step.  The LLM
receives a structured text prompt describing the current observation and
must return a JSON action conforming to the LastMileAction Pydantic model.

Anti-Ping-Pong Architecture:
  - BFS-based next-hop computation eliminates directional ambiguity.
  - Movement history tracking forbids backtracking.
  - Explicit ACTION DIRECTIVE tells the LLM exactly what to do.
  - When vehicle is MOVING, the LLM receives a pre-built WAIT JSON.
"""

import asyncio
import json
import os
import re
import sys
from collections import deque
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from last_mile_env import LastMileEnv, LastMileAction

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("[FATAL] HF_TOKEN environment variable is required but not set.", flush=True)
    sys.exit(1)

TASK_NAME = os.getenv("LMLC_TASK", "easy")
BENCHMARK = "last_mile_logistics_controller"
MAX_STEPS = 200

# ── OpenAI Client ────────────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Graph Definition (mirrors server) ────────────────────────────────────────

ADJ_LIST: Dict[int, List[int]] = {
    0: [1, 4, 7],
    1: [0, 2, 5],
    2: [1, 3, 7],
    3: [2, 6],
    4: [0, 5],
    5: [4, 1, 6],
    6: [5, 3],
    7: [0, 2],
}


def bfs_next_hop(start: int, goal: int) -> Optional[int]:
    """Return the first node on the shortest path from start to goal.
    Returns None if start == goal or no path exists."""
    if start == goal:
        return None
    visited = {start}
    # queue entries: (current_node, first_hop)
    queue = deque()
    for neighbor in ADJ_LIST.get(start, []):
        if neighbor == goal:
            return neighbor
        queue.append((neighbor, neighbor))
        visited.add(neighbor)
    while queue:
        current, first_hop = queue.popleft()
        for neighbor in ADJ_LIST.get(current, []):
            if neighbor == goal:
                return first_hop
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, first_hop))
    return None  # no path


def bfs_shortest_path(start: int, goal: int) -> List[int]:
    """Return the full shortest path from start to goal (inclusive)."""
    if start == goal:
        return [start]
    visited = {start}
    queue = deque()
    queue.append((start, [start]))
    while queue:
        current, path = queue.popleft()
        for neighbor in ADJ_LIST.get(current, []):
            if neighbor == goal:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return [start]  # fallback


# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a strict logistics dispatcher. You control ONE vehicle in a delivery network.
You MUST follow the decision tree below EXACTLY. No creativity. No deviation.

## STRICT DECISION TREE (follow in order, stop at the FIRST match)

1. IF vehicle status == "moving" → output WAIT. NOTHING ELSE. EVER.
2. IF vehicle status == "broken" → output WAIT.
3. IF vehicle status == "idle":
   a. IF you are at the dropoff_node of an order you are carrying → output DELIVER for that order.
   b. IF you are at the pickup_node of a queued order and you are NOT carrying it → output PICKUP for that order.
   c. IF you are carrying an order → output ASSIGN to the RECOMMENDED NEXT HOP (given in the observation).
   d. IF you are NOT carrying anything and there is a queued order → output ASSIGN to the RECOMMENDED NEXT HOP toward the pickup node.
   e. IF nothing else applies → output WAIT.

## CRITICAL RULES

- WHEN MOVING: YOU MUST OUTPUT WAIT. Any other action wastes fuel and gets rejected.
- BACKTRACKING IS FORBIDDEN: If "PREVIOUS NODE" is listed, NEVER assign to that node.
- Only move to ADJACENT nodes (listed in the observation).
- Follow the RECOMMENDED action from the ACTION DIRECTIVE section exactly.

## OUTPUT FORMAT
Return ONLY a single JSON object. No explanation, no markdown, no extra text.
Examples:
{"vehicle_id": "v1", "action_type": "wait"}
{"vehicle_id": "v1", "action_type": "pickup", "order_id": "easy_1"}
{"vehicle_id": "v1", "action_type": "assign", "target_node": 2}
{"vehicle_id": "v1", "action_type": "deliver", "order_id": "easy_1"}
"""


# ── Observation Serializer ───────────────────────────────────────────────────

def serialize_observation(
    obs_dict: Dict[str, Any],
    previous_node: Optional[int] = None,
) -> str:
    """Convert an observation dict into a structured text prompt for the LLM.

    Includes an ACTION DIRECTIVE section that tells the LLM exactly what to do,
    eliminating ambiguity and preventing ping-pong behavior.
    """
    lines: List[str] = []
    lines.append(f"=== TIMESTEP {obs_dict.get('timestep', '?')} ===\n")

    # ── Vehicles ─────────────────────────────────────────────────────
    vehicles = obs_dict.get("vehicles", [])
    lines.append("VEHICLES:")
    for v in vehicles:
        dest = f" → node {v['destination_node']}" if v.get("destination_node") is not None else ""
        etr = f" (ETA: {v['time_to_arrival']:.1f} steps)" if v.get("status") == "moving" else ""
        load = f", carrying: {v['current_load']}" if v.get("current_load") else ", carrying: NOTHING"
        lines.append(
            f"  {v['id']}: at node {v['location_node']}{dest} | "
            f"status={v['status']}{etr} | fuel={v.get('fuel', 0):.1f} | "
            f"capacity={v['capacity']}{load}"
        )

    # ── Orders ───────────────────────────────────────────────────────
    lines.append("\nORDERS:")
    for o in obs_dict.get("active_orders", []):
        lines.append(
            f"  {o['id']}: pickup=node {o['pickup_node']} → dropoff=node {o['dropoff_node']} | "
            f"status={o['status']} | priority={o['priority']} | deadline=step {o['deadline']}"
        )

    # ── Traffic (only noteworthy edges) ──────────────────────────────
    traffic = obs_dict.get("traffic_map", {})
    high_traffic = {k: v for k, v in traffic.items() if v > 1.0}
    if high_traffic:
        lines.append("\nTRAFFIC (edges with multiplier > 1.0):")
        for edge, mult in sorted(high_traffic.items(), key=lambda x: -x[1]):
            severity = "🔴 HEAVY" if mult > 3.0 else "🟡 moderate"
            lines.append(f"  edge {edge}: {mult:.2f}x ({severity})")
    else:
        lines.append("\nTRAFFIC: all edges clear (1.0x)")

    # ── ACTION DIRECTIVE (the key anti-ping-pong section) ────────────
    lines.append("\n" + "=" * 50)
    lines.append("=== ACTION DIRECTIVE (FOLLOW THIS EXACTLY) ===")
    lines.append("=" * 50)

    if not vehicles:
        lines.append("ERROR: No vehicles found.")
        return "\n".join(lines)

    v = vehicles[0]  # Primary vehicle
    vid = v["id"]
    status = v["status"]
    location = v["location_node"]
    load = v.get("current_load", [])
    neighbors = ADJ_LIST.get(location, [])

    if status == "moving":
        # ── MOVING: Force WAIT ───────────────────────────────────────
        dest = v.get("destination_node", "?")
        eta = v.get("time_to_arrival", "?")
        lines.append(f"Vehicle Status: MOVING to Node {dest} (ETA: {eta} steps)")
        lines.append("⛔ YOU MUST OUTPUT WAIT. The vehicle is in transit.")
        lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "wait"}}')

    elif status == "broken":
        lines.append("Vehicle Status: BROKEN")
        lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "wait"}}')

    elif status == "idle":
        lines.append(f"Vehicle Status: IDLE at Node {location}")
        lines.append(f"Adjacent nodes: {neighbors}")
        if previous_node is not None:
            lines.append(f"Previous node: {previous_node} (⛔ FORBIDDEN — do NOT go back here)")
        lines.append(f"Current load: {load if load else 'EMPTY'}")

        orders = obs_dict.get("active_orders", [])

        # Priority 1: Can we DELIVER?
        deliverable = [
            o for o in orders
            if o["status"] == "assigned"
            and o["id"] in load
            and o["dropoff_node"] == location
        ]
        if deliverable:
            oid = deliverable[0]["id"]
            lines.append(f"\n✅ ACTION: DELIVER order '{oid}' — you are at its dropoff node!")
            lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "deliver", "order_id": "{oid}"}}')

        else:
            # Priority 2: Can we PICKUP?
            pickupable = [
                o for o in orders
                if o["status"] == "queued"
                and o["pickup_node"] == location
            ]
            if pickupable:
                oid = pickupable[0]["id"]
                lines.append(f"\n✅ ACTION: PICKUP order '{oid}' — you are at its pickup node!")
                lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "pickup", "order_id": "{oid}"}}')

            else:
                # Priority 3: Move toward goal
                if load:
                    # Carrying an order → move toward its dropoff
                    carried_order = next(
                        (o for o in orders if o["id"] == load[0] and o["status"] == "assigned"),
                        None,
                    )
                    if carried_order:
                        goal = carried_order["dropoff_node"]
                        path = bfs_shortest_path(location, goal)
                        next_hop = bfs_next_hop(location, goal)

                        # Anti-backtrack: if next_hop is the previous node, try alternate
                        if next_hop == previous_node and len(path) > 2:
                            # Try to find an alternate path avoiding previous_node
                            alt_neighbors = [n for n in neighbors if n != previous_node]
                            # Pick the neighbor closest to goal
                            best_alt = None
                            best_len = float("inf")
                            for n in alt_neighbors:
                                alt_path = bfs_shortest_path(n, goal)
                                if len(alt_path) < best_len:
                                    best_len = len(alt_path)
                                    best_alt = n
                            if best_alt is not None:
                                next_hop = best_alt

                        path_str = " → ".join(str(n) for n in path)
                        lines.append(f"\nGoal: Deliver order '{load[0]}' to Node {goal}")
                        lines.append(f"Shortest path: {path_str}")
                        lines.append(f"✅ ACTION: ASSIGN to Node {next_hop}")
                        lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "assign", "target_node": {next_hop}}}')
                    else:
                        lines.append("\n⚠️ Carrying order but cannot find it in active orders. WAIT.")
                        lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "wait"}}')

                else:
                    # Not carrying → move toward nearest queued order's pickup
                    queued = [o for o in orders if o["status"] == "queued"]
                    if queued:
                        # Find nearest queued order
                        nearest = min(
                            queued,
                            key=lambda o: len(bfs_shortest_path(location, o["pickup_node"])),
                        )
                        goal = nearest["pickup_node"]

                        if goal == location:
                            # We're at pickup but didn't match above — pickup it
                            lines.append(f"\n✅ ACTION: PICKUP order '{nearest['id']}'")
                            lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "pickup", "order_id": "{nearest["id"]}"}}')
                        else:
                            path = bfs_shortest_path(location, goal)
                            next_hop = bfs_next_hop(location, goal)

                            # Anti-backtrack
                            if next_hop == previous_node:
                                alt_neighbors = [n for n in neighbors if n != previous_node]
                                best_alt = None
                                best_len = float("inf")
                                for n in alt_neighbors:
                                    alt_path = bfs_shortest_path(n, goal)
                                    if len(alt_path) < best_len:
                                        best_len = len(alt_path)
                                        best_alt = n
                                if best_alt is not None:
                                    next_hop = best_alt

                            path_str = " → ".join(str(n) for n in path)
                            lines.append(f"\nGoal: Go to pickup node {goal} for order '{nearest['id']}'")
                            lines.append(f"Shortest path: {path_str}")
                            lines.append(f"✅ ACTION: ASSIGN to Node {next_hop}")
                            lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "assign", "target_node": {next_hop}}}')
                    else:
                        # No queued orders, nothing to do
                        lines.append("\nNo pending orders. WAIT.")
                        lines.append(f'REQUIRED OUTPUT: {{"vehicle_id": "{vid}", "action_type": "wait"}}')

    return "\n".join(lines)


# ── LLM Action Parser ───────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict]:
    """Extract a JSON object from LLM response text, handling code fences."""
    # Try to find JSON inside markdown code fences first
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Try to find a raw JSON object
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    # Last resort: try parsing the whole text
    return json.loads(text.strip())


def get_llm_action(
    obs_dict: Dict[str, Any],
    previous_node: Optional[int] = None,
) -> tuple[LastMileAction, Optional[str]]:
    """
    Query the LLM for the next action.

    Returns:
        (action, error_msg)  – error_msg is None on success, else a description.
        On failure the action is a WAIT fallback.
    """
    # Determine a default vehicle_id for fallback
    vehicles = obs_dict.get("vehicles", [])
    fallback_vid = vehicles[0]["id"] if vehicles else "v1"

    try:
        user_message = serialize_observation(obs_dict, previous_node=previous_node)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,  # Deterministic — no creativity needed
            max_tokens=128,   # Short response — just a JSON object
        )

        raw = response.choices[0].message.content.strip()
        parsed = _extract_json(raw)
        action = LastMileAction(**parsed)
        return action, None

    except json.JSONDecodeError as e:
        error = f"JSON parse error: {e}"
    except Exception as e:
        error = f"LLM error: {e}"

    # Fallback: WAIT
    fallback = LastMileAction(
        vehicle_id=fallback_vid,
        action_type="wait",
        target_node=None,
        order_id=None,
    )
    return fallback, error


# ── Stdout Formatters ────────────────────────────────────────────────────────

def emit_start(task: str, model: str) -> None:
    print(
        f"[START] task={task} env={BENCHMARK} model={model}",
        flush=True,
    )


def emit_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    err_str = "null" if error is None else error
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={err_str}",
        flush=True,
    )


def emit_end(success: bool, steps: int, rewards: List[float]) -> None:
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={reward_str}",
        flush=True,
    )


# ── Main Loop ────────────────────────────────────────────────────────────────

async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False

    # Movement history tracking for anti-backtracking
    previous_node: Optional[int] = None
    last_known_location: Optional[int] = None

    async with LastMileEnv(base_url="http://localhost:8000") as env:
        emit_start(task=TASK_NAME, model=MODEL_NAME)

        try:
            # Reset with the correct task scenario
            result = await env.reset(task_id=TASK_NAME)

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_dict = json.loads(result.observation.model_dump_json())

                # Track vehicle location for anti-backtracking
                vehicles = obs_dict.get("vehicles", [])
                if vehicles:
                    current_location = vehicles[0]["location_node"]
                    current_status = vehicles[0]["status"]

                    # Update previous_node only when the vehicle has ARRIVED
                    # at a new location (status is idle and location changed)
                    if current_status == "idle" and last_known_location is not None and current_location != last_known_location:
                        previous_node = last_known_location

                    if current_status == "idle":
                        last_known_location = current_location

                # Ask the LLM for the next action
                action, error = get_llm_action(obs_dict, previous_node=previous_node)

                # Execute the action in the environment
                result = await env.step(action)

                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step

                # Build a concise action string
                act_type = action.action_type
                action_str = act_type.value if hasattr(act_type, "value") else str(act_type)
                if action.target_node is not None:
                    action_str += f"({action.target_node})"
                if action.order_id is not None:
                    action_str += f"[{action.order_id}]"

                emit_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=result.done,
                    error=error,
                )

                # Check for episode completion
                if result.done:
                    meta = getattr(result.observation, "metadata", {}) or {}
                    if meta.get("final_score"):
                        success = meta["final_score"] > 0.4
                    else:
                        all_delivered = all(
                            o.status == "delivered"
                            for o in result.observation.active_orders
                        )
                        success = all_delivered and bool(
                            result.observation.active_orders
                        )
                    break

        except Exception as e:
            # Emit a final error step
            emit_step(
                step=steps_taken + 1,
                action="wait",
                reward=0.0,
                done=True,
                error=str(e),
            )
        finally:
            emit_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass