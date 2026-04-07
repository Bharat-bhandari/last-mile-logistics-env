---
title: Last-Mile Logistics Controller
emoji: "🚚"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - logistics
  - reinforcement-learning
---

# Last-Mile Logistics Controller

OpenEnv environment for sequential dispatch decisions under uncertainty.

This environment simulates Santacruz, Mumbai micro-routing across a 4-node directed graph and supports three deterministic benchmark tasks:

- **easy**: static traffic, one vehicle, one order, generous deadline
- **medium**: dynamic traffic, two vehicles, two orders
- **hard**: non-stationary traffic, one vehicle, three orders, tight deadlines

## Graph

```
Station(0) ↔ SV_Road(1) ↔ Linking_Road(2) ↔ Juhu_Tara(3)
         5min        4min              7min
```

## Action Contract

`LastMileAction`:

```json
{
    "vehicle_id": "v1",
    "action_type": "assign" | "reroute" | "wait",
    "target_node": 2,
    "order_id": "e1"
}
```

- **assign**: Move idle vehicle to an adjacent node
- **reroute**: Redirect a moving vehicle to a different adjacent node
- **wait**: Keep vehicle idle at current location

## Observation Contract

`LastMileObservation` includes:

- `timestep`: current step count
- `vehicles`: list with location, load, status, destination, time_to_arrival
- `active_orders`: list with pickup/dropoff nodes, deadline, priority, status
- `traffic_map`: edge multiplier dict (e.g. `"0_1": 3.5`)
- `done`: boolean
- `reward`: float
- `metadata`: dict (includes `final_score` and `grader` at episode end)

## Reward Design

```text
per_step    = -0.1   (operational cost)
on_time     = +20.0  (delivery before deadline)
late_deliver= +5.0   (delivery after deadline)
late_expire = -10.0  (order exceeds deadline undelivered)
reroute     = -2.0   (rerouting penalty)
```

## Order Lifecycle

`queued` → `assigned` (vehicle at pickup) → `delivered` (vehicle at dropoff)

Orders that exceed their deadline while `queued` or `assigned` are marked `late`.

## Task Selection

Set `LMLC_TASK` env var or pass `task_id` in the reset options dict:

```bash
export LMLC_TASK=hard
```

Supported: `easy`, `medium`, `hard`.

## Local Run

```bash
cd last_mile_env
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Expected routes: `/reset`, `/step`, `/state`, `/health`, `/docs`.

## Docker Build

```bash
docker build -t last-mile-logistics:latest .
docker run --rm -p 8000:8000 last-mile-logistics:latest
```

## OpenEnv Validate

```bash
cd last_mile_env
openenv validate
```

## Inference

Root-level `inference.py` runs the agent loop with strict logging:

- `[START]`
- `[STEP]`
- `[END]`

Set `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` env vars before running.
