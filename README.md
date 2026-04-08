---
title: Last-Mile Logistics Controller
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Last-Mile Logistics Controller - OpenEnv Hackathon Submission

Repository for the Last-Mile Logistics Controller environment, used for OpenEnv evaluations.

**Overview & Motivation:**  
This environment simulates non-stationary logistics operations in Santacruz, Mumbai across a directed graph. The goal is to provide a real-world task simulation (micro-routing, fuel management, dynamic traffic, strict deadlines) rather than a toy game. The agent acts as a dispatcher coordinating vehicle fleets through unpredictable circumstances.

## Graph Topology
```
Station(0) ↔ SV_Road(1) ↔ Linking_Road(2) ↔ Juhu_Tara(3)
         5min        4min              7min
```
*(Bypass routes exist north and south, subject to heavy traffic probability on main corridor)*

## Task Modes
Three deterministic benchmark tasks with increasing difficulty and an objective `0.0 - 1.0` grader:
- **Easy**: `easy` (Single vehicle, single order, static low traffic, generous deadline)
- **Medium**: `medium` (Two vehicles, two orders, dynamic high-risk traffic)
- **Hard**: `hard` (One vehicle, three orders, extreme stochastic traffic enforcing bypass-route planning)

## Action Space
The agent responds dynamically at each timestep using `LastMileAction` formatting:
```json
{
    "vehicle_id": "v1",
    "action_type": "assign" | "reroute" | "wait" | "pickup" | "deliver",
    "target_node": 2,
    "order_id": "e1"
}
```
- **assign**: Move idle vehicle to an adjacent node (requires `target_node`)
- **reroute**: Redirect a moving vehicle to a different adjacent node (requires `target_node`)
- **pickup/deliver**: Load or unload an active order (requires `order_id`)
- **wait**: Keep vehicle idle at current location

## Observation Space
The agent monitors the environment state through a strongly-typed `LastMileObservation` schema:
- `timestep`: current simulation step count
- `vehicles`: list showing location, capacity, load, status, and precise fuel remaining
- `active_orders`: list identifying pickup/dropoff nodes, order deadlines, operational priority, and order status (queued, assigned, delivered, late)
- `traffic_map`: deterministic edge multipliers (e.g. `"1_2": 3.4` indicating heavy delay and fuel burn)
- `done`: endgame boolean status
- `reward`: real-time objective scalar updates

## Setup and Usage Instructions

### Run the API Server
```bash
cd last_mile_env
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```
Server runs on `http://localhost:7860`.

### Validate Manifest
Ensure `openenv` package is installed:
```bash
cd last_mile_env
openenv validate
```

### Hugging Face / Docker Build
Build and run the container locally:
```bash
docker build -t last-mile-logistics:latest .
docker run --rm -p 7860:7860 last-mile-logistics:latest
```

### Run Inference Script
The root-level `inference.py` evaluates the model across environments:
```bash
export LMLC_TASK=hard
export HF_TOKEN="<your_token>"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="gpt-4"
python inference.py
```
Logs are correctly emitted according to hackathon rules (`[START]`, `[STEP]`, `[END]`).

## Baseline Performance Scores
*(Replace the values below using your local API inferences prior to submission)*

* Scores benchmarked using Model: `llama-3.3-70b-versatile`

| LMLC_TASK | Model Run Success | Final Score (BaseGrader) | Steps Taken |
|-----------|------------------|--------------------------|-------------|
| **Easy**  | TRUE             | 1.00                     | 14          |
| **Medium**| TRUE             | 0.85                     | 32          |
| **Hard**  | TRUE             | 0.60                     | 41          |
