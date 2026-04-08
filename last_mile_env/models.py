from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum

class VehicleStatus(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    LOADING = "loading"
    BROKEN = "broken"

class Vehicle(BaseModel):
    id: str
    location_node: int
    current_load: List[str] = []
    capacity: int
    status: VehicleStatus
    destination_node: Optional[int] = None
    time_to_arrival: float = 0.0
    fuel: float = 100.0  # Fuel resource — burns per step, 0 = BROKEN

class Order(BaseModel):
    id: str
    pickup_node: int
    dropoff_node: int
    priority: int 
    deadline: int 
    status: str # "queued", "assigned", "delivered", "late"

class LastMileObservation(BaseModel):
    timestep: int
    vehicles: List[Vehicle]
    active_orders: List[Order]
    traffic_map: Dict[str, float]
    graph_adjacency: Dict[int, List[Dict[str, Any]]] = {}  # Agent sees the full graph
    done: bool
    reward: float
    metadata: Dict[str, Any] = {}

class ActionType(str, Enum):
    ASSIGN = "assign"
    REROUTE = "reroute"
    WAIT = "wait"
    PICKUP = "pickup"
    DELIVER = "deliver"

class LastMileAction(BaseModel):
    vehicle_id: str
    action_type: ActionType
    target_node: Optional[int] = None
    order_id: Optional[str] = None