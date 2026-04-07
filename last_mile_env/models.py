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
    current_load: List[str] = [] # Fixed: was current_load: int
    capacity: int
    status: VehicleStatus
    destination_node: Optional[int] = None
    time_to_arrival: float = 0.0 # New: Visible "countdown" for agent planning

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
    done: bool
    reward: float
    metadata: Dict[str, Any] = {}

class ActionType(str, Enum):
    ASSIGN = "assign"
    REROUTE = "reroute"
    WAIT = "wait"

class LastMileAction(BaseModel):
    vehicle_id: str
    action_type: ActionType
    target_node: Optional[int] = None
    order_id: Optional[str] = None