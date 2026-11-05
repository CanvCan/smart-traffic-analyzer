from traffic_analyzer.domain.models import VehicleClass

CLASS_COLORS = {
    VehicleClass.BUS.class_id:        (220, 200,   0),  # Bus        -> Cyan
    VehicleClass.CAR.class_id:        (  0, 220,   0),  # Car        -> Green
    VehicleClass.DOLMUS.class_id:     (  0, 140, 255),  # Dolmus     -> Orange
    VehicleClass.MOTORCYCLE.class_id: (255,  80,  80),  # Motorcycle -> Blue
    VehicleClass.TAXI.class_id:       (  0, 255, 255),  # Taxi       -> Yellow
    VehicleClass.TRUCK.class_id:      (  0,   0, 220),  # Truck      -> Red
}

CLASS_LABELS = {vc.class_id: vc.label for vc in VehicleClass if vc != VehicleClass.UNKNOWN}

LANE_PALETTE = [
    (255, 220, 180),  # soft sky blue
    (200, 255, 180),  # soft mint green
    (160, 200, 255),  # soft peach
    (255, 180, 220),  # soft lavender
    (160, 255, 255),  # soft yellow
    (230, 230, 160),  # soft teal
]
