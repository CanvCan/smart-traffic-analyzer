# visualization/colors.py
#
# Color and label constants for rendering.
#
# CLASS_LABELS is no longer defined here; it is derived from the domain
# VehicleClass enum so there is a single source of truth for vehicle names.
# Renderer reads CLASS_LABELS from this module without depending on domain
# directly in its hot-path code.

from traffic_analyzer.domain.models import VehicleClass

# BGR colour palette per vehicle class (OpenCV uses BGR, not RGB)
CLASS_COLORS = {
    VehicleClass.BUS.value:        (220, 200,   0),  # Bus        -> Cyan
    VehicleClass.CAR.value:        (  0, 220,   0),  # Car        -> Green
    VehicleClass.DOLMUS.value:     (  0, 140, 255),  # Dolmus     -> Orange
    VehicleClass.MOTORCYCLE.value: (255,  80,  80),  # Motorcycle -> Blue
    VehicleClass.TAXI.value:       (  0, 255, 255),  # Taxi       -> Yellow
    VehicleClass.TRUCK.value:      (  0,   0, 220),  # Truck      -> Red
}

# Derived from domain — single source of truth for label strings.
CLASS_LABELS = {vc.value: vc.label for vc in VehicleClass if vc != VehicleClass.UNKNOWN}

# Soft muted BGR palette for lane polygon overlays — one colour per lane index
LANE_PALETTE = [
    (255, 220, 180),  # soft sky blue
    (200, 255, 180),  # soft mint green
    (160, 200, 255),  # soft peach
    (255, 180, 220),  # soft lavender
    (160, 255, 255),  # soft yellow
    (230, 230, 160),  # soft teal
]
