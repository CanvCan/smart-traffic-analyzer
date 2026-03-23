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
    VehicleClass.CAR.value:        (80, 220, 80),    # Car        -> Green
    VehicleClass.MOTORCYCLE.value: (220, 80, 220),   # Motorcycle -> Purple
    VehicleClass.BUS.value:        (50, 180, 220),   # Bus        -> Cyan
    VehicleClass.TRUCK.value:      (60, 60, 220),    # Truck      -> Red
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
