# BGR color palette per vehicle class
CLASS_COLORS = {
    2: (80, 220, 80),  # Car        → Green
    3: (220, 80, 220),  # Motorcycle → Purple
    5: (50, 180, 220),  # Bus        → Orange-yellow
    7: (60, 60, 220),  # Truck      → Red
}

CLASS_LABELS = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Soft muted BGR palette for lane overlays — one color per lane index
LANE_PALETTE = [
    (255, 220, 180),  # soft sky blue
    (200, 255, 180),  # soft mint green
    (160, 200, 255),  # soft peach
    (255, 180, 220),  # soft lavender
    (160, 255, 255),  # soft yellow
    (230, 230, 160),  # soft teal
]
