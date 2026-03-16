"""
spark/schemas.py

Single source of truth for all Spark schemas.
If EventBuilder's JSON schema changes, update here too.

Rule: adding new fields is backward compatible.
      removing or renaming fields will break streaming queries.
"""

from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, BooleanType, ArrayType
)

# ── vehicle_detected schema ──────────────────────────────────────────────────
VEHICLE_SCHEMA = StructType([
    StructField('event_type', StringType()),
    StructField('timestamp', DoubleType()),
    StructField('camera_id', StringType()),
    StructField('frame_id', IntegerType()),

    StructField('vehicle', StructType([
        StructField('id', IntegerType()),
        StructField('class', StringType()),
        StructField('class_id', IntegerType()),
        StructField('is_heavy', BooleanType()),
        StructField('bbox', ArrayType(IntegerType())),
        StructField('lane', StringType()),
    ])),

    StructField('kinematics', StructType([
        StructField('speed_px_per_sec', DoubleType()),
        StructField('is_slow', BooleanType()),
        StructField('is_stopped', BooleanType()),
        StructField('direction', StringType()),
    ])),

    StructField('residence', StructType([
        StructField('entry_frame', IntegerType()),
        StructField('frames_in_roi', IntegerType()),
        StructField('seconds_in_roi', DoubleType()),
    ])),

    StructField('anomaly', StructType([
        StructField('is_anomaly', BooleanType()),
        StructField('type', StringType()),
        StructField('stop_seconds', DoubleType()),
    ])),
])

# ── traffic_snapshot schema ──────────────────────────────────────────────────
SNAPSHOT_SCHEMA = StructType([
    StructField('event_type', StringType()),
    StructField('timestamp', DoubleType()),
    StructField('camera_id', StringType()),
    StructField('frame_id', IntegerType()),

    StructField('counts', StructType([
        StructField('total', IntegerType()),
        StructField('car', IntegerType()),
        StructField('motorcycle', IntegerType()),
        StructField('bus', IntegerType()),
        StructField('truck', IntegerType()),
        StructField('heavy_vehicle_ratio', DoubleType()),
    ])),

    StructField('speed', StructType([
        StructField('average_px', DoubleType()),
        StructField('min_px', DoubleType()),
        StructField('max_px', DoubleType()),
    ])),

    StructField('density', StructType([
        StructField('status', StringType()),
        StructField('occupancy_ratio', DoubleType()),
    ])),

    # IMPORTANT: lane_counts field names must match lane names defined in app config.
    # Spark schemas are compile-time static — if you rename or add lanes in the
    # config, update these StructFields to match, otherwise the snapshot stream
    # will silently produce nulls for mismatched lane names.
    StructField('lane_counts', StructType([
        StructField('Right_Lane', IntegerType()),
        StructField('Middle_Lane', IntegerType()),
        StructField('Left_Lane', IntegerType()),
    ])),

    StructField('anomalies', ArrayType(StructType([
        StructField('vehicle_id', IntegerType()),
        StructField('class', StringType()),
        StructField('type', StringType()),
        StructField('stop_seconds', DoubleType()),
        StructField('speed_px', DoubleType()),
        StructField('lane', StringType()),
    ]))),
])
