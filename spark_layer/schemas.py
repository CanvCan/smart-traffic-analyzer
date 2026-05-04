"""
spark/schemas.py

Single source of truth for all Spark schemas.
If EventBuilder's JSON schema changes, update here too.

Rule: adding new fields is backward compatible.
      removing or renaming fields will break streaming queries.
"""

from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, BooleanType, ArrayType, MapType
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
        StructField('dolmus', IntegerType()),
        StructField('taxi', IntegerType()),
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

    # MapType allows any set of lane names (Right_Lane, Left_Lane, custom names, etc.)
    # so cameras with different ROI configurations never cause a schema mismatch.
    StructField('lane_counts', MapType(StringType(), IntegerType())),

    StructField('anomalies', ArrayType(StructType([
        StructField('vehicle_id', IntegerType()),
        StructField('class', StringType()),
        StructField('type', StringType()),
        StructField('stop_seconds', DoubleType()),
        StructField('speed_px', DoubleType()),
        StructField('lane', StringType()),
    ]))),
])
