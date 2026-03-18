"""
spark_layer/stream_processor.py

Reads traffic events from Kafka and performs all windowed analyses.

Queries:
  Q1  — Lane-based vehicle count       Tumbling 1 min
  Q2  — Speed tracking per lane+class  Sliding  2 min / 30 sec
  Q3  — Heavy vehicle ratio            Tumbling 5 min
  Q4  — Anomaly density                Tumbling 1 min
  Q5  — Traffic status transitions     Sliding  3 min / 1 min
  Q6  — Lane speed metrics             Sliding  3 min / 1 min
  Q7  — Flow rate per lane             Tumbling 1 min
  Q8  — Dwell time per lane            Tumbling 2 min
  Q9  — Lane occupancy                 Tumbling 1 min
  Q10 — Direction analysis             Tumbling 1 min
  Q11 — Vehicle class breakdown        Tumbling 1 min

All measurements include camera_id as a tag for multi-camera support.

Usage:
    python spark_layer/stream_processor.py
"""

import os
import sys
import tempfile

# ── Path setup — allows running from project root ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import VEHICLE_SCHEMA, SNAPSHOT_SCHEMA
from influx_sink import (
    write_q1, write_q2, write_q3, write_q4, write_q5, write_q6,
    write_q7, write_q8, write_q9, write_q10, write_q11,
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window,
    count, avg, max as _max, min as _min, stddev as _stddev,
    round as _round, when, approx_count_distinct,
)

# ── Constants ────────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = 'localhost:9092'
TOPIC_VEHICLES = 'traffic.vehicles'
TOPIC_SNAPSHOTS = 'traffic.snapshots'
CHECKPOINT_BASE = os.path.join(tempfile.gettempdir(), 'traffic-checkpoint')
WATERMARK_DELAY = '12 seconds'
TRIGGER_INTERVAL = '15 seconds'

# ── Spark Session ────────────────────────────────────────────────────────────
os.environ.setdefault('PYSPARK_PYTHON', 'python')

spark = SparkSession.builder \
    .appName('SmartTrafficAnalyzer') \
    .config('spark.jars.packages',
            'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.sql.session.timeZone', 'UTC') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print("[Spark] Session started.")
print("[Spark] Connecting to Kafka at", BOOTSTRAP_SERVERS)


# ── Read from Kafka ──────────────────────────────────────────────────────────
def read_topic(topic: str):
    return spark.readStream \
        .format('kafka') \
        .option('kafka.bootstrap.servers', BOOTSTRAP_SERVERS) \
        .option('subscribe', topic) \
        .option('startingOffsets', 'latest') \
        .option('failOnDataLoss', 'false') \
        .load() \
        .select(col('value').cast('string').alias('raw'))


raw_vehicles = read_topic(TOPIC_VEHICLES)
raw_snapshots = read_topic(TOPIC_SNAPSHOTS)

# ── Parse JSON ───────────────────────────────────────────────────────────────
vehicles = raw_vehicles.select(
    from_json(col('raw'), VEHICLE_SCHEMA).alias('d')
).select('d.*') \
    .withColumn('event_time', col('timestamp').cast('timestamp')) \
    .filter(col('vehicle.lane').isNotNull())

snapshots = raw_snapshots.select(
    from_json(col('raw'), SNAPSHOT_SCHEMA).alias('d')
).select('d.*') \
    .withColumn('event_time', col('timestamp').cast('timestamp'))

print("[Spark] Schemas loaded and streams parsed.")


def _start(df, write_fn, name, mode='update'):
    """Start console + InfluxDB sinks for a query dataframe."""
    df.writeStream \
        .outputMode(mode) \
        .format('console') \
        .option('truncate', False) \
        .option('numRows', 20) \
        .trigger(processingTime=TRIGGER_INTERVAL) \
        .option('checkpointLocation', f'{CHECKPOINT_BASE}/{name}_console') \
        .queryName(f'{name}_console') \
        .start()

    df.writeStream \
        .outputMode(mode) \
        .foreachBatch(write_fn) \
        .trigger(processingTime=TRIGGER_INTERVAL) \
        .option('checkpointLocation', f'{CHECKPOINT_BASE}/{name}_influx') \
        .queryName(f'{name}_influx') \
        .start()

    print(f"[Spark] {name} started.")


# ── Q1 — Lane-based vehicle count (tumbling 1 min) ───────────────────────────
q1_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '1 minute'),
    col('camera_id'),
    col('vehicle.lane').alias('lane'),
    col('vehicle.class').alias('vehicle_class'),
).agg(
    approx_count_distinct('vehicle.id').alias('vehicle_count'),
    _round(avg('kinematics.speed_px_per_sec'), 2).alias('avg_speed_px'),
    _round(_max('kinematics.speed_px_per_sec'), 2).alias('max_speed_px'),
    _round(_min('kinematics.speed_px_per_sec'), 2).alias('min_speed_px'),
)
_start(q1_df, write_q1, 'Q1')

# ── Q2 — Speed tracking per lane + class (sliding 2 min / 30 sec) ────────────
q2_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '2 minutes', '30 seconds'),
    col('camera_id'),
    col('vehicle.lane').alias('lane'),
    col('vehicle.class').alias('vehicle_class'),
).agg(
    _round(avg('kinematics.speed_px_per_sec'), 2).alias('avg_speed_px'),
    _round(_stddev('kinematics.speed_px_per_sec'), 2).alias('speed_stddev_px'),
    _round(
        avg(when(col('kinematics.is_stopped') == True, 1).otherwise(0)) * 100, 1
    ).alias('stopped_pct'),
    _round(
        avg(when(col('kinematics.is_slow') == True, 1).otherwise(0)) * 100, 1
    ).alias('slow_pct'),
)
_start(q2_df, write_q2, 'Q2')

# ── Q3 — Heavy vehicle ratio (tumbling 5 min) ────────────────────────────────
q3_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '5 minutes'),
    col('camera_id'),
    col('vehicle.lane').alias('lane'),
).agg(
    approx_count_distinct('vehicle.id').alias('total_vehicles'),
    approx_count_distinct(
        when(col('vehicle.is_heavy') == True, col('vehicle.id'))
    ).alias('heavy_count'),
    approx_count_distinct(
        when(col('vehicle.class') == 'Bus', col('vehicle.id'))
    ).alias('bus_count'),
    approx_count_distinct(
        when(col('vehicle.class') == 'Truck', col('vehicle.id'))
    ).alias('truck_count'),
    _round(
        approx_count_distinct(
            when(col('vehicle.is_heavy') == True, col('vehicle.id'))
        ) / approx_count_distinct('vehicle.id') * 100, 2
    ).alias('heavy_ratio_pct'),
)
_start(q3_df, write_q3, 'Q3')

# ── Q4 — Anomaly density per lane (tumbling 1 min) ───────────────────────────
# withWatermark before filter — watermark advances on all vehicle events,
# not just anomalies (which can be sparse).
q4_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .filter(col('anomaly.is_anomaly') == True) \
    .groupBy(
    window('event_time', '1 minute'),
    col('camera_id'),
    col('vehicle.lane').alias('lane'),
    col('anomaly.type').alias('anomaly_type'),
).agg(
    approx_count_distinct('vehicle.id').alias('anomaly_count'),
    _round(avg('anomaly.stop_seconds'), 1).alias('avg_stop_seconds'),
    _round(_max('anomaly.stop_seconds'), 1).alias('max_stop_seconds'),
)
_start(q4_df, write_q4, 'Q4')

# ── Q5 — Traffic status transitions (sliding 3 min / 1 min) ──────────────────
q5_df = snapshots \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '3 minutes', '1 minute'),
    col('camera_id'),
    col('density.status').alias('traffic_status'),
).agg(
    count('*').alias('snapshot_count'),
    _round(avg('speed.average_px'), 2).alias('avg_speed_px'),
    _round(avg('speed.min_px'), 2).alias('avg_min_speed_px'),
    _round(avg('speed.max_px'), 2).alias('avg_max_speed_px'),
    _round(avg('counts.total'), 1).alias('avg_vehicle_count'),
    _round(avg('density.occupancy_ratio'), 4).alias('avg_occupancy'),
    _round(avg('counts.heavy_vehicle_ratio') * 100, 2).alias('avg_heavy_pct'),
)
_start(q5_df, write_q5, 'Q5')

# ── Q6 — Lane speed metrics (sliding 3 min / 1 min) ──────────────────────────
q6_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '3 minutes', '1 minute'),
    col('camera_id'),
    col('vehicle.lane').alias('lane'),
).agg(
    _round(avg('kinematics.speed_px_per_sec'), 2).alias('avg_speed_px'),
    _round(_stddev('kinematics.speed_px_per_sec'), 2).alias('speed_stddev_px'),
    approx_count_distinct('vehicle.id').alias('unique_vehicles'),
    _round(
        avg(when(col('kinematics.is_stopped') == True, 1).otherwise(0)) * 100, 1
    ).alias('stopped_pct'),
)
_start(q6_df, write_q6, 'Q6')

# ── Q7 — Flow rate per lane (tumbling 1 min) ─────────────────────────────────
q7_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '1 minute'),
    col('camera_id'),
    col('vehicle.lane').alias('lane'),
).agg(
    approx_count_distinct('vehicle.id').alias('vehicle_count'),
    _round(avg('kinematics.speed_px_per_sec'), 2).alias('avg_speed_px'),
)
_start(q7_df, write_q7, 'Q7')

# ── Q8 — Dwell time per lane (tumbling 2 min) ────────────────────────────────
# Step 1: max seconds_in_roi per vehicle per lane.
# Step 2: write_q8 aggregates to avg/max per (camera_id, lane) in foreachBatch.
q8_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '2 minutes'),
    col('camera_id'),
    col('vehicle.lane').alias('lane'),
    col('vehicle.id').alias('vehicle_id'),
).agg(
    _round(_max('residence.seconds_in_roi'), 1).alias('dwell_seconds'),
)
_start(q8_df, write_q8, 'Q8')

# ── Q9 — Lane occupancy from snapshots (tumbling 1 min) ──────────────────────
q9_df = snapshots \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '1 minute'),
    col('camera_id'),
).agg(
    _round(avg('lane_counts.Right_Lane'), 2).alias('right_lane_avg'),
    _round(avg('lane_counts.Middle_Lane'), 2).alias('middle_lane_avg'),
    _round(avg('lane_counts.Left_Lane'), 2).alias('left_lane_avg'),
    _round(_max(col('lane_counts.Right_Lane').cast('double')), 0).alias('right_lane_max'),
    _round(_max(col('lane_counts.Middle_Lane').cast('double')), 0).alias('middle_lane_max'),
    _round(_max(col('lane_counts.Left_Lane').cast('double')), 0).alias('left_lane_max'),
)
_start(q9_df, write_q9, 'Q9')

# ── Q10 — Direction analysis per lane (tumbling 1 min) ───────────────────────
q10_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '1 minute'),
    col('camera_id'),
    col('vehicle.lane').alias('lane'),
    col('kinematics.direction').alias('direction'),
).agg(
    approx_count_distinct('vehicle.id').alias('vehicle_count'),
    _round(avg('kinematics.speed_px_per_sec'), 2).alias('avg_speed_px'),
)
_start(q10_df, write_q10, 'Q10')

# ── Q11 — Vehicle class breakdown from snapshots (tumbling 1 min) ─────────────
# Tracks average count per vehicle class over time.
# Source: snapshots (pre-aggregated counts, no frame inflation risk).
q11_df = snapshots \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '1 minute'),
    col('camera_id'),
).agg(
    _round(avg('counts.car'), 2).alias('avg_car'),
    _round(avg('counts.motorcycle'), 2).alias('avg_motorcycle'),
    _round(avg('counts.bus'), 2).alias('avg_bus'),
    _round(avg('counts.truck'), 2).alias('avg_truck'),
    _round(avg('counts.total'), 2).alias('avg_total'),
)
_start(q11_df, write_q11, 'Q11')

print()
print("=" * 60)
print("  All 11 queries running. Press Ctrl+C to stop.")
print("=" * 60)
print()

# ── Keep alive ───────────────────────────────────────────────────────────────
try:
    spark.streams.awaitAnyTermination()
except KeyboardInterrupt:
    print("\n[Spark] Shutting down...")
    for q in spark.streams.active:
        q.stop()
    spark.stop()
    print("[Spark] All queries stopped.")