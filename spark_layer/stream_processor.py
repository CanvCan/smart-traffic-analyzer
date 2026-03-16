"""
spark_layer/stream_processor.py

Reads traffic events from Kafka and performs all planned windowed analyses.

Analysis queries (as defined in integration_plan.pdf):
  Q1 — Lane-based vehicle count       Tumbling 1 min
  Q2 — Average speed per lane + class Sliding  2 min / 30 sec
  Q3 — Heavy vehicle ratio            Tumbling 5 min
  Q4 — Anomaly density                Tumbling 1 min
  Q5 — Traffic status transitions     Sliding  3 min / 1 min
  Q6 — Lane-based Traffic Status      Sliding  3 min / 1 min

Usage:
    python spark_layer/stream_processor.py
"""

import os
import sys

# ── Path setup — allows running from project root ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import VEHICLE_SCHEMA, SNAPSHOT_SCHEMA
from influx_sink import write_q1, write_q2, write_q3, write_q4, write_q5, write_q6

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window,
    count, avg, sum as _sum, max as _max, min as _min,
    round as _round, to_timestamp, when, approx_count_distinct
)

# ── Constants ────────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = 'localhost:9092'
TOPIC_VEHICLES = 'traffic.vehicles'
TOPIC_SNAPSHOTS = 'traffic.snapshots'
CHECKPOINT_BASE = '/tmp/traffic-checkpoint'
WATERMARK_DELAY = '12 seconds'
TRIGGER_INTERVAL = '15 seconds'

# ── Spark Session ────────────────────────────────────────────────────────────
os.environ.setdefault('PYSPARK_PYTHON', 'python')

spark = SparkSession.builder \
    .appName('SmartTrafficAnalyzer') \
    .config('spark.jars.packages',
            'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
    .config('spark.sql.shuffle.partitions', '4') \
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
    .withColumn('event_time', to_timestamp(col('timestamp').cast('long'))) \
    .filter(col('vehicle.lane').isNotNull())
# ^ Defence-in-depth: event_builder already gates on lane=None at the source,
#   but any replayed / legacy messages or schema mismatches are caught here.
#   Applied once so all 6 queries below inherit the filter automatically.

snapshots = raw_snapshots.select(
    from_json(col('raw'), SNAPSHOT_SCHEMA).alias('d')
).select('d.*') \
    .withColumn('event_time', to_timestamp(col('timestamp').cast('long')))

print("[Spark] Schemas loaded and streams parsed.")

# ── Q1 — Lane-based vehicle count (tumbling 1 min) ───────────────────────────
q1_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '1 minute'),
    col('vehicle.lane').alias('lane'),
    col('vehicle.class').alias('vehicle_class'),
).agg(
    count('*').alias('vehicle_count'),
    _round(avg('kinematics.speed_px_per_sec'), 2).alias('avg_speed_px'),
    _round(_max('kinematics.speed_px_per_sec'), 2).alias('max_speed_px'),
    _round(_min('kinematics.speed_px_per_sec'), 2).alias('min_speed_px'),
)

q1_console = q1_df.writeStream \
    .outputMode('update') \
    .format('console') \
    .option('truncate', False) \
    .option('numRows', 20) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q1_console') \
    .queryName('Q1_console') \
    .start()

q1_influx = q1_df.writeStream \
    .outputMode('update') \
    .foreachBatch(write_q1) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q1_influx') \
    .queryName('Q1_influx') \
    .start()

print("[Spark] Q1 started — lane-based vehicle count (1 min tumbling)")

# ── Q2 — Average speed per lane + class (sliding 2 min / 30 sec) ─────────────
q2_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '2 minutes', '30 seconds'),
    col('vehicle.lane').alias('lane'),
    col('vehicle.class').alias('vehicle_class'),
).agg(
    _round(avg('kinematics.speed_px_per_sec'), 2).alias('avg_speed_px'),
    count('*').alias('sample_count'),
    _round(
        avg(when(col('kinematics.is_stopped') == True, 1).otherwise(0)) * 100, 1
    ).alias('stopped_pct'),
    _round(
        avg(when(col('kinematics.is_slow') == True, 1).otherwise(0)) * 100, 1
    ).alias('slow_pct'),
)

q2_console = q2_df.writeStream \
    .outputMode('update') \
    .format('console') \
    .option('truncate', False) \
    .option('numRows', 20) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q2_console') \
    .queryName('Q2_console') \
    .start()

q2_influx = q2_df.writeStream \
    .outputMode('update') \
    .foreachBatch(write_q2) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q2_influx') \
    .queryName('Q2_influx') \
    .start()

print("[Spark] Q2 started — avg speed per lane + class (2 min / 30 sec sliding)")

# ── Q3 — Heavy vehicle ratio (tumbling 5 min) ────────────────────────────────
q3_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '5 minutes'),
    col('vehicle.lane').alias('lane'),
).agg(
    count('*').alias('total_vehicles'),
    _sum(when(col('vehicle.is_heavy') == True, 1).otherwise(0)).alias('heavy_count'),
    _sum(when(col('vehicle.class') == 'Bus', 1).otherwise(0)).alias('bus_count'),
    _sum(when(col('vehicle.class') == 'Truck', 1).otherwise(0)).alias('truck_count'),
    _round(
        _sum(when(col('vehicle.is_heavy') == True, 1).otherwise(0)) /
        count('*') * 100, 2
    ).alias('heavy_ratio_pct'),
)

q3_console = q3_df.writeStream \
    .outputMode('update') \
    .format('console') \
    .option('truncate', False) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q3_console') \
    .queryName('Q3_console') \
    .start()

q3_influx = q3_df.writeStream \
    .outputMode('update') \
    .foreachBatch(write_q3) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q3_influx') \
    .queryName('Q3_influx') \
    .start()

print("[Spark] Q3 started — heavy vehicle ratio (5 min tumbling)")

# ── Q4 — Anomaly density per lane (tumbling 1 min) ───────────────────────────
q4_df = vehicles \
    .filter(col('anomaly.is_anomaly') == True) \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '1 minute'),
    col('vehicle.lane').alias('lane'),
    col('anomaly.type').alias('anomaly_type'),
).agg(
    count('*').alias('anomaly_count'),
    _round(avg('anomaly.stop_seconds'), 1).alias('avg_stop_seconds'),
    _round(_max('anomaly.stop_seconds'), 1).alias('max_stop_seconds'),
)

q4_console = q4_df.writeStream \
    .outputMode('update') \
    .format('console') \
    .option('truncate', False) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q4_console') \
    .queryName('Q4_console') \
    .start()

q4_influx = q4_df.writeStream \
    .outputMode('update') \
    .foreachBatch(write_q4) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q4_influx') \
    .queryName('Q4_influx') \
    .start()

print("[Spark] Q4 started — anomaly density per lane (1 min tumbling)")

# ── Q5 — Traffic status transitions (sliding 3 min / 1 min) ──────────────────
q5_df = snapshots \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '3 minutes', '1 minute'),
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

q5_console = q5_df.writeStream \
    .outputMode('update') \
    .format('console') \
    .option('truncate', False) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q5_console') \
    .queryName('Q5_console') \
    .start()

q5_influx = q5_df.writeStream \
    .outputMode('update') \
    .foreachBatch(write_q5) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q5_influx') \
    .queryName('Q5_influx') \
    .start()

print("[Spark] Q5 started — traffic status transitions (3 min / 1 min sliding)")

# ── Q6 — Lane-based Traffic Status (sliding 3 min / 1 min) ───────────────────
q6_df = vehicles \
    .withWatermark('event_time', WATERMARK_DELAY) \
    .groupBy(
    window('event_time', '3 minutes', '1 minute'),
    col('vehicle.lane').alias('lane'),
).agg(
    _round(avg('kinematics.speed_px_per_sec'), 2).alias('avg_speed_px'),
    approx_count_distinct('vehicle.id').alias('unique_vehicles'),
).withColumn(
    'traffic_status',
    # Thresholds mirror metrics.py: SPEED_FREE=65, SPEED_FLOW=40, SPEED_JAMMED=15
    when(col('avg_speed_px') >= 65, 'FREE')
    .when(col('avg_speed_px') >= 40, 'FLOW')
    .when(col('avg_speed_px') >= 15, 'HEAVY')
    .otherwise('JAMMED')
)

q6_console = q6_df.writeStream \
    .outputMode('update') \
    .format('console') \
    .option('truncate', False) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q6_console') \
    .queryName('Q6_console') \
    .start()

q6_influx = q6_df.writeStream \
    .outputMode('update') \
    .foreachBatch(write_q6) \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .option('checkpointLocation', f'{CHECKPOINT_BASE}/q6_influx') \
    .queryName('Q6_influx') \
    .start()

print("[Spark] Q6 started — lane-based traffic status (3 min / 1 min sliding)")

print()
print("=" * 60)
print("  All 6 queries running. Press Ctrl+C to stop.")
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
