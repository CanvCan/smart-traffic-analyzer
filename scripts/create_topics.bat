@echo off
echo [Kafka] Creating topics...

docker exec kafka_layer kafka-topics --create ^
  --bootstrap-server localhost:9092 ^
  --topic traffic.vehicles ^
  --partitions 3 ^
  --replication-factor 1

docker exec kafka_layer kafka-topics --create ^
  --bootstrap-server localhost:9092 ^
  --topic traffic.snapshots ^
  --partitions 1 ^
  --replication-factor 1

echo.
echo [Kafka] Existing topics:
docker exec kafka_layer kafka-topics --list --bootstrap-server localhost:9092

echo.
echo [Kafka] Done!
pause