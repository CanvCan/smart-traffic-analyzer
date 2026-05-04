# Smart Traffic Analyzer

Gerçek zamanlı araç tespiti, çoklu nesne takibi ve trafik analizi sistemi.
Türkiye'ye özgü trafik koşulları için fine-tune edilmiş RF-DETR modeli üzerine inşa edilmiştir.

---

## Mimari Genel Bakış

```
IZUM Kameralar / Video Dosyası
           │
           ▼
    ┌─────────────┐
    │  GUI Launcher│  (CustomTkinter — launcher/app.py)
    └──────┬──────┘
           │ Composition Root — bağımlılıkları enjekte eder
           ▼
    ┌─────────────┐
    │   Analyzer   │  (application/analyzer.py)
    └──────┬──────┘
           │
    ┌──────▼──────┐      ┌─────────────┐
    │FrameProcessor│─────▶│  Detector   │  RF-DETR (adapters/detector.py)
    └──────┬──────┘      └─────────────┘
           │             ┌─────────────┐
           ├────────────▶│   Tracker   │  ByteTrack (adapters/tracker.py)
           │             └─────────────┘
           │             ┌─────────────┐
           └────────────▶│EventBuilder │  Kinematik + Anomali (application/event_builder.py)
                         └──────┬──────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
        ┌─────────┐      ┌──────────┐     ┌──────────┐
        │  Kafka  │      │ InfluxDB │     │  Ekran   │
        │Producer │      │Publisher │     │Renderer  │
        └────┬────┘      └──────────┘     └──────────┘
             │
             ▼
      ┌─────────────┐
      │Apache Spark │  11 windowed sorgu (spark_layer/)
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │  InfluxDB   │  Zaman serisi veritabanı
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │   Grafana   │  Canlı dashboard görselleştirme
      └─────────────┘
```

---

## Özellikler

- **RF-DETR ile Araç Tespiti** — 6 sınıflı Türkiye modeli: Otobüs, Araba, Dolmuş, Motosiklet, Taksi, Kamyon
- **ByteTrack Takibi** — Yüksek ID tutarlılığı, kayıp iz yönetimi, ghost track görselleştirme
- **ROI Tabanlı Şerit Analizi** — Her kamera için polygon destekli şerit konfigürasyonu
- **Kinematik Metrikler** — Piksel hızı, yön tespiti, ROI bekleme süresi
- **Anomali Tespiti** — Duran araç / ters yön / ani yavaşlama
- **11 Spark Streaming Sorgusu** — Kayan ve tümbling pencereli gerçek zamanlı analizler
- **InfluxDB v2 Zaman Serisi** — `camera_id` etiketi ile çok kameralı destek
- **Grafana Dashboard** — Canlı trafik görselleştirme
- **IZUM Entegrasyonu** — İzmir Büyükşehir MJPEG stream API
- **GUI Başlatıcı** — Kamera arama, ROI düzenleme, analiz başlatma

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| Tespit | RF-DETR (dinov2_windowed_base, 6 sınıf) |
| Takip | ByteTrack (supervision) |
| Mesajlaşma | Apache Kafka (KRaft modu, Docker) |
| Streaming Analitik | Apache Spark Structured Streaming 3.5 |
| Zaman Serisi | InfluxDB v2 (Docker) |
| Dashboard | Grafana (Docker) |
| GUI | CustomTkinter |
| Dil | Python 3.10+ |

---

## Proje Yapısı

```
GraduationProject/
├── docker-compose.yml           # Kafka + InfluxDB + Grafana
├── best.pt                      # RF-DETR model ağırlıkları (git ignored)
├── launcher/
│   └── app.py                   # Composition Root — GUI + pipeline kurulumu
├── spark_layer/
│   ├── stream_processor.py      # 11 Spark Streaming sorgusu
│   ├── influx_sink.py           # Spark → InfluxDB yazıcıları
│   └── schemas.py               # Kafka JSON şemaları
├── scripts/
│   ├── kafka_monitor.py         # Kafka topic izleme aracı
│   └── create_topics.bat        # Kafka topic oluşturma
├── traffic_analyzer/
│   ├── config.json              # Kamera + model + InfluxDB + şerit konfigürasyonu
│   ├── cameras/                 # Per-kamera ROI JSON dosyaları
│   ├── domain/
│   │   ├── models.py            # VehicleClass, Direction, TrafficStatus, AnomalyType
│   │   └── ports.py             # IEventPublisher arayüzü
│   ├── application/
│   │   ├── analyzer.py          # Pipeline orkestratörü
│   │   ├── frame_processor.py   # Tek kare işleme (Detect→Track→Metrics→Render)
│   │   └── event_builder.py     # Domain event üreticisi
│   ├── services/
│   │   ├── vehicle_metrics.py   # Araç kinematik metrikleri
│   │   ├── scene_metrics.py     # Sahne seviyesi trafik metrikleri
│   │   └── anomaly_detector.py  # Anomali tespit mantığı
│   ├── adapters/
│   │   ├── detector.py          # RFDETRDetector (BaseDetector)
│   │   ├── tracker.py           # ByteTracker (BaseTracker)
│   │   ├── influx_publisher.py  # InfluxDB yayımcısı
│   │   ├── kafka_producer.py    # Kafka yayımcısı
│   │   └── camera/
│   │       ├── camera_fetcher.py   # IZUM API kamera listesi
│   │       └── camera_selector.py  # Kamera seçim yardımcısı
│   ├── infrastructure/
│   │   ├── video_loop.py        # Video I/O + görüntü döngüsü
│   │   └── config_loader.py     # AppConfig, LaneConfig veri sınıfları
│   └── visualization/
│       ├── renderer.py          # Şerit, araç, legend overlay
│       ├── colors.py            # Sınıf bazlı renk paleti
│       ├── ghost_track_manager.py  # Geçici kayıp iz yönetimi
│       └── roi_selector.py      # İnteraktif ROI çizim aracı
└── grafana/
    └── provisioning/
        └── datasources/
            └── influxdb.yml     # Grafana InfluxDB veri kaynağı
```

---

## Kurulum

### Gereksinimler

- Python 3.10+
- CUDA destekli GPU (önerilir, CPU'da da çalışır)
- Docker Desktop
- Java 17 (Spark için zorunlu)
- Hadoop winutils (Windows için zorunlu)

### Python Bağımlılıkları

```bash
pip install rfdetr supervision customtkinter \
            influxdb-client kafka-python pyspark torch torchvision \
            opencv-python numpy python-dotenv questionary requests
```

### Docker Servisleri

```bash
docker-compose up -d
```

Bu komut şunları başlatır:
- **Kafka** — `localhost:9092` (KRaft modu)
- **InfluxDB** — `http://localhost:8086` (org: `myorg`, bucket: `traffic_metrics`)
- **Grafana** — `http://localhost:3000`

### Windows — Spark Ortam Değişkenleri

Spark'ı çalıştırmadan önce her PowerShell oturumunda aşağıdaki komutları çalıştırın:

```powershell
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH = "$env:HADOOP_HOME\bin;$env:PATH"
```

> Bu değişkenler kalıcı olarak sistem ortam değişkenlerine de eklenebilir (Sistem Özellikleri → Gelişmiş → Ortam Değişkenleri).

### Kafka Topic Oluşturma

```bash
scripts/create_topics.bat
```

Oluşturulan topic'ler:
- `traffic.vehicles` — 3 partition (per-vehicle events)
- `traffic.snapshots` — 1 partition (sahne özetleri)

---

## Konfigürasyon

`traffic_analyzer/config.json`:

```json
{
  "camera_settings": {
    "source_id": "local",
    "video_path": "video.mp4",
    "display_width": 960
  },
  "model_settings": {
    "model_path": "best.pt",
    "threshold": 0.30,
    "resolution": 880
  },
  "influx_settings": {
    "url": "http://localhost:8086",
    "token": "",
    "org": "myorg",
    "bucket": "traffic_metrics"
  },
  "lanes": {}
}
```

Şerit/ROI konfigürasyonları `traffic_analyzer/cameras/<camera_id>.json` dosyalarında ayrı tutulur.

---

## Çalıştırma

### 1. GUI Başlatıcı

```bash
python -m launcher.app
```

GUI'den:
1. Kaynak seçin: **Canlı Kamera** veya **Video Dosyası**
2. Kamera seçin (IZUM listesinden) veya dosya göz atın
3. **ROI Düzenle** ile şerit bölgelerini çizin
4. **Basla** ile analizi başlatın

### 2. Spark Streaming

Analiz çalışırken ayrı bir terminalde:

```bash
python spark_layer/stream_processor.py
```

11 sorgu otomatik başlar ve Kafka'dan okuyup InfluxDB'ye yazar.

### 3. Grafana Dashboard

`http://localhost:3000` adresine gidin (varsayılan: admin/admin).
InfluxDB veri kaynağı provisioning ile otomatik yapılandırılmıştır.

---

## Tespit Modeli

Model, IZUM kameralarından toplanan trafik görüntüleri üzerinde RF-DETR üzerine fine-tune edilmiştir.

| Sınıf ID | Sınıf |
|----------|-------|
| 0 | Bus (Otobüs) |
| 1 | Car (Araba) |
| 2 | Dolmus (Dolmuş) |
| 3 | Motorcycle (Motosiklet) |
| 4 | Taxi (Taksi) |
| 5 | Truck (Kamyon) |

**Mimari:** `dinov2_windowed_base`, 512 gizli boyut, 5 decoder katmanı, 300 sorgu
**Ağırlık dosyası:** `best.pt` (proje kökünde, `.gitignore` ile hariç tutulmuş)

---

## Anomali Tespiti

Öncelik sırasına göre:

| Anomali | Koşul |
|---------|-------|
| `stopped_vehicle` | Araç eşik süre boyunca hareketsiz |
| `wrong_way` | Araç şeridin beklenen yönünün tam tersine gidiyor |
| `sudden_slowdown` | Kısa pencerede hız ani düşüş |

---

## Spark Sorguları

| Sorgu | Açıklama | Pencere |
|-------|----------|---------|
| Q1 | Şerit + sınıf bazlı araç sayısı | Tümbling 1 dk |
| Q2 | Şerit + sınıf bazlı hız takibi | Kayan 2 dk / 30 sn |
| Q3 | Ağır araç oranı | Tümbling 5 dk |
| Q4 | Anomali yoğunluğu | Tümbling 1 dk |
| Q5 | Trafik durum geçişleri | Kayan 3 dk / 1 dk |
| Q6 | Şerit hız metrikleri | Kayan 3 dk / 1 dk |
| Q7 | Şerit akış hızı | Tümbling 1 dk |
| Q8 | Şerit bekleme süresi | Tümbling 2 dk |
| Q9 | Şerit araç sayısı (snapshot) | Tümbling 1 dk |
| Q10 | Yön analizi | Tümbling 1 dk |
| Q11 | Araç sınıf dağılımı | Tümbling 1 dk |

---

## Clean Architecture

Sistem [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) prensiplerine göre tasarlanmıştır:

```
domain/          — Saf Python, sıfır dış bağımlılık (VehicleClass, Direction, AnomalyType)
application/     — İş mantığı orkestrasyonu (Analyzer, FrameProcessor, EventBuilder)
services/        — Domain servisleri (VehicleMetrics, SceneMetrics, AnomalyDetector)
adapters/        — Dış sistem bağdaştırıcıları (RF-DETR, ByteTrack, Kafka, InfluxDB)
infrastructure/  — Teknik altyapı (VideoLoop, ConfigLoader)
visualization/   — Görüntü overlay katmanı (Renderer, ROISelector)
```

Bağımlılıklar yalnızca içe doğru akar — `domain` katmanı hiçbir şeye bağımlı değildir.

---

## Lisans

Bu proje bir mezuniyet tezi kapsamında geliştirilmiştir.
