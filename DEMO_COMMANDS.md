# Demo Commands (Teacher Presentation)

Use these commands in order.

## 1) Open project
```powershell
cd D:\LAKSHYA\Desktop\Predictive_maintainence
```

## 2) Build data + model + overfitting graph
```powershell
d:/LAKSHYA/Desktop/Predictive_maintainence/.venv/Scripts/python.exe run_combined_pipeline.py
d:/LAKSHYA/Desktop/Predictive_maintainence/.venv/Scripts/python.exe export_all_predictions.py
d:/LAKSHYA/Desktop/Predictive_maintainence/.venv/Scripts/python.exe overfitting_graph.py
```

## 3) Start Kafka broker
```powershell
docker start redpanda
```

## 4) Start Spark stream (Terminal 1)
```powershell
.\run_spark_stream.ps1
```

## 5) Send live data to Kafka (Terminal 2)
```powershell
.\run_kafka_producer.ps1
```

## 6) Show outputs
```powershell
dir .\outputs
dir .\outputs\predictions
dir .\outputs\stream_predictions
dir .\models
```

## 7) Open graph and teacher notes
```powershell
start .\outputs\overfitting_graph.png
start .\TEACHER_DEMO.md
```

## 8) Stop broker after demo
```powershell
docker stop redpanda
```
