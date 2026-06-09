$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
$env:HADOOP_HOME="C:\hadoop"
[System.Environment]::SetEnvironmentVariable("hadoop.home.dir","C:\hadoop","Process")
$env:PATH="$env:JAVA_HOME\bin;$env:HADOOP_HOME\bin;$env:PATH"
$env:PYSPARK_PYTHON="d:/LAKSHYA/Desktop/Predictive_maintainence/.venv/Scripts/python.exe"
$env:PYSPARK_DRIVER_PYTHON="d:/LAKSHYA/Desktop/Predictive_maintainence/.venv/Scripts/python.exe"
$env:PYSPARK_SUBMIT_ARGS="--packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 pyspark-shell"

d:/LAKSHYA/Desktop/Predictive_maintainence/.venv/Scripts/python.exe -m src.spark.streaming_stub
