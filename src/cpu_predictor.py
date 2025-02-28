import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import time
import os
from datetime import datetime, timedelta
import pytz
import numpy as np
from prometheus_client import start_http_server, Gauge

# 환경 변수에서 설정값 읽기 (기본값 제공)
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
STEP_INTERVAL = os.environ.get("STEP_INTERVAL", "300s")
TIME_RANGE_DAYS = int(os.environ.get("TIME_RANGE_DAYS", "30"))

CPU_PREDICTION_GAUGE = Gauge(
    "cpu_usage_prediction",
    "Predicted CPU usage for pods",
    ["pod_name", "namespace"]
)

def fetch_prometheus_data(start_time, end_time, retries=3, delay=5):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    kst = pytz.timezone("Asia/Seoul")
    utc = pytz.UTC
    
    query_cpu = 'rate(container_cpu_usage_seconds_total[5m])'
    params_cpu = {"query": query_cpu, "start": int(start_time), "end": int(end_time), "step": STEP_INTERVAL}
    
    query_mem = 'container_memory_usage_bytes'
    params_mem = {"query": query_mem, "start": int(start_time), "end": int(end_time), "step": STEP_INTERVAL}

    with open("app/output/log.txt", "a") as f:
        f.write(f"{time.ctime()}: 요청 URL: {url}\n")
        f.write(f"{time.ctime()}: 시간 범위 (KST): {datetime.fromtimestamp(start_time, kst)} ~ {datetime.fromtimestamp(end_time, kst)}\n")
    
    data_cpu = {"data": {"result": []}}
    for attempt in range(retries):
        try:
            response_cpu = requests.get(url, params=params_cpu, timeout=30)
            response_cpu.raise_for_status()
            data_cpu = response_cpu.json()
            with open("app/output/log.txt", "a") as f:
                f.write(f"{time.ctime()}: CPU Prometheus 응답: {len(data_cpu['data']['result'])} 개 메트릭 수집\n")
                f.write(f"{time.ctime()}: CPU 데이터 샘플: {data_cpu['data']['result'][:2]}\n")
            break
        except Exception as e:
            with open("app/output/log.txt", "a") as f:
                f.write(f"{time.ctime()}: CPU 데이터 가져오기 실패 (시도 {attempt+1}/{retries}): {e}\n")
            if attempt < retries - 1:
                time.sleep(delay)

    data_mem = {"data": {"result": []}}
    for attempt in range(retries):
        try:
            response_mem = requests.get(url, params=params_mem, timeout=30)
            response_mem.raise_for_status()
            data_mem = response_mem.json()
            with open("app/output/log.txt", "a") as f:
                f.write(f"{time.ctime()}: 메모리 Prometheus 응답: {len(data_mem['data']['result'])} 개 메트릭 수집\n")
                f.write(f"{time.ctime()}: 메모리 데이터 샘플: {data_mem['data']['result'][:2]}\n")
            break
        except Exception as e:
            with open("app/output/log.txt", "a") as f:
                f.write(f"{time.ctime()}: 메모리 데이터 가져오기 실패 (시도 {attempt+1}/{retries}): {e}\n")
            if attempt < retries - 1:
                time.sleep(delay)

    return data_cpu, data_mem

def process_prometheus_data(data_cpu, data_mem):
    cpu_rows = []
    for r in data_cpu["data"]["result"]:
        pod_name = r["metric"].get("container_label_io_kubernetes_pod_name", "unknown")
        namespace = r["metric"].get("container_label_io_kubernetes_pod_namespace", "unknown")
        for timestamp, value in r["values"]:
            cpu_rows.append({
                "timestamp": pd.to_datetime(int(timestamp), unit="s"),
                "cpu_value": float(value),
                "pod_name": pod_name,
                "namespace": namespace
            })
    df_cpu = pd.DataFrame(cpu_rows)
    if df_cpu.empty:
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: CPU 데이터 비어 있음\n")
    else:
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: CPU 데이터 네임스페이스: {df_cpu['namespace'].unique()}\n")

    mem_rows = []
    for r in data_mem["data"]["result"]:
        pod_name = r["metric"].get("container_label_io_kubernetes_pod_name", "unknown")
        namespace = r["metric"].get("container_label_io_kubernetes_pod_namespace", "unknown")
        for timestamp, value in r["values"]:
            mem_rows.append({
                "timestamp": pd.to_datetime(int(timestamp), unit="s"),
                "memory_usage": float(value),
                "pod_name": pod_name,
                "namespace": namespace
            })
    df_mem = pd.DataFrame(mem_rows)
    if df_mem.empty:
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: 메모리 데이터 비어 있음\n")
    else:
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: 메모리 데이터 네임스페이스: {df_mem['namespace'].unique()}\n")

    if df_cpu.empty or df_mem.empty:
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: CPU 또는 메모리 데이터 부족으로 병합 실패\n")
        return pd.DataFrame(columns=["timestamp", "cpu_value", "memory_usage", "pod_name", "namespace"])
    
    df = pd.merge(df_cpu, df_mem, on=["timestamp", "pod_name", "namespace"], how="inner")
    df = df.drop_duplicates(subset=["timestamp", "pod_name", "namespace"])
    with open("app/output/log.txt", "a") as f:
        f.write(f"{time.ctime()}: 병합 후 데이터: {len(df)} 행\n")
        f.write(f"{time.ctime()}: 병합 후 네임스페이스 목록: {df['namespace'].unique()}\n")
    return df

def update_data_file():
    data_file = "app/data/cpu_usage_data.csv"
    kst = pytz.timezone("Asia/Seoul")
    utc = pytz.UTC
    now_kst = datetime.now(utc).astimezone(kst)
    today_10am_kst = now_kst.replace(hour=10, minute=0, second=0, microsecond=0)
    thirty_days_ago_kst = today_10am_kst - timedelta(days=TIME_RANGE_DAYS)

    today_10am_utc = today_10am_kst.astimezone(utc)
    thirty_days_ago_utc = thirty_days_ago_kst.astimezone(utc)

    with open("app/output/log.txt", "a") as f:
        f.write(f"{time.ctime()}: 현재 시간 (KST): {now_kst}\n")

    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: 기존 데이터 로드: {len(df)} 행\n")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df = pd.DataFrame(columns=["timestamp", "cpu_value", "memory_usage", "pod_name", "namespace"])
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: 새 데이터 파일 생성\n")

    if not df.empty and "timestamp" in df.columns:
        thirty_days_ago_utc_np = np.datetime64(thirty_days_ago_utc.replace(tzinfo=None))
        df = df[df["timestamp"] >= thirty_days_ago_utc_np]
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: {TIME_RANGE_DAYS}일 이전 데이터 제거 후: {len(df)} 행\n")
    else:
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: 기존 데이터가 없거나 timestamp 열 없음\n")

    data_cpu, data_mem = fetch_prometheus_data(
        (today_10am_utc - timedelta(days=TIME_RANGE_DAYS)).timestamp(),
        (today_10am_utc + timedelta(days=1)).timestamp()
    )
    new_data = process_prometheus_data(data_cpu, data_mem)
    with open("app/output/log.txt", "a") as f:
        f.write(f"{time.ctime()}: 새 데이터: {len(new_data)} 행\n")

    df = pd.concat([df, new_data]).drop_duplicates(subset=["timestamp", "pod_name", "namespace"]).sort_values(["pod_name", "namespace", "timestamp"])
    df["time_seq"] = df.groupby(["pod_name", "namespace"]).cumcount() + 1
    with open("app/output/log.txt", "a") as f:
        f.write(f"{time.ctime()}: 최종 데이터: {len(df)} 행\n")

    if not df.empty:
        df.to_csv(data_file, index=False)
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: CSV 저장 완료: {data_file}\n")
    else:
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: 저장할 데이터 없음\n")
    return df

def retrain_model(df):
    if df.empty:
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: 데이터가 없어 모델 학습 불가\n")
        return {}
    models = {}
    for (pod_name, namespace), group in df.groupby(["pod_name", "namespace"]):
        group.fillna({"cpu_value": group["cpu_value"].mean(), "memory_usage": group["memory_usage"].mean()}, inplace=True)
        group_clean = group.dropna(subset=["time_seq", "memory_usage", "cpu_value"])
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: {pod_name} ({namespace}): NaN 제거 후 {len(group_clean)} 행\n")
            f.write(f"{time.ctime()}: 데이터 샘플: {group_clean[['time_seq', 'cpu_value', 'memory_usage']].tail(3)}\n")
        if len(group_clean) < 2:
            with open("app/output/log.txt", "a") as f:
                f.write(f"{time.ctime()}: {pod_name} ({namespace}): 데이터 부족으로 모델 학습 스킵\n")
            continue
        X = group_clean[["time_seq", "memory_usage"]]
        y = group_clean["cpu_value"]
        model = LinearRegression()
        model.fit(X, y)
        model_file = f"app/models/cpu_predictor_model_{pod_name}_{namespace}.pkl"
        joblib.dump(model, model_file)
        models[(pod_name, namespace)] = model_file
        with open("app/output/log.txt", "a") as f:
            f.write(f"{time.ctime()}: 모델 학습 완료: {pod_name} ({namespace})\n")
    return models

def predict_cpu_usage(df, models):
    predictions = {}
    for (pod_name, namespace), group in df.groupby(["pod_name", "namespace"]):
        model_file = f"app/models/cpu_predictor_model_{pod_name}_{namespace}.pkl"
        if model_file not in models.values():
            with open("app/output/log.txt", "a") as f:
                f.write(f"{time.ctime()}: {pod_name} ({namespace}): 모델 파일 없음\n")
            continue
        model = joblib.load(model_file)
        latest_row = group.dropna(subset=["time_seq", "memory_usage"]).iloc[-1]
        latest_time_seq = latest_row["time_seq"]
        latest_memory = latest_row["memory_usage"]
        next_time = pd.DataFrame([[latest_time_seq + 1, latest_memory]], columns=["time_seq", "memory_usage"])
        prediction = model.predict(next_time)[0]
        prediction = max(0, prediction)
        predictions[(pod_name, namespace)] = prediction
        CPU_PREDICTION_GAUGE.labels(pod_name=pod_name, namespace=namespace).set(prediction)
    return predictions

def main():
    start_http_server(8000)
    with open("app/output/log.txt", "a") as f:
        f.write(f"{time.ctime()}: Prometheus 메트릭 서버 시작: http://localhost:8000\n")

    os.makedirs("app/data", exist_ok=True)
    os.makedirs("app/models", exist_ok=True)
    os.makedirs("app/output", exist_ok=True)

    kst = pytz.timezone("Asia/Seoul")
    utc = pytz.UTC
    last_retrain_time = None

    while True:
        now_kst = datetime.now(utc).astimezone(kst)
        now_hour = now_kst.hour
        now_minute = now_kst.minute

        if last_retrain_time is None or (now_hour == 0 and now_minute == 0 and (last_retrain_time is None or last_retrain_time.day != now_kst.day)):
            with open("app/output/log.txt", "a") as f:
                f.write(f"{time.ctime()}: 모델 재학습 시작\n")
            df = update_data_file()
            models = retrain_model(df)
            last_retrain_time = now_kst
        else:
            models = {}
            for model_file in os.listdir("app/models"):
                if model_file.endswith(".pkl"):
                    pod_name = model_file.split("cpu_predictor_model_")[1].split("_")[0]
                    namespace = model_file.split("_")[1].replace(".pkl", "")
                    models[(pod_name, namespace)] = f"app/models/{model_file}"

        predictions = predict_cpu_usage(df, models)
        if predictions:
            for (pod_name, namespace), prediction in predictions.items():
                result = f"다음 CPU 사용량 예측 ({pod_name}, {namespace}): {prediction}"
                with open("app/output/log.txt", "a") as f:
                    f.write(f"{time.ctime()}: {result}\n")
                with open("app/output/prediction.txt", "a") as f:
                    if os.path.getsize("app/output/prediction.txt") > 10 * 1024 * 1024:
                        f.truncate(0)
                    f.write(f"{time.ctime()}: {result}\n")
        else:
            with open("app/output/log.txt", "a") as f:
                f.write(f"{time.ctime()}: 예측 실패: 모델 또는 데이터 없음\n")

        time.sleep(60)

if __name__ == "__main__":
    main()