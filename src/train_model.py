import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import time

# Prometheus 데이터 가져오기
def fetch_prometheus_data():
    url = "http://133.186.215.116:32211/api/v1/query_range"  # 로컬에서 실행 시
    query = 'rate(container_cpu_usage_seconds_total{namespace="kube-system", container_label_io_kubernetes_pod_name="calico-node-lw4h8", cpu="cpu00"}[5m])'
    params = {
        "query": query,
        "start": int(time.time()) - 86400,  # 최근 1일 데이터
        "end": int(time.time()),
        "step": "300s"  # 5분 간격
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# 데이터 처리
def process_data(data):
    rows = []
    for result in data["data"]["result"]:
        for timestamp, value in result["values"]:
            rows.append({"timestamp": timestamp, "value": float(value)})
    df = pd.DataFrame(rows)
    df["time_seq"] = range(1, len(df) + 1)
    return df

# 모델 학습 및 저장
def train_and_save_model(df):
    X = df[["time_seq"]]
    y = df["value"]
    model = LinearRegression()
    model.fit(X, y)
    # 모델 저장
    joblib.dump(model, "cpu_predictor_model.pkl")
    print("모델 저장 완료!")

if __name__ == "__main__":
    raw_data = fetch_prometheus_data()
    df = process_data(raw_data)
    train_and_save_model(df)