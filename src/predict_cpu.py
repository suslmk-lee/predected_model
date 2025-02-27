import requests
import pandas as pd
import joblib
import time

# Prometheus 데이터 가져오기 (최신 데이터만)
def fetch_prometheus_data():
    url = "http://133.186.215.116:32211/api/v1/query_range"
    query = 'rate(container_cpu_usage_seconds_total{namespace="kube-system", container_label_io_kubernetes_pod_name="calico-node-lw4h8", cpu="cpu00"}[5m])'
    params = {
        "query": query,
        "start": int(time.time()) - 3600,  # 최근 1시간
        "end": int(time.time()),
        "step": "60s"
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

# 저장된 모델로 예측
def predict_with_saved_model(df):
    model = joblib.load("cpu_predictor_model.pkl")
    next_time = pd.DataFrame([[len(df) + 1]], columns=["time_seq"])
    prediction = model.predict(next_time)[0]
    return prediction

if __name__ == "__main__":
    try:
        raw_data = fetch_prometheus_data()
        df = process_data(raw_data)
        prediction = predict_with_saved_model(df)
        result = f"다음 CPU 사용량 예측 (calico-node-lw4h8, cpu00): {prediction}"
        print(result)
        with open("../output/prediction.txt", "a") as f:
            f.write(f"{time.ctime()}: {result}\n")
    except Exception as e:
        print(f"오류 발생: {e}")