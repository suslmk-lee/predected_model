import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import time
import os
from datetime import datetime, timedelta
import pytz
import numpy as np

def fetch_prometheus_data(start_time, end_time):
    url = "http://133.186.215.116:32211/api/v1/query_range"
    # 전체 파드에 대한 쿼리 (필터 제거)
    query = 'rate(container_cpu_usage_seconds_total[5m])'
    kst = pytz.timezone("Asia/Seoul")
    utc = pytz.UTC
    params = {
        "query": query,
        "start": int(start_time),
        "end": int(end_time),
        "step": "300s"
    }
    print(f"요청 URL: {url}")
    print(f"쿼리: {query}")
    print(f"시간 범위 (KST): {datetime.fromtimestamp(start_time, kst)} ~ {datetime.fromtimestamp(end_time, kst)}")
    print(f"시간 범위 (UTC): {datetime.fromtimestamp(start_time, utc)} ~ {datetime.fromtimestamp(end_time, utc)}")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Prometheus 응답: {len(data['data']['result'])} 개 메트릭 수집")
        return data
    except Exception as e:
        print(f"Prometheus 데이터 가져오기 실패: {e}")
        return {"data": {"result": []}}

def update_data_file():
    data_file = "app/data/cpu_usage_data.csv"
    kst = pytz.timezone("Asia/Seoul")
    utc = pytz.UTC
    now_kst = datetime.now(utc).astimezone(kst)
    today_10am_kst = now_kst.replace(hour=10, minute=0, second=0, microsecond=0)
    yesterday_10am_kst = today_10am_kst - timedelta(days=1)
    thirty_days_ago_kst = today_10am_kst - timedelta(days=30)

    today_10am_utc = today_10am_kst.astimezone(utc)
    yesterday_10am_utc = yesterday_10am_kst.astimezone(utc)
    thirty_days_ago_utc = thirty_days_ago_kst.astimezone(utc)

    print(f"현재 시간 (KST): {now_kst}")

    # 기존 데이터 로드
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        print(f"기존 데이터 로드: {len(df)} 행")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df = pd.DataFrame(columns=["timestamp", "value", "pod_name", "namespace"])
        print("새 데이터 파일 생성")

    # 30일 이전 데이터 제거
    if not df.empty and "timestamp" in df.columns:
        thirty_days_ago_utc_np = np.datetime64(thirty_days_ago_utc)
        df = df[df["timestamp"] >= thirty_days_ago_utc_np]
        print(f"30일 이전 데이터 제거 후: {len(df)} 행")
    else:
        print("기존 데이터가 없거나 timestamp 열 없음")

    # 클러스터 전체 데이터 가져오기
    raw_data = fetch_prometheus_data(
        (today_10am_utc - timedelta(days=2)).timestamp(),
        (today_10am_utc + timedelta(days=1)).timestamp()
    )
    rows = []
    for result in raw_data["data"]["result"]:
        pod_name = result["metric"].get("container_label_io_kubernetes_pod_name", "unknown")
        namespace = result["metric"].get("namespace", "unknown")
        for timestamp, value in result["values"]:
            rows.append({
                "timestamp": pd.to_datetime(timestamp, unit="s"),
                "value": float(value),
                "pod_name": pod_name,
                "namespace": namespace
            })
    new_data = pd.DataFrame(rows)
    print(f"새 데이터: {len(new_data)} 행")

    # 데이터 합치기
    df = pd.concat([df, new_data]).drop_duplicates(subset=["timestamp", "pod_name", "namespace"]).sort_values("timestamp")
    # 파드별 time_seq 생성
    df["time_seq"] = df.groupby(["pod_name", "namespace"]).cumcount() + 1
    print(f"최종 데이터: {len(df)} 행")

    if not df.empty:
        df.to_csv(data_file, index=False)
        print(f"CSV 저장 완료: {data_file}")
    else:
        print("저장할 데이터 없음")
    return df

def retrain_model(df):
    if df.empty:
        print("데이터가 없어 모델 학습 불가")
        return {}
    models = {}
    # 파드별로 모델 학습
    for (pod_name, namespace), group in df.groupby(["pod_name", "namespace"]):
        if len(group) < 2:  # 최소 2개 데이터 필요
            print(f"{pod_name} ({namespace}): 데이터 부족으로 모델 학습 스킵")
            continue
        X = group[["time_seq"]]
        y = group["value"]
        model = LinearRegression()
        model.fit(X, y)
        model_file = f"app/models/cpu_predictor_model_{pod_name}_{namespace}.pkl"
        joblib.dump(model, model_file)
        models[(pod_name, namespace)] = model_file
        print(f"모델 학습 완료: {pod_name} ({namespace})")
    return models

def predict_cpu_usage(df, models):
    predictions = {}
    for (pod_name, namespace), group in df.groupby(["pod_name", "namespace"]):
        model_file = f"app/models/cpu_predictor_model_{pod_name}_{namespace}.pkl"
        if not os.path.exists(model_file):
            print(f"{pod_name} ({namespace}): 모델 파일 없음")
            continue
        model = joblib.load(model_file)
        next_time = pd.DataFrame([[len(group) + 1]], columns=["time_seq"])
        prediction = model.predict(next_time)[0]
        predictions[(pod_name, namespace)] = prediction
    return predictions

if __name__ == "__main__":
    try:
        os.makedirs("app/data", exist_ok=True)
        os.makedirs("app/models", exist_ok=True)
        os.makedirs("app/output", exist_ok=True)

        df = update_data_file()

        # 모델 재학습
        models = retrain_model(df)

        # 예측
        predictions = predict_cpu_usage(df, models)
        if predictions:
            for (pod_name, namespace), prediction in predictions.items():
                result = f"다음 CPU 사용량 예측 ({pod_name}, {namespace}): {prediction}"
                print(result)
                with open("app/output/prediction.txt", "a") as f:
                    f.write(f"{time.ctime()}: {result}\n")
        else:
            print("예측 실패: 모델 또는 데이터 없음")

    except Exception as e:
        print(f"오류 발생: {e}")