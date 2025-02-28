FROM python:3.11-slim
WORKDIR /app
RUN pip install requests pandas scikit-learn joblib pytz numpy prometheus-client
COPY src/cpu_predictor.py .
RUN mkdir -p app/data app/models app/output
EXPOSE 8000
CMD ["python", "cpu_predictor.py"]