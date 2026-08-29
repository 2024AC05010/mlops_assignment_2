from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Prometheus metrics
HTTP_REQUESTS_TOTAL = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
HTTP_REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request latency')
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions', ['prediction_class'])

class MetricsCollector:
    def __init__(self):
        self.tRequest_Count = 0
        self.tTotal_Latency = 0.0
        self.tPrediction_Counts = {"cat": 0, "dog": 0}

    def record_request(self, pLatency, pPrediction):
        self.tRequest_Count += 1
        self.tTotal_Latency += pLatency
        if pPrediction in self.tPrediction_Counts:
            self.tPrediction_Counts[pPrediction] += 1
        
        # Update Prometheus metrics
        HTTP_REQUEST_DURATION.observe(pLatency)
        PREDICTION_COUNTER.labels(prediction_class=pPrediction).inc()

    def get_metrics(self):
        tAvg_Latency = self.tTotal_Latency / self.tRequest_Count if self.tRequest_Count > 0 else 0.0
        return {
            "total_requests": self.tRequest_Count,
            "average_latency_seconds": round(tAvg_Latency, 4),
            "prediction_distribution": self.tPrediction_Counts
        }


tMetrics_Instance = MetricsCollector()