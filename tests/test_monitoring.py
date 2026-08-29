import pytest
from api.monitoring import MetricsCollector


class TestMonitoring:

    def test_metrics_collector_initialization(self):
        """Test that metrics collector initializes correctly"""
        collector = MetricsCollector()
        assert collector.tRequest_Count == 0
        assert collector.tTotal_Latency == 0.0
        assert collector.tPrediction_Counts == {"cat": 0, "dog": 0}

    def test_record_request(self):
        """Test recording a request"""
        collector = MetricsCollector()
        collector.record_request(0.5, "cat")
        
        assert collector.tRequest_Count == 1
        assert collector.tTotal_Latency == 0.5
        assert collector.tPrediction_Counts["cat"] == 1
        assert collector.tPrediction_Counts["dog"] == 0

    def test_get_metrics(self):
        """Test getting metrics"""
        collector = MetricsCollector()
        collector.record_request(0.3, "dog")
        collector.record_request(0.7, "cat")
        
        metrics = collector.get_metrics()
        
        assert metrics["total_requests"] == 2
        assert metrics["average_latency_seconds"] == 0.5
        assert metrics["prediction_distribution"]["cat"] == 1
        assert metrics["prediction_distribution"]["dog"] == 1

    def test_average_latency_calculation(self):
        """Test average latency is calculated correctly"""
        collector = MetricsCollector()
        collector.record_request(1.0, "cat")
        collector.record_request(2.0, "dog")
        collector.record_request(3.0, "cat")
        
        metrics = collector.get_metrics()
        assert metrics["average_latency_seconds"] == 2.0
