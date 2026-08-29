#!/bin/bash
set -e

BASE_URL="http://localhost:8000"
echo "=========================================="
echo "Running post-deployment smoke tests..."
echo "=========================================="

echo ""
echo "[Test 1] Health Check Endpoint..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/health)
if [ "$HEALTH_RESPONSE" -eq 200 ]; then
    echo "  PASSED - Health endpoint returned 200"
else
    echo "  FAILED - Health endpoint returned $HEALTH_RESPONSE"
    exit 1
fi

echo ""
echo "[Test 2] Health Response Content..."
HEALTH_BODY=$(curl -s $BASE_URL/health)
echo "  Response: $HEALTH_BODY"
if echo "$HEALTH_BODY" | grep -q '"status":"healthy"'; then
    echo "  PASSED - Service is healthy"
else
    echo "  FAILED - Unexpected health response"
    exit 1
fi

echo ""
echo "[Test 3] Prediction Endpoint..."
PRED_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -F "pFile=@data/sample/cat_sample_1.jpg" \
    $BASE_URL/predict)
HTTP_CODE=$(echo "$PRED_RESPONSE" | tail -1)
PRED_BODY=$(echo "$PRED_RESPONSE" | head -1)
echo "  Response: $PRED_BODY"
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "  PASSED - Prediction endpoint returned 200"
else
    echo "  FAILED - Prediction endpoint returned $HTTP_CODE"
    exit 1
fi

echo ""
echo "[Test 4] Metrics Endpoint..."
METRICS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/metrics)
if [ "$METRICS_RESPONSE" -eq 200 ]; then
    echo "  PASSED - Metrics endpoint returned 200"
else
    echo "  FAILED - Metrics endpoint returned $METRICS_RESPONSE"
    exit 1
fi

echo ""
echo "[Test 5] Prometheus Metrics Endpoint..."
PROMETHEUS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/metrics/prometheus)
if [ "$PROMETHEUS_RESPONSE" -eq 200 ]; then
    echo "  PASSED - Prometheus metrics endpoint returned 200"
else
    echo "  FAILED - Prometheus metrics endpoint returned $PROMETHEUS_RESPONSE"
    exit 1
fi

echo ""
echo "=========================================="
echo "All smoke tests passed!"
echo "=========================================="