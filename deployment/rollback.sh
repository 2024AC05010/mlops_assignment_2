#!/bin/bash
set -e

# Simple Deployment Rollback Script
# Usage: bash deployment/rollback.sh

echo "=========================================="
echo "Simple Deployment Rollback"
echo "=========================================="

COMPOSE_FILE="deployment/docker-compose.yml"

echo "Stopping current deployment..."
docker compose -f $COMPOSE_FILE down

echo "Pulling previous stable version..."
# Pull the previous version (you can specify the tag)
docker pull ${DOCKER_USERNAME:-myuser}/cats-dogs-classifier:previous || echo "Using local fallback"

echo "Starting rollback deployment..."
docker compose -f $COMPOSE_FILE up -d

echo "Waiting for service to start..."
sleep 15

echo "Verifying rollback..."
HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$HEALTH_CHECK" -eq 200 ]; then
    echo "Rollback successful - Service is healthy"
else
    echo "Rollback verification failed"
    exit 1
fi
