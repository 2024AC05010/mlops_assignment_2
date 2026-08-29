# Project Running Guide for Screen Recording

## Step-by-Step Instructions (4-5 minutes total)

### Step 1: Project Overview (30 seconds)
```bash
# Show project structure
ls -la
```
**Say:** "This is my MLOps project for cats vs dogs classification. Let me show you the complete pipeline."

### Step 2: Code Change (1 minute)
```bash
# Make a small change to demonstrate CI/CD
# Edit src/train.py - change learning rate from 0.001 to 0.0005
# Open the file in your editor and make the change

# Commit the change
git add src/train.py
git commit -m "Experiment: Reduce learning rate for better convergence"
git push
```
**Say:** "I'm making a small hyperparameter change to trigger the CI/CD pipeline."

### Step 3: Show CI/CD Pipeline (1 minute)
- Open GitHub in browser
- Navigate to Actions tab
- Show the pipeline running (Test → Build → Deploy)
**Say:** "The GitHub Actions workflow is automatically testing, building, and deploying the changes."

### Step 4: Test Deployment (1 minute)
```bash
# Check service health
curl http://localhost:8000/health

# Run smoke tests
cd deployment
bash smoke_test.sh
cd ..
```
**Say:** "Let me verify the deployment is working correctly with health checks and smoke tests."

### Step 5: Test Model Prediction (1 minute)
```bash
# Test the API with a sample image
curl -X POST -F "pFile=@data/sample/cat_sample.jpg" http://localhost:8000/predict
```
**Say:** "Now I'll test the actual model prediction with a cat image."

### Step 6: Show Monitoring (30 seconds)
- Open browser to http://localhost:3000 (Grafana)
- Login with admin/admin
- Show the monitoring dashboard
**Say:** "Grafana provides real-time monitoring of the API performance."

### Step 7: Show MLflow (30 seconds)
```bash
# Start MLflow UI
mlflow ui
```
- Open browser to http://localhost:5000
- Show experiment tracking
**Say:** "MLflow tracks all my experiments and model versions."

### Step 8: Conclusion (30 seconds)
**Say:** "The complete MLOps pipeline is working - from code change to automated deployment with monitoring and experiment tracking."

---

## Pre-Recording Setup (Do this before recording)

### 1. Start Services
```bash
# Start the API and monitoring stack
cd deployment
docker-compose up -d
cd ..
```

### 2. Verify Services are Running
```bash
# Check API health
curl http://localhost:8000/health

# Check Grafana is accessible
# Open http://localhost:3000 in browser
```

### 3. Ensure Git is Ready
```bash
# Check git status
git status
```

---

## Quick Reference Commands

```bash
# Project structure
ls -la

# Git operations
git add src/train.py
git commit -m "Experiment: Reduce learning rate"
git push

# Health check
curl http://localhost:8000/health

# Smoke tests
cd deployment
bash smoke_test.sh
cd ..

# Model prediction
curl -X POST -F "pFile=@data/sample/cat_sample.jpg" http://localhost:8000/predict

# MLflow UI
mlflow ui
# Then open http://localhost:5000

# Grafana
# Open http://localhost:3000 (admin/admin)
```

---

## Tips for Smooth Recording

1. **Practice once** before actual recording
2. **Have all browser tabs ready** (GitHub, Grafana, MLflow)
3. **Keep terminal windows organized** (split screen works well)
4. **Speak clearly and at a steady pace**
5. **Point to key elements** on screen when explaining
6. **Keep the recording under 5 minutes**

---

## Services to Show

- **GitHub Actions**: Automated CI/CD pipeline
- **API Health Check**: http://localhost:8000/health
- **Model Prediction**: Working prediction endpoint
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **MLflow UI**: http://localhost:5000

---

## Cleanup After Recording

```bash
# Stop services if needed
cd deployment
docker-compose down
cd ..
```
