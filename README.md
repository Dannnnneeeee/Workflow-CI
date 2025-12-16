# Workflow-CI: Toyota Price Prediction CI/CD

CI/CD Pipeline untuk automatic model training menggunakan MLflow Project dan GitHub Actions.

## �� Features

- ✅ Automatic model training dengan MLflow Project
- ✅ Hyperparameter configuration via GitHub Actions
- ✅ Artifacts saved to GitHub repository
- ✅ Docker image published to Docker Hub
- ✅ Re-training on push to MLProject/

##  Quick Start

### Manual Trigger
1. Go to **Actions** tab
2. Select **MLflow CI/CD Pipeline**
3. Click **Run workflow**
4. Set parameters (optional)
5. Click **Run workflow**

### Auto Trigger
Push changes to `MLProject/` folder will automatically trigger training.

##  Docker Image
```bash
docker pull daneeeee/toyota-mlflow-ci:latest
```

**Docker Hub:** https://hub.docker.com/r/daneeeee/toyota-mlflow-ci

## 📦 Artifacts

Trained models and artifacts are:
1. Uploaded to GitHub Actions (90 days retention)
2. Committed to `saved_artifacts/` folder in repository

## 🏗️ Structure
```
Workflow-CI/
├── .github/workflows/
│   └── ci_workflow.yml
├── MLProject/
│   ├── conda.yaml
│   ├── MLProject
│   ├── modelling.py
│   └── toyota_clean.csv
└── saved_artifacts/
```

## 👤 Author

Muhammad Wildan - MSML Dicoding Submission
