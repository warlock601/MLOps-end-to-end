# MLOps-end-to-end
This project demonstrates a complete Machine Learning Operations (MLOps) workflow — taking a model from raw data all the way to automated deployment and continuous retraining.

### Overview:
The pipeline:
- Trains an ML model (on Kaggle data, e.g., Titanic dataset) using scikit-learn.
- Tracks experiments and metrics using MLflow, storing models and parameters for comparison.
- Packages the trained model inside a Docker container with a FastAPI inference server.
- Deploys the container to Kubernetes (or AWS SageMaker) as a scalable REST API.
- Implements CI/CD using GitHub Actions (or Jenkins) to automate testing, image building, and deployment.
- Monitors data drift in production using Evidently, automatically triggering model retraining when drift is detected.

### Pipeline Flow
- Data Preparation & Training → preprocess Kaggle data, train model, log to MLflow.
- Model Packaging & Serving → build FastAPI Docker image for inference.
- Continuous Integration (CI) → auto test, build, and push images on code commits.
- Continuous Deployment (CD) → auto deploy to K8s or SageMaker.
- Monitoring → periodically run drift detection on production data.
- Automated Retraining → if drift detected, trigger GitHub Action to retrain and redeploy the model.

### Tach Stack used:
- MLOps / AIOps: MLflow, Evidently, Drift Detection
- DevOps: Docker, Kubernetes, GitHub Actions, CI/CD
- Data Science: scikit-learn, Pandas, Model Evaluation
- Cloud & Deployment: AWS SageMaker / EKS / Minikube
- Monitoring & Automation: Scheduled jobs, Retrain triggers
