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
