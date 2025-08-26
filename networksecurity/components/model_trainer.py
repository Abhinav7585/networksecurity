import os
import sys
import shutil

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='Abhinav7585', repo_name='networksecurity', mlflow=True)


# Recommended: export these in your shell rather than hardcoding.
# e.g.
# export MLFLOW_TRACKING_URI="https://dagshub.com/<username>/<repo>.mlflow"
# export MLFLOW_TRACKING_USERNAME="<username_or_token>"
# export MLFLOW_TRACKING_PASSWORD="<password_or_token>"
# (On Windows PowerShell: $env:MLFLOW_TRACKING_URI = "..." )
os.environ.setdefault("MLFLOW_TRACKING_URI", "https://dagshub.com/Abhinav7585/networksecurity.mlflow")
# Optionally set username/token and password via env vars outside code:
# os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "<username_or_token>")
# os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", "<password_or_token>")


def get_or_create_experiment_id(name: str):
    """
    Create or get an MLflow experiment by name (pattern taken from DagsHub docs).
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classificationmetric):
        """
        Log metrics and model artifacts to MLflow (DagsHub-compatible).
        Saves model locally using mlflow.sklearn.save_model() then uploads folder via mlflow.log_artifacts().
        """
        try:
            # Ensure MLflow uses DagsHub tracking URI (per docs)
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Abhinav7585/networksecurity.mlflow")
            mlflow.set_tracking_uri(tracking_uri)

            # Optionally get or create an experiment to group runs (DagsHub docs recommend this helper)
            experiment_name = "NetworkSecurity_Experiments"
            experiment_id = get_or_create_experiment_id(experiment_name)

            # Start an MLflow run under the experiment
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log metrics (single values)
                mlflow.log_metric("f1_score", float(classificationmetric.f1_score))
                mlflow.log_metric("precision", float(classificationmetric.precision_score))
                mlflow.log_metric("recall_score", float(classificationmetric.recall_score))

                # Optionally log params / metadata
                mlflow.log_param("model_class", best_model.__class__.__name__)

                # Save the sklearn model locally (mlflow format) then upload as artifacts.
                temp_dir = "temp_saved_model"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)

                # This creates MLmodel, conda.yaml (if possible), and model binary in temp_dir
                mlflow.sklearn.save_model(best_model, temp_dir)

                # Upload the entire directory as artifacts under "model" path (per docs)
                mlflow.log_artifacts(temp_dir, artifact_path="model")

                # Cleanup local temp dir
                shutil.rmtree(temp_dir)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params = {
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{},
                "AdaBoost":{
                    'learning_rate':[.1,.01,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test,
                models=models, param=params
            )
            
            # best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Fit the chosen model if evaluate_models doesn't already fit it (ensure model is trained)
            # (Assuming evaluate_models trains models; if not, uncomment the line below)
            # best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            # Track train metrics + artifacts
            self.track_mlflow(best_model, classification_train_metric)

            # Evaluate on test set
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Track test metrics + artifacts
            self.track_mlflow(best_model, classification_test_metric)

            # Load preprocessor
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
                
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            # Save trained model instance (preprocessor + model)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
