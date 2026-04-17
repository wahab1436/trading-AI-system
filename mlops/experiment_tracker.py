"""MLflow-based experiment tracking for all model experiments"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import json
import uuid
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Central experiment tracking using MLflow"""
    
    def __init__(
        self,
        tracking_uri: str = None,
        experiment_name: str = "trading-ai-system"
    ):
        """
        Initialize experiment tracker
        
        Args:
            tracking_uri: MLflow tracking server URI (None for local)
            experiment_name: Name of the experiment
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Experiment tracker initialized: {experiment_name} (ID: {self.experiment_id})")
        
    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """Get existing experiment or create new one"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if experiment:
            return experiment.experiment_id
        else:
            return self.client.create_experiment(experiment_name)
            
    @contextmanager
    def start_run(self, run_name: str = None, tags: Dict = None):
        """Context manager for MLflow runs"""
        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id) as run:
            if tags:
                mlflow.set_tags(tags)
            yield run
            logger.info(f"Run completed: {run.info.run_id} - {run_name}")
            
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to current run"""
        mlflow.log_metrics(metrics, step=step)
        
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log an artifact file"""
        mlflow.log_artifact(local_path, artifact_path)
        
    def log_model(
        self,
        model,
        model_name: str,
        model_type: str = "xgboost",
        signatures: Dict = None
    ):
        """Log model to MLflow registry"""
        
        if model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path=model_name,
                signature=signatures
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                model,
                artifact_path=model_name,
                signature=signatures
            )
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                signature=signatures
            )
        else:
            mlflow.log_artifact(model, artifact_path=model_name)
            
        logger.info(f"Model logged: {model_name}")
        
    def log_cnn_training(
        self,
        run_name: str,
        config: Dict,
        history: Dict,
        model_path: Path,
        dataset_version: str
    ):
        """Log CNN training results"""
        
        with self.start_run(run_name=run_name, tags={"model_type": "cnn", "dataset_version": dataset_version}):
            # Log config
            self.log_params({
                "architecture": config.get("architecture", "efficientnet-b3"),
                "embedding_dim": config.get("embedding_dim", 1536),
                "dropout": config.get("dropout", 0.3),
                "batch_size": config.get("batch_size", 32),
                "learning_rate": config.get("lr", 0.0001),
                "epochs": config.get("epochs", 50),
                "dataset_version": dataset_version
            })
            
            # Log metrics from each epoch
            for epoch, (train_loss, val_loss, val_f1) in enumerate(
                zip(history.get("train_loss", []), 
                    history.get("val_loss", []),
                    history.get("val_f1", []))
            ):
                self.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": val_f1
                }, step=epoch + 1)
                
            # Log final metrics
            self.log_metrics({
                "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0,
                "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
                "best_val_f1": max(history.get("val_f1", [0])),
                "total_epochs": len(history.get("train_loss", []))
            })
            
            # Log model artifact
            if model_path and model_path.exists():
                self.log_artifact(str(model_path), "model")
                
            # Log training history
            with open("training_history.json", "w") as f:
                json.dump(history, f)
            self.log_artifact("training_history.json")
            
    def log_fusion_training(
        self,
        run_name: str,
        config: Dict,
        results: Dict,
        model_path: Path,
        dataset_version: str,
        cv_scores: List[float] = None
    ):
        """Log XGBoost fusion model training results"""
        
        with self.start_run(run_name=run_name, tags={"model_type": "fusion", "dataset_version": dataset_version}):
            # Log parameters
            self.log_params({
                "n_estimators": config.get("n_estimators", 500),
                "max_depth": config.get("max_depth", 6),
                "learning_rate": config.get("learning_rate", 0.05),
                "subsample": config.get("subsample", 0.8),
                "colsample_bytree": config.get("colsample_bytree", 0.8),
                "pca_components": config.get("pca_components", 128),
                "dataset_version": dataset_version
            })
            
            # Log metrics
            metrics = {
                "mean_cv_score": results.get("mean_cv_score", 0),
                "best_score": results.get("best_score", 0),
                "train_accuracy": results.get("train_accuracy", 0),
                "val_accuracy": results.get("val_accuracy", 0),
                "profit_factor": results.get("profit_factor", 0),
                "sharpe_ratio": results.get("sharpe_ratio", 0),
                "max_drawdown": results.get("max_drawdown", 0),
                "win_rate": results.get("win_rate", 0)
            }
            self.log_metrics(metrics)
            
            # Log cross-validation scores
            if cv_scores:
                for i, score in enumerate(cv_scores):
                    self.log_metrics({f"cv_fold_{i+1}": score})
                    
            # Log feature importance (top 20)
            importance = results.get("feature_importance", {})
            for name, imp in list(importance.items())[:20]:
                self.log_metrics({f"feature_imp_{name[:50]}": imp})
                
            # Log model artifact
            if model_path and model_path.exists():
                self.log_artifact(str(model_path), "fusion_model")
                
    def get_best_runs(
        self,
        metric: str = "val_f1",
        max_results: int = 5,
        model_type: str = None
    ) -> List[Dict]:
        """Get best runs based on metric"""
        
        # Build filter string
        filter_string = ""
        if model_type:
            filter_string = f"tags.model_type = '{model_type}'"
            
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{metric} DESC"],
            max_results=max_results
        )
        
        results = []
        for run in runs:
            results.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000)
            })
            
        return results
        
    def compare_runs(self, run_ids: List[str]) -> Dict:
        """Compare multiple runs"""
        
        comparison = {
            "runs": [],
            "best_by_metric": {}
        }
        
        metrics_summary = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            
            run_data = {
                "run_id": run_id,
                "run_name": run.info.run_name,
                "metrics": run.data.metrics,
                "params": run.data.params
            }
            comparison["runs"].append(run_data)
            
            # Track best by each metric
            for metric, value in run.data.metrics.items():
                if metric not in metrics_summary:
                    metrics_summary[metric] = []
                metrics_summary[metric].append((value, run_id))
                
        # Find best for each metric
        for metric, values in metrics_summary.items():
            best_value, best_run = max(values, key=lambda x: x[0]) if "loss" not in metric else min(values, key=lambda x: x[0])
            comparison["best_by_metric"][metric] = {
                "run_id": best_run,
                "value": best_value
            }
            
        return comparison
        
    def get_run_artifacts(self, run_id: str) -> List[str]:
        """List all artifacts for a run"""
        
        artifacts = self.client.list_artifacts(run_id)
        return [a.path for a in artifacts]
        
    def download_artifact(self, run_id: str, artifact_path: str, output_path: Path):
        """Download artifact from run"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client.download_artifacts(run_id, artifact_path, dst_path=str(output_path.parent))
        logger.info(f"Downloaded {artifact_path} to {output_path}")
        
    def delete_run(self, run_id: str):
        """Delete a run (soft delete)"""
        self.client.delete_run(run_id)
        logger.info(f"Deleted run: {run_id}")
        
    def set_run_tag(self, run_id: str, key: str, value: str):
        """Set tag on a run"""
        self.client.set_tag(run_id, key, value)
        
    def transition_run_stage(self, run_id: str, stage: str):
        """Transition run to a different stage"""
        self.client.set_tag(run_id, "stage", stage)
        logger.info(f"Run {run_id} transitioned to {stage}")


def track_experiment(func):
    """Decorator to automatically track function execution as experiment"""
    
    def wrapper(*args, **kwargs):
        tracker = ExperimentTracker()
        
        run_name = f"{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with tracker.start_run(run_name=run_name):
            # Log function parameters
            params = {}
            for i, arg in enumerate(args):
                params[f"arg_{i}"] = str(arg)
            params.update({k: str(v) for k, v in kwargs.items()})
            tracker.log_params(params)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log result metrics if it's a dict
            if isinstance(result, dict):
                metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                if metrics:
                    tracker.log_metrics(metrics)
                    
            return result
            
    return wrapper
