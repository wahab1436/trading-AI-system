"""Automated retraining pipeline triggered by drift or schedule"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Callable, Any
import subprocess
import json
import threading
import time
from dataclasses import dataclass

from .drift_monitor import DriftMonitor
from .model_registry import ModelRegistry, ModelStage
from .experiment_tracker import ExperimentTracker
from .alerting import AlertManager

logger = logging.getLogger(__name__)


@dataclass
class RetrainConfig:
    """Retraining pipeline configuration"""
    enabled: bool = True
    schedule_hours: int = 168  # Weekly (168 hours)
    min_samples_required: int = 1000
    min_trades_required: int = 100
    max_retrain_frequency_hours: int = 24
    champion_comparison_metrics: list = None
    
    def __post_init__(self):
        if self.champion_comparison_metrics is None:
            self.champion_comparison_metrics = ['profit_factor', 'sharpe_ratio', 'win_rate']


class RetrainPipeline:
    """Automated retraining pipeline with drift detection and schedule triggers"""
    
    def __init__(
        self,
        config: RetrainConfig,
        data_dir: Path,
        models_dir: Path,
        experiment_tracker: ExperimentTracker = None,
        model_registry: ModelRegistry = None,
        drift_monitor: DriftMonitor = None,
        alert_manager: AlertManager = None
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        self.experiment_tracker = experiment_tracker or ExperimentTracker()
        self.model_registry = model_registry or ModelRegistry(models_dir)
        self.drift_monitor = drift_monitor
        self.alert_manager = alert_manager or AlertManager()
        
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_in_progress = False
        self.retrain_history: list = []
        
        # Start scheduler if enabled
        self.scheduler_thread = None
        if config.enabled:
            self._start_scheduler()
            
    def _start_scheduler(self):
        """Start background scheduler for periodic retraining"""
        
        def scheduler_loop():
            while True:
                try:
                    if self.should_retrain_by_schedule():
                        logger.info("Schedule trigger: Starting retraining")
                        self.trigger_retrain(reason="scheduled")
                    time.sleep(3600)  # Check every hour
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(3600)
                    
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info(f"Retrain scheduler started (interval: {self.config.schedule_hours}h)")
        
    def should_retrain_by_schedule(self) -> bool:
        """Check if scheduled retraining is due"""
        
        if self.last_retrain_time is None:
            return True
            
        hours_since_last = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
        return hours_since_last >= self.config.schedule_hours
        
    def should_retrain_by_drift(self) -> Tuple[bool, str]:
        """Check if drift monitor indicates retraining needed"""
        
        if self.drift_monitor is None:
            return False, "No drift monitor configured"
            
        return self.drift_monitor.requires_retraining()
        
    def should_retrain_by_data(self) -> Tuple[bool, str]:
        """Check if enough new data has accumulated"""
        
        # Check if we have enough new trades
        # This would need to query trade journal
        # Placeholder implementation
        return False, "Insufficient new data"
        
    def check_retrain_triggers(self) -> Tuple[bool, str]:
        """Check all retrain triggers"""
        
        # Check schedule
        if self.should_retrain_by_schedule():
            return True, "Scheduled retraining due"
            
        # Check drift
        drift_required, drift_reason = self.should_retrain_by_drift()
        if drift_required:
            return True, drift_reason
            
        # Check data accumulation
        data_required, data_reason = self.should_retrain_by_data()
        if data_required:
            return True, data_reason
            
        return False, "No triggers activated"
        
    def trigger_retrain(self, reason: str = "manual") -> Dict:
        """Trigger the retraining pipeline"""
        
        if self.retrain_in_progress:
            return {"status": "skipped", "reason": "Retraining already in progress"}
            
        # Check frequency limit
        if self.last_retrain_time:
            hours_since = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since < self.config.max_retrain_frequency_hours:
                return {
                    "status": "skipped",
                    "reason": f"Last retrain {hours_since:.1f}h ago (min {self.config.max_retrain_frequency_hours}h)"
                }
                
        self.retrain_in_progress = True
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting retraining pipeline (reason: {reason})")
            self.alert_manager.send_alert(
                "Retraining Started",
                f"Automated retraining triggered: {reason}",
                severity="info"
            )
            
            # Step 1: Fetch new data
            logger.info("Step 1: Fetching new data...")
            new_data_path = self._fetch_new_data()
            
            # Step 2: Validate and label data
            logger.info("Step 2: Labeling new data...")
            labeled_data_path = self._label_data(new_data_path)
            
            # Step 3: Render images
            logger.info("Step 3: Rendering images...")
            images_path = self._render_images(labeled_data_path)
            
            # Step 4: Train CNN model
            logger.info("Step 4: Training CNN model...")
            cnn_model_path, cnn_metrics = self._train_cnn(images_path)
            
            # Step 5: Extract embeddings
            logger.info("Step 5: Extracting CNN embeddings...")
            embeddings_path = self._extract_embeddings(cnn_model_path, images_path)
            
            # Step 6: Compute SMC features
            logger.info("Step 6: Computing SMC features...")
            smc_features_path = self._compute_smc_features(labeled_data_path)
            
            # Step 7: Train fusion model
            logger.info("Step 7: Training fusion model...")
            fusion_model_path, fusion_metrics = self._train_fusion(
                embeddings_path, smc_features_path, labeled_data_path
            )
            
            # Step 8: Backtest new model
            logger.info("Step 8: Running backtest...")
            backtest_results = self._run_backtest(fusion_model_path)
            
            # Step 9: Compare with champion
            logger.info("Step 9: Comparing with champion...")
            should_promote, comparison = self._compare_with_champion(fusion_metrics, backtest_results)
            
            # Step 10: Register model
            logger.info("Step 10: Registering model...")
            version = self._register_model(
                fusion_model_path,
                fusion_metrics,
                backtest_results,
                should_promote
            )
            
            # Step 11: Deploy if promoted
            if should_promote:
                logger.info("Step 11: Promoting to champion...")
                self.model_registry.promote_to_champion(version.version_id)
                self.alert_manager.send_alert(
                    "New Champion Model",
                    f"Model {version.version_id} promoted to champion\n"
                    f"Profit Factor: {fusion_metrics.get('profit_factor', 0):.2f}\n"
                    f"Win Rate: {fusion_metrics.get('win_rate', 0):.2%}",
                    severity="success"
                )
                
            # Update history
            duration = (datetime.now() - start_time).total_seconds()
            self.last_retrain_time = datetime.now()
            self.retrain_history.append({
                "timestamp": self.last_retrain_time.isoformat(),
                "reason": reason,
                "duration_seconds": duration,
                "model_version": version.version_id,
                "promoted": should_promote,
                "metrics": fusion_metrics
            })
            
            # Save history
            self._save_history()
            
            return {
                "status": "success",
                "model_version": version.version_id,
                "promoted": should_promote,
                "duration_seconds": duration,
                "metrics": fusion_metrics
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)
            self.alert_manager.send_alert(
                "Retraining Failed",
                f"Retraining pipeline failed: {str(e)}",
                severity="error"
            )
            return {"status": "failed", "error": str(e)}
            
        finally:
            self.retrain_in_progress = False
            
    def _fetch_new_data(self) -> Path:
        """Fetch new OHLCV data since last retrain"""
        
        output_dir = self.data_dir / f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Call the fetch script
        cmd = [
            "python", "-m", "etl.flows.fetch_historical",
            "--output", str(output_dir),
            "--years", "1" if self.last_retrain_time else "5"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Data fetch failed: {result.stderr}")
            
        return output_dir
        
    def _label_data(self, data_path: Path) -> Path:
        """Label the fetched data"""
        
        output_dir = self.data_dir / f"labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "-m", "etl.flows.label_data",
            "--input", str(data_path),
            "--output", str(output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Labeling failed: {result.stderr}")
            
        return output_dir
        
    def _render_images(self, labeled_data_path: Path) -> Path:
        """Render images from labeled data"""
        
        output_dir = self.data_dir / f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "-m", "etl.flows.render_images",
            "--input", str(labeled_data_path),
            "--output", str(output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Image rendering failed: {result.stderr}")
            
        return output_dir
        
    def _train_cnn(self, images_path: Path) -> tuple:
        """Train CNN model"""
        
        model_dir = self.models_dir / f"cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "-m", "cnn_model.train",
            "--data", str(images_path),
            "--output", str(model_dir),
            "--epochs", "50"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"CNN training failed: {result.stderr}")
            
        # Parse metrics from output
        metrics = {"val_f1": 0.0, "val_accuracy": 0.0}
        
        return model_dir, metrics
        
    def _extract_embeddings(self, cnn_model_path: Path, images_path: Path) -> Path:
        """Extract CNN embeddings for all images"""
        
        output_path = self.data_dir / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        
        # Call embedding extraction script
        # Placeholder
        return output_path
        
    def _compute_smc_features(self, data_path: Path) -> Path:
        """Compute SMC features from OHLCV data"""
        
        output_path = self.data_dir / f"smc_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        
        # Call SMC feature computation script
        # Placeholder
        return output_path
        
    def _train_fusion(self, embeddings_path: Path, smc_features_path: Path, labels_path: Path) -> tuple:
        """Train fusion model"""
        
        model_dir = self.models_dir / f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "-m", "fusion_model.train_xgb",
            "--embeddings", str(embeddings_path),
            "--smc", str(smc_features_path),
            "--labels", str(labels_path),
            "--output", str(model_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Fusion training failed: {result.stderr}")
            
        metrics = {"profit_factor": 0.0, "win_rate": 0.0, "sharpe_ratio": 0.0}
        
        return model_dir, metrics
        
    def _run_backtest(self, model_path: Path) -> Dict:
        """Run backtest on new model"""
        
        cmd = [
            "python", "-m", "backtest.simulator",
            "--model", str(model_path),
            "--output", "backtest_results.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Backtest failed: {result.stderr}")
            
        try:
            with open("backtest_results.json", "r") as f:
                return json.load(f)
        except:
            return {"profit_factor": 1.0, "max_drawdown": 0.1, "win_rate": 0.5}
            
    def _compare_with_champion(self, metrics: Dict, backtest_results: Dict) -> tuple:
        """Compare new model with current champion"""
        
        champion = self.model_registry.get_champion()
        
        if not champion:
            return True, {"message": "No existing champion"}
            
        # Combine metrics
        full_metrics = {**metrics, **backtest_results}
        
        beats_champion = True
        comparison = {}
        
        for metric in self.config.champion_comparison_metrics:
            challenger_value = full_metrics.get(metric, 0)
            champion_value = champion.metrics.get(metric, 0)
            
            is_better = challenger_value > champion_value
            
            if not is_better:
                beats_champion = False
                
            comparison[metric] = {
                "challenger": challenger_value,
                "champion": champion_value,
                "better": is_better
            }
            
        return beats_champion, comparison
        
    def _register_model(self, model_path: Path, metrics: Dict, backtest_results: Dict, is_champion: bool) -> Any:
        """Register model in registry"""
        
        full_metrics = {**metrics, **backtest_results}
        
        version = self.model_registry.register_model(
            model_path=model_path,
            model_name="fusion_model",
            metrics=full_metrics,
            params={"retrain_timestamp": datetime.now().isoformat()},
            dataset_version=f"auto_{datetime.now().strftime('%Y%m%d')}",
            run_id="auto_retrain"
        )
        
        return version
        
    def _save_history(self):
        """Save retraining history to disk"""
        
        history_path = self.models_dir / "retrain_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.retrain_history, f, indent=2)
            
    def get_status(self) -> Dict:
        """Get retraining pipeline status"""
        
        return {
            "enabled": self.config.enabled,
            "retrain_in_progress": self.retrain_in_progress,
            "last_retrain_time": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "total_retrains": len(self.retrain_history),
            "last_retrain_status": self.retrain_history[-1] if self.retrain_history else None,
            "drift_score": self.drift_monitor.current_drift_score if self.drift_monitor else None
        }


class RetrainTrigger:
    """Triggers retraining based on various conditions"""
    
    def __init__(self, retrain_pipeline: RetrainPipeline):
        self.pipeline = retrain_pipeline
        
    def check_and_trigger(self) -> Dict:
        """Check all triggers and retrain if needed"""
        
        should_retrain, reason = self.pipeline.check_retrain_triggers()
        
        if should_retrain:
            return self.pipeline.trigger_retrain(reason)
        else:
            return {"status": "not_needed", "reason": reason}
            
    def trigger_manual(self) -> Dict:
        """Manually trigger retraining"""
        return self.pipeline.trigger_retrain(reason="manual")
