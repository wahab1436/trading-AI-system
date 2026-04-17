"""Model registry for champion/challenger tracking and versioning"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from enum import Enum
import pickle
import shutil
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages"""
    CHALLENGER = "challenger"
    CHAMPION = "champion"
    ARCHIVED = "archived"
    RETIRED = "retired"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_name: str
    stage: ModelStage
    created_at: datetime
    metrics: Dict[str, float]
    params: Dict[str, Any]
    dataset_version: str
    run_id: str
    description: str = ""
    parent_version_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['stage'] = self.stage.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        data['stage'] = ModelStage(data['stage'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelRegistry:
    """Registry for tracking model versions and champion/challenger"""
    
    def __init__(self, registry_path: Path = Path("models/registry")):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.registry_path / "metadata.json"
        self._load_registry()
        
    def _load_registry(self):
        """Load registry from disk"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.versions = {
                    k: ModelVersion.from_dict(v) 
                    for k, v in data.get('versions', {}).items()
                }
                self.champion_id = data.get('champion_id')
                self.challenger_id = data.get('challenger_id')
        else:
            self.versions = {}
            self.champion_id = None
            self.challenger_id = None
            
    def _save_registry(self):
        """Save registry to disk"""
        data = {
            'versions': {k: v.to_dict() for k, v in self.versions.items()},
            'champion_id': self.champion_id,
            'challenger_id': self.challenger_id
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def register_model(
        self,
        model_path: Path,
        model_name: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        dataset_version: str,
        run_id: str,
        description: str = "",
        parent_version_id: Optional[str] = None
    ) -> ModelVersion:
        """Register a new model version"""
        
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_name}_{timestamp}"
        
        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            stage=ModelStage.CHALLENGER,
            created_at=datetime.now(),
            metrics=metrics,
            params=params,
            dataset_version=dataset_version,
            run_id=run_id,
            description=description,
            parent_version_id=parent_version_id
        )
        
        # Copy model to registry
        version_path = self.models_path / version_id
        version_path.mkdir(exist_ok=True)
        
        if model_path.is_file():
            shutil.copy2(model_path, version_path / model_path.name)
        elif model_path.is_dir():
            shutil.copytree(model_path, version_path / "model", dirs_exist_ok=True)
            
        # Save metadata
        with open(version_path / "metadata.json", 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
            
        # Add to registry
        self.versions[version_id] = version
        self.challenger_id = version_id
        
        self._save_registry()
        
        logger.info(f"Registered model: {version_id}")
        
        # Auto-evaluate challenger
        if self.champion_id:
            self.evaluate_challenger(version_id)
            
        return version
        
    def evaluate_challenger(
        self,
        challenger_id: str,
        metrics_to_beat: List[str] = None
    ) -> Tuple[bool, Dict]:
        """
        Evaluate challenger against champion
        
        Returns:
            (should_promote, comparison_results)
        """
        
        if challenger_id not in self.versions:
            raise ValueError(f"Challenger {challenger_id} not found")
            
        if not self.champion_id:
            # No champion, promote challenger
            self.promote_to_champion(challenger_id)
            return True, {"message": "No existing champion, promoted"}
            
        challenger = self.versions[challenger_id]
        champion = self.versions[self.champion_id]
        
        if metrics_to_beat is None:
            # Default metrics: f1_score, profit_factor, sharpe_ratio
            metrics_to_beat = ['f1_score', 'profit_factor', 'sharpe_ratio']
            
        comparison = {}
        beats_champion = True
        
        for metric in metrics_to_beat:
            challenger_value = challenger.metrics.get(metric, 0)
            champion_value = champion.metrics.get(metric, 0)
            
            # For loss metrics, lower is better
            is_loss_metric = 'loss' in metric or 'drawdown' in metric
            
            if is_loss_metric:
                beats = challenger_value < champion_value
            else:
                beats = challenger_value > champion_value
                
            comparison[metric] = {
                'challenger': challenger_value,
                'champion': champion_value,
                'beats': beats,
                'improvement': (challenger_value - champion_value) / champion_value if champion_value != 0 else 0
            }
            
            if not beats:
                beats_champion = False
                
        # Log comparison
        logger.info(f"Challenger {challenger_id} vs Champion {self.champion_id}")
        for metric, data in comparison.items():
            logger.info(f"  {metric}: {data['challenger']:.4f} vs {data['champion']:.4f} "
                       f"({'+' if data['beats'] else ''}{data['improvement']:.2%})")
                       
        # Promote if beats champion on all metrics
        if beats_champion:
            self.promote_to_champion(challenger_id)
            return True, comparison
        else:
            return False, comparison
            
    def promote_to_champion(self, version_id: str):
        """Promote a model version to champion"""
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
            
        # Demote current champion to archived
        if self.champion_id:
            old_champion = self.versions[self.champion_id]
            old_champion.stage = ModelStage.ARCHIVED
            self._update_model_metadata(old_champion)
            
        # Promote new champion
        new_champion = self.versions[version_id]
        new_champion.stage = ModelStage.CHAMPION
        self.champion_id = version_id
        self.challenger_id = None
        
        self._update_model_metadata(new_champion)
        self._save_registry()
        
        logger.info(f"Promoted {version_id} to CHAMPION")
        
    def get_champion(self) -> Optional[ModelVersion]:
        """Get current champion model"""
        if self.champion_id:
            return self.versions.get(self.champion_id)
        return None
        
    def get_challenger(self) -> Optional[ModelVersion]:
        """Get current challenger model"""
        if self.challenger_id:
            return self.versions.get(self.challenger_id)
        return None
        
    def get_model_path(self, version_id: str) -> Path:
        """Get filesystem path for a model version"""
        return self.models_path / version_id
        
    def load_champion_model(self, model_type: str = "fusion"):
        """Load the champion model for inference"""
        
        champion = self.get_champion()
        if not champion:
            logger.warning("No champion model found")
            return None
            
        model_path = self.get_model_path(champion.version_id)
        
        if model_type == "fusion":
            from fusion_model.train_xgb import FusionModel
            model = FusionModel()
            model.load(model_path / "model")
            return model
        elif model_type == "cnn":
            import torch
            from cnn_model.model import ChartCNN
            model = ChartCNN()
            model.load_state_dict(torch.load(model_path / "model.pt"))
            model.eval()
            return model
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
            
    def rollback(self) -> Optional[ModelVersion]:
        """Rollback to previous champion"""
        
        # Find previous champion (most recent archived)
        archived = [
            v for v in self.versions.values() 
            if v.stage == ModelStage.ARCHIVED
        ]
        
        if not archived:
            logger.warning("No archived models to rollback to")
            return None
            
        # Sort by creation time
        archived.sort(key=lambda x: x.created_at, reverse=True)
        previous_champion = archived[0]
        
        # Promote to champion
        self.promote_to_champion(previous_champion.version_id)
        
        logger.info(f"Rolled back to {previous_champion.version_id}")
        return previous_champion
        
    def _update_model_metadata(self, version: ModelVersion):
        """Update model metadata file"""
        model_path = self.models_path / version.version_id
        with open(model_path / "metadata.json", 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
            
    def list_models(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List all models, optionally filtered by stage"""
        
        models = list(self.versions.values())
        
        if stage:
            models = [m for m in models if m.stage == stage]
            
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models
        
    def get_model_performance_history(self, metric: str = "profit_factor") -> List[Dict]:
        """Get performance history over model versions"""
        
        history = []
        
        for version in sorted(self.versions.values(), key=lambda x: x.created_at):
            if metric in version.metrics:
                history.append({
                    'version_id': version.version_id,
                    'created_at': version.created_at.isoformat(),
                    'metric_value': version.metrics[metric],
                    'stage': version.stage.value
                })
                
        return history
        
    def archive_old_models(self, keep_last_n: int = 5):
        """Archive old models to keep registry clean"""
        
        all_models = list(self.versions.values())
        all_models.sort(key=lambda x: x.created_at, reverse=True)
        
        # Keep champion and last N models
        to_archive = all_models[keep_last_n:]
        
        for model in to_archive:
            if model.version_id != self.champion_id:
                model.stage = ModelStage.ARCHIVED
                self._update_model_metadata(model)
                
        self._save_registry()
        logger.info(f"Archived {len(to_archive)} old models")
