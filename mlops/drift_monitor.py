"""Concept drift detection using PSI, KS-test, and performance monitoring"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Drift alert data structure"""
    timestamp: datetime
    alert_type: str  # 'psi', 'ks_test', 'confidence', 'performance'
    severity: str  # 'warning', 'critical'
    feature_name: str
    value: float
    threshold: float
    message: str


class PSICalculator:
    """Population Stability Index calculator for drift detection"""
    
    @staticmethod
    def calculate(expected_dist: np.ndarray, actual_dist: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Population Stability Index
        
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        
        Rules of thumb:
        PSI < 0.1: No significant drift
        0.1 <= PSI < 0.2: Moderate drift (warning)
        PSI >= 0.2: Significant drift (retrain needed)
        """
        
        # Create bins
        min_val = min(expected_dist.min(), actual_dist.min())
        max_val = max(expected_dist.max(), actual_dist.max())
        
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate distributions
        expected_hist, _ = np.histogram(expected_dist, bins=bins)
        actual_hist, _ = np.histogram(actual_dist, bins=bins)
        
        # Add small epsilon to avoid division by zero
        expected_pct = (expected_hist + 1e-10) / len(expected_dist)
        actual_pct = (actual_hist + 1e-10) / len(actual_dist)
        
        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
        
    @staticmethod
    def interpret(psi: float) -> Tuple[str, str]:
        """Interpret PSI value"""
        if psi < 0.1:
            return "stable", "No significant drift detected"
        elif psi < 0.2:
            return "warning", "Moderate drift detected - monitor closely"
        else:
            return "critical", "Significant drift detected - retraining recommended"


class DriftMonitor:
    """Monitors multiple drift detection methods"""
    
    def __init__(
        self,
        window_size: int = 1000,
        psi_threshold_warning: float = 0.1,
        psi_threshold_critical: float = 0.2,
        ks_threshold: float = 0.05,
        confidence_drop_threshold: float = 0.1,
        performance_drop_threshold: float = 0.1
    ):
        self.window_size = window_size
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_critical = psi_threshold_critical
        self.ks_threshold = ks_threshold
        self.confidence_drop_threshold = confidence_drop_threshold
        self.performance_drop_threshold = performance_drop_threshold
        
        # Rolling windows for data
        self.feature_buffers: Dict[str, deque] = {}
        self.prediction_buffers: deque = deque(maxlen=window_size)
        self.confidence_buffers: deque = deque(maxlen=window_size)
        self.outcome_buffers: deque = deque(maxlen=window_size)
        
        # Baseline distributions (from training)
        self.baseline_distributions: Dict[str, np.ndarray] = {}
        self.baseline_win_rate: float = None
        self.baseline_avg_confidence: float = None
        
        # Alert history
        self.alerts: List[DriftAlert] = []
        
        # Current drift status
        self.current_drift_score: float = 0.0
        self.last_drift_check: datetime = None
        
    def set_baseline(
        self,
        feature_distributions: Dict[str, np.ndarray],
        win_rate: float,
        avg_confidence: float
    ):
        """Set baseline distributions from training data"""
        self.baseline_distributions = feature_distributions
        self.baseline_win_rate = win_rate
        self.baseline_avg_confidence = avg_confidence
        logger.info(f"Baseline set: Win Rate={win_rate:.2%}, Avg Confidence={avg_confidence:.2%}")
        
    def update(
        self,
        features: Dict[str, float],
        prediction: int,
        confidence: float,
        outcome: float = None
    ):
        """Update buffers with new data point"""
        
        # Update feature buffers
        for name, value in features.items():
            if name not in self.feature_buffers:
                self.feature_buffers[name] = deque(maxlen=self.window_size)
            self.feature_buffers[name].append(value)
            
        # Update prediction buffers
        self.prediction_buffers.append(prediction)
        self.confidence_buffers.append(confidence)
        
        if outcome is not None:
            self.outcome_buffers.append(outcome)
            
    def check_psi_drift(self) -> List[DriftAlert]:
        """Check PSI drift for all features"""
        
        alerts = []
        
        for feature_name, baseline_dist in self.baseline_distributions.items():
            if feature_name in self.feature_buffers and len(self.feature_buffers[feature_name]) >= 100:
                current_dist = np.array(list(self.feature_buffers[feature_name]))
                
                psi = PSICalculator.calculate(baseline_dist, current_dist)
                severity, message = PSICalculator.interpret(psi)
                
                if severity == "warning" and psi >= self.psi_threshold_warning:
                    alerts.append(DriftAlert(
                        timestamp=datetime.now(),
                        alert_type="psi",
                        severity="warning",
                        feature_name=feature_name,
                        value=psi,
                        threshold=self.psi_threshold_warning,
                        message=message
                    ))
                elif severity == "critical" and psi >= self.psi_threshold_critical:
                    alerts.append(DriftAlert(
                        timestamp=datetime.now(),
                        alert_type="psi",
                        severity="critical",
                        feature_name=feature_name,
                        value=psi,
                        threshold=self.psi_threshold_critical,
                        message=message
                    ))
                    
        return alerts
        
    def check_ks_test_drift(self) -> List[DriftAlert]:
        """Check Kolmogorov-Smirnov test for distribution drift"""
        
        alerts = []
        
        for feature_name, baseline_dist in self.baseline_distributions.items():
            if feature_name in self.feature_buffers and len(self.feature_buffers[feature_name]) >= 100:
                current_dist = np.array(list(self.feature_buffers[feature_name]))
                
                # Perform KS test
                ks_statistic, p_value = stats.ks_2samp(baseline_dist, current_dist)
                
                # If p-value < threshold, distributions are significantly different
                if p_value < self.ks_threshold:
                    severity = "warning" if p_value < 0.01 else "critical"
                    alerts.append(DriftAlert(
                        timestamp=datetime.now(),
                        alert_type="ks_test",
                        severity=severity,
                        feature_name=feature_name,
                        value=p_value,
                        threshold=self.ks_threshold,
                        message=f"KS test indicates distribution shift (p={p_value:.4f})"
                    ))
                    
        return alerts
        
    def check_confidence_drift(self) -> List[DriftAlert]:
        """Check if average confidence has dropped significantly"""
        
        if len(self.confidence_buffers) < 100 or self.baseline_avg_confidence is None:
            return []
            
        current_avg_confidence = np.mean(self.confidence_buffers)
        confidence_drop = (self.baseline_avg_confidence - current_avg_confidence) / self.baseline_avg_confidence
        
        if confidence_drop > self.confidence_drop_threshold:
            severity = "warning" if confidence_drop < 0.2 else "critical"
            return [DriftAlert(
                timestamp=datetime.now(),
                alert_type="confidence",
                severity=severity,
                feature_name="avg_confidence",
                value=confidence_drop,
                threshold=self.confidence_drop_threshold,
                message=f"Average confidence dropped by {confidence_drop:.1%}"
            )]
            
        return []
        
    def check_performance_drift(self) -> List[DriftAlert]:
        """Check if live performance has degraded"""
        
        if len(self.outcome_buffers) < 50 or self.baseline_win_rate is None:
            return []
            
        # Calculate rolling win rate
        outcomes = np.array(self.outcome_buffers)
        wins = np.sum(outcomes > 0)
        current_win_rate = wins / len(outcomes) if len(outcomes) > 0 else 0
        
        performance_drop = (self.baseline_win_rate - current_win_rate) / self.baseline_win_rate if self.baseline_win_rate > 0 else 0
        
        if performance_drop > self.performance_drop_threshold:
            severity = "warning" if performance_drop < 0.15 else "critical"
            return [DriftAlert(
                timestamp=datetime.now(),
                alert_type="performance",
                severity=severity,
                feature_name="win_rate",
                value=performance_drop,
                threshold=self.performance_drop_threshold,
                message=f"Win rate dropped from {self.baseline_win_rate:.1%} to {current_win_rate:.1%}"
            )]
            
        return []
        
    def check_all(self) -> List[DriftAlert]:
        """Run all drift detection methods"""
        
        all_alerts = []
        
        all_alerts.extend(self.check_psi_drift())
        all_alerts.extend(self.check_ks_test_drift())
        all_alerts.extend(self.check_confidence_drift())
        all_alerts.extend(self.check_performance_drift())
        
        # Store alerts
        self.alerts.extend(all_alerts)
        
        # Calculate overall drift score
        if all_alerts:
            critical_count = sum(1 for a in all_alerts if a.severity == "critical")
            warning_count = sum(1 for a in all_alerts if a.severity == "warning")
            self.current_drift_score = min(1.0, (critical_count * 0.3 + warning_count * 0.1))
            
        self.last_drift_check = datetime.now()
        
        # Log alerts
        for alert in all_alerts:
            logger.warning(f"Drift Alert [{alert.severity}]: {alert.message}")
            
        return all_alerts
        
    def requires_retraining(self) -> Tuple[bool, str]:
        """Check if drift severity requires retraining"""
        
        # Check recent alerts (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_alerts = [a for a in self.alerts if a.timestamp > one_hour_ago]
        
        critical_alerts = [a for a in recent_alerts if a.severity == "critical"]
        warning_alerts = [a for a in recent_alerts if a.severity == "warning"]
        
        # Retrain if:
        # 1. Any critical drift in PSI or performance
        # 2
