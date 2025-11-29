"""
RSCS-Q Booklet 7: Reflective Matrix Engine (RME)
=================================================

Core engine for reflective cognition:
- Matrix Drift Scanning
- Emergent Validity Index (EVI)
- Matrix Discovery Score (MDS)
- Capsule Drift Forecast Graphs
- Cross-capsule coordination API

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import hashlib
import json

from capsule_lineage import CapsuleFingerprint, CapsuleLineageTree, LineageNode


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_HISTORY_SIZE = 100
DEFAULT_FORECAST_HORIZON = 10
DEFAULT_ANOMALY_THRESHOLD = 0.7


# =============================================================================
# DRIFT TYPES (from Booklet 6 Appendix D, expanded)
# =============================================================================

class DriftType(Enum):
    """Classification of drift types"""
    RUBRIC = "rubric"           # Evaluation criteria drift
    EXECUTION = "execution"     # Behavioral drift under same conditions
    OUTCOME = "outcome"         # Result space divergence
    SIGNAL = "signal"           # Input distribution drift
    META = "meta"               # Self-comparator threshold drift


class DriftSeverity(Enum):
    """Severity levels for drift"""
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


# =============================================================================
# DRIFT RECORD
# =============================================================================

@dataclass
class DriftRecord:
    """
    A single drift observation.
    """
    capsule_id: str
    drift_type: DriftType
    drift_score: float
    severity: DriftSeverity
    baseline_hash: str
    current_hash: str
    delta_vector: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'drift_type': self.drift_type.value,
            'drift_score': self.drift_score,
            'severity': self.severity.value,
            'baseline_hash': self.baseline_hash,
            'current_hash': self.current_hash,
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# EMERGENT VALIDITY INDEX (EVI)
# =============================================================================

@dataclass
class EVIResult:
    """
    Emergent Validity Index computation result.
    
    EVI measures how well a capsule's behavior aligns with emergent
    patterns across the swarm/lineage.
    
    EVI = (coherence * stability * lineage_fidelity) ^ (1/3)
    
    Where:
    - coherence: Agreement with peer capsules
    - stability: Temporal consistency
    - lineage_fidelity: Alignment with ancestral patterns
    """
    capsule_id: str
    evi_score: float
    coherence: float
    stability: float
    lineage_fidelity: float
    confidence: float
    sample_size: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_valid(self, threshold: float = 0.5) -> bool:
        """Check if EVI indicates valid emergent behavior"""
        return self.evi_score >= threshold and self.confidence >= 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'evi_score': self.evi_score,
            'coherence': self.coherence,
            'stability': self.stability,
            'lineage_fidelity': self.lineage_fidelity,
            'confidence': self.confidence,
            'sample_size': self.sample_size,
            'is_valid': self.is_valid(),
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# MATRIX DISCOVERY SCORE (MDS)
# =============================================================================

@dataclass
class MDSResult:
    """
    Matrix Discovery Score computation result.
    
    MDS measures the novelty and significance of patterns discovered
    by a capsule relative to known patterns.
    
    MDS = novelty_score * significance_weight * confirmation_ratio
    
    Where:
    - novelty_score: Distance from known pattern clusters
    - significance_weight: Impact on system understanding
    - confirmation_ratio: How many peers confirm the pattern
    """
    capsule_id: str
    mds_score: float
    novelty_score: float
    significance_weight: float
    confirmation_ratio: float
    pattern_id: Optional[str] = None
    pattern_class: Optional[str] = None
    confirming_capsules: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_significant(self, threshold: float = 0.3) -> bool:
        """Check if discovery is significant"""
        return self.mds_score >= threshold and self.confirmation_ratio >= 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'mds_score': self.mds_score,
            'novelty_score': self.novelty_score,
            'significance_weight': self.significance_weight,
            'confirmation_ratio': self.confirmation_ratio,
            'pattern_id': self.pattern_id,
            'is_significant': self.is_significant(),
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# DRIFT FORECAST
# =============================================================================

@dataclass
class DriftForecast:
    """
    Predicted drift trajectory for a capsule.
    """
    capsule_id: str
    current_drift: float
    predicted_drift: List[float]  # Future drift values
    horizon: int  # Steps ahead
    trend: str  # "increasing", "decreasing", "stable"
    confidence: float
    breach_step: Optional[int] = None  # Step when threshold breached
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def will_breach(self, threshold: float = 0.7) -> bool:
        """Check if drift will breach threshold"""
        return any(d >= threshold for d in self.predicted_drift)
    
    def steps_to_breach(self, threshold: float = 0.7) -> Optional[int]:
        """Get number of steps until threshold breach"""
        for i, d in enumerate(self.predicted_drift):
            if d >= threshold:
                return i + 1
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'current_drift': self.current_drift,
            'predicted_drift': self.predicted_drift,
            'horizon': self.horizon,
            'trend': self.trend,
            'confidence': self.confidence,
            'will_breach': self.will_breach(),
            'steps_to_breach': self.steps_to_breach(),
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# REFLECTIVE MATRIX ENGINE
# =============================================================================

class ReflectiveMatrixEngine:
    """
    Core engine for reflective cognition and drift management.
    
    Provides:
    - Matrix drift scanning across capsules
    - EVI computation for emergent validity
    - MDS computation for pattern discovery
    - Drift forecasting and trend analysis
    - Cross-capsule coordination API
    """
    
    def __init__(
        self,
        engine_id: str,
        history_size: int = DEFAULT_HISTORY_SIZE,
        forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    ):
        self.engine_id = engine_id
        self.history_size = history_size
        self.forecast_horizon = forecast_horizon
        
        # Connected components
        self.lineage_tree: Optional[CapsuleLineageTree] = None
        
        # History buffers
        self.drift_history: Dict[str, deque] = {}  # capsule_id -> drift records
        self.evi_history: Dict[str, deque] = {}
        self.mds_history: Dict[str, deque] = {}
        
        # Pattern library
        self.known_patterns: Dict[str, np.ndarray] = {}
        self.pattern_classes: Dict[str, List[str]] = {}  # class -> pattern_ids
        
        # Baselines
        self.baselines: Dict[str, CapsuleFingerprint] = {}
        
        # Statistics
        self.total_scans = 0
        self.total_anomalies = 0
        self.total_discoveries = 0
        self.created_at = datetime.utcnow()
    
    def connect_lineage_tree(self, tree: CapsuleLineageTree) -> None:
        """Connect to a lineage tree"""
        self.lineage_tree = tree
    
    def set_baseline(self, capsule_id: str, fingerprint: CapsuleFingerprint) -> None:
        """Set baseline fingerprint for a capsule"""
        self.baselines[capsule_id] = fingerprint
    
    # -------------------------------------------------------------------------
    # DRIFT SCANNING
    # -------------------------------------------------------------------------
    
    def scan_drift(
        self,
        capsule_id: str,
        current_fingerprint: CapsuleFingerprint,
        drift_type: DriftType = DriftType.EXECUTION
    ) -> DriftRecord:
        """
        Scan for drift between current state and baseline.
        
        Args:
            capsule_id: Target capsule
            current_fingerprint: Current capsule state
            drift_type: Type of drift to measure
            
        Returns:
            DriftRecord with measurement
        """
        baseline = self.baselines.get(capsule_id)
        
        if baseline is None:
            # No baseline - establish one
            self.set_baseline(capsule_id, current_fingerprint)
            baseline = current_fingerprint
        
        # Compute drift score
        similarity = baseline.similarity(current_fingerprint)
        drift_score = 1.0 - similarity
        
        # Compute delta vector
        delta_vector = current_fingerprint.vector - baseline.vector
        
        # Determine severity
        if drift_score < 0.1:
            severity = DriftSeverity.NONE
        elif drift_score < 0.3:
            severity = DriftSeverity.LOW
        elif drift_score < 0.5:
            severity = DriftSeverity.MODERATE
        elif drift_score < 0.7:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL
        
        record = DriftRecord(
            capsule_id=capsule_id,
            drift_type=drift_type,
            drift_score=drift_score,
            severity=severity,
            baseline_hash=baseline.hash(),
            current_hash=current_fingerprint.hash(),
            delta_vector=delta_vector
        )
        
        # Store in history
        if capsule_id not in self.drift_history:
            self.drift_history[capsule_id] = deque(maxlen=self.history_size)
        self.drift_history[capsule_id].append(record)
        
        self.total_scans += 1
        if severity.value >= DriftSeverity.HIGH.value:
            self.total_anomalies += 1
        
        return record
    
    def scan_all_drift_types(
        self,
        capsule_id: str,
        current_fingerprint: CapsuleFingerprint
    ) -> Dict[DriftType, DriftRecord]:
        """Scan all drift types for a capsule"""
        results = {}
        for drift_type in DriftType:
            results[drift_type] = self.scan_drift(
                capsule_id, current_fingerprint, drift_type
            )
        return results
    
    def get_drift_trend(self, capsule_id: str, window: int = 10) -> str:
        """Get drift trend for a capsule"""
        history = self.drift_history.get(capsule_id)
        if not history or len(history) < 2:
            return "unknown"
        
        recent = list(history)[-window:]
        scores = [r.drift_score for r in recent]
        
        if len(scores) < 2:
            return "stable"
        
        # Simple trend detection
        first_half = np.mean(scores[:len(scores)//2])
        second_half = np.mean(scores[len(scores)//2:])
        
        diff = second_half - first_half
        if diff > 0.05:
            return "increasing"
        elif diff < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    # -------------------------------------------------------------------------
    # EMERGENT VALIDITY INDEX (EVI)
    # -------------------------------------------------------------------------
    
    def compute_evi(
        self,
        capsule_id: str,
        current_fingerprint: CapsuleFingerprint,
        peer_fingerprints: List[CapsuleFingerprint]
    ) -> EVIResult:
        """
        Compute Emergent Validity Index for a capsule.
        
        EVI = (coherence * stability * lineage_fidelity) ^ (1/3)
        
        Args:
            capsule_id: Target capsule
            current_fingerprint: Current state
            peer_fingerprints: Fingerprints of peer capsules
            
        Returns:
            EVIResult with computed metrics
        """
        # Coherence: Agreement with peers (use absolute similarity for stability)
        if peer_fingerprints:
            similarities = [abs(current_fingerprint.similarity(p)) for p in peer_fingerprints]
            coherence = float(np.mean(similarities))
        else:
            coherence = 0.5  # Neutral if no peers
        
        # Stability: Temporal consistency
        history = self.drift_history.get(capsule_id)
        if history and len(history) >= 2:
            recent_drifts = [r.drift_score for r in list(history)[-10:]]
            std_dev = float(np.std(recent_drifts))
            stability = max(0.001, 1.0 - std_dev)
            stability = min(1.0, stability)
        else:
            stability = 0.5  # Neutral if no history
        
        # Lineage fidelity: Alignment with ancestors
        lineage_fidelity = 0.5  # Default
        if self.lineage_tree:
            ancestors = self.lineage_tree.get_ancestors(capsule_id)
            if ancestors:
                ancestor_sims = [abs(current_fingerprint.similarity(a.fingerprint)) 
                               for a in ancestors[:3]]  # Check up to 3 ancestors
                lineage_fidelity = float(np.mean(ancestor_sims))
        
        # Compute EVI (geometric mean) - ensure all values positive
        coherence = max(0.001, coherence)
        stability = max(0.001, stability)
        lineage_fidelity = max(0.001, lineage_fidelity)
        evi_score = float((coherence * stability * lineage_fidelity) ** (1/3))
        
        # Confidence based on sample size
        sample_size = len(peer_fingerprints) + (len(history) if history else 0)
        confidence = min(1.0, sample_size / 20)
        
        result = EVIResult(
            capsule_id=capsule_id,
            evi_score=evi_score,
            coherence=coherence,
            stability=stability,
            lineage_fidelity=lineage_fidelity,
            confidence=confidence,
            sample_size=sample_size
        )
        
        # Store in history
        if capsule_id not in self.evi_history:
            self.evi_history[capsule_id] = deque(maxlen=self.history_size)
        self.evi_history[capsule_id].append(result)
        
        return result
    
    # -------------------------------------------------------------------------
    # MATRIX DISCOVERY SCORE (MDS)
    # -------------------------------------------------------------------------
    
    def compute_mds(
        self,
        capsule_id: str,
        pattern_vector: np.ndarray,
        confirming_capsules: List[str] = None
    ) -> MDSResult:
        """
        Compute Matrix Discovery Score for a pattern.
        
        MDS = novelty_score * significance_weight * confirmation_ratio
        
        Args:
            capsule_id: Discovering capsule
            pattern_vector: The discovered pattern
            confirming_capsules: IDs of capsules confirming pattern
            
        Returns:
            MDSResult with computed metrics
        """
        confirming_capsules = confirming_capsules or []
        
        # Novelty: Distance from known patterns
        if self.known_patterns:
            distances = [np.linalg.norm(pattern_vector - p) 
                        for p in self.known_patterns.values()]
            min_distance = min(distances)
            novelty_score = float(np.tanh(min_distance))  # Normalize to [0,1]
        else:
            novelty_score = 1.0  # Completely novel if no patterns known
        
        # Significance: Based on vector magnitude and structure
        magnitude = np.linalg.norm(pattern_vector)
        # Use variance as a proxy for information content
        variance = np.var(pattern_vector)
        significance_weight = float(np.tanh(magnitude * variance))
        
        # Confirmation ratio
        total_capsules = len(self.baselines) if self.baselines else 1
        confirmation_ratio = len(confirming_capsules) / max(1, total_capsules - 1)
        
        # Compute MDS - ensure all components are positive
        novelty_score = max(0.0, novelty_score)
        significance_weight = max(0.0, significance_weight)
        confirmation_ratio = max(0.1, confirmation_ratio)  # Minimum 10%
        
        mds_score = novelty_score * significance_weight * confirmation_ratio
        
        # Generate pattern ID
        pattern_hash = hashlib.sha256(pattern_vector.tobytes()).hexdigest()[:8]
        pattern_id = f"PAT-{pattern_hash}"
        
        # Classify pattern
        pattern_class = self._classify_pattern(pattern_vector)
        
        result = MDSResult(
            capsule_id=capsule_id,
            mds_score=mds_score,
            novelty_score=novelty_score,
            significance_weight=significance_weight,
            confirmation_ratio=confirmation_ratio,
            pattern_id=pattern_id,
            pattern_class=pattern_class,
            confirming_capsules=confirming_capsules
        )
        
        # Store pattern if significant
        if result.is_significant():
            self.known_patterns[pattern_id] = pattern_vector
            if pattern_class not in self.pattern_classes:
                self.pattern_classes[pattern_class] = []
            self.pattern_classes[pattern_class].append(pattern_id)
            self.total_discoveries += 1
        
        # Store in history
        if capsule_id not in self.mds_history:
            self.mds_history[capsule_id] = deque(maxlen=self.history_size)
        self.mds_history[capsule_id].append(result)
        
        return result
    
    def _classify_pattern(self, pattern_vector: np.ndarray) -> str:
        """Classify a pattern based on its structure"""
        # Simple classification based on vector properties
        mean = np.mean(pattern_vector)
        std = np.std(pattern_vector)
        max_val = np.max(np.abs(pattern_vector))
        
        if std < 0.1:
            return "uniform"
        elif max_val > 2 * std:
            return "peaked"
        elif mean > 0.5:
            return "positive_bias"
        elif mean < -0.5:
            return "negative_bias"
        else:
            return "mixed"
    
    # -------------------------------------------------------------------------
    # DRIFT FORECASTING
    # -------------------------------------------------------------------------
    
    def forecast_drift(
        self,
        capsule_id: str,
        horizon: int = None
    ) -> DriftForecast:
        """
        Forecast future drift for a capsule.
        
        Uses simple linear extrapolation with dampening.
        
        Args:
            capsule_id: Target capsule
            horizon: Steps to forecast (default: self.forecast_horizon)
            
        Returns:
            DriftForecast with predictions
        """
        horizon = horizon or self.forecast_horizon
        history = self.drift_history.get(capsule_id)
        
        if not history or len(history) < 3:
            # Insufficient history - return neutral forecast
            return DriftForecast(
                capsule_id=capsule_id,
                current_drift=0.0,
                predicted_drift=[0.0] * horizon,
                horizon=horizon,
                trend="unknown",
                confidence=0.0
            )
        
        # Extract recent drift scores
        recent = list(history)[-20:]
        scores = np.array([r.drift_score for r in recent])
        current_drift = float(scores[-1])
        
        # Fit simple linear trend
        x = np.arange(len(scores))
        coeffs = np.polyfit(x, scores, 1)
        slope = coeffs[0]
        
        # Predict with dampening (trend decays over time)
        predictions = []
        for i in range(horizon):
            dampen = 0.9 ** i
            pred = current_drift + slope * (i + 1) * dampen
            pred = max(0, min(1, pred))  # Clamp to [0, 1]
            predictions.append(float(pred))
        
        # Determine trend
        if slope > 0.01:
            trend = "increasing"
        elif slope < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Confidence based on fit quality and sample size
        residuals = scores - np.polyval(coeffs, x)
        r_squared = 1 - np.var(residuals) / np.var(scores) if np.var(scores) > 0 else 0
        confidence = float(max(0, r_squared) * min(1, len(scores) / 10))
        
        # Check for threshold breach
        breach_step = None
        for i, pred in enumerate(predictions):
            if pred >= DEFAULT_ANOMALY_THRESHOLD:
                breach_step = i + 1
                break
        
        return DriftForecast(
            capsule_id=capsule_id,
            current_drift=current_drift,
            predicted_drift=predictions,
            horizon=horizon,
            trend=trend,
            confidence=confidence,
            breach_step=breach_step
        )
    
    # -------------------------------------------------------------------------
    # CROSS-CAPSULE COORDINATION API
    # -------------------------------------------------------------------------
    
    def compare_capsules(
        self,
        capsule_a_id: str,
        capsule_b_id: str
    ) -> Dict[str, Any]:
        """Compare two capsules across all metrics"""
        fp_a = self.baselines.get(capsule_a_id)
        fp_b = self.baselines.get(capsule_b_id)
        
        if not fp_a or not fp_b:
            return {'error': 'Missing fingerprints'}
        
        # Fingerprint similarity
        similarity = fp_a.similarity(fp_b)
        
        # Drift comparison
        drift_a = list(self.drift_history.get(capsule_a_id, []))
        drift_b = list(self.drift_history.get(capsule_b_id, []))
        
        avg_drift_a = np.mean([r.drift_score for r in drift_a]) if drift_a else 0
        avg_drift_b = np.mean([r.drift_score for r in drift_b]) if drift_b else 0
        
        # EVI comparison
        evi_a = list(self.evi_history.get(capsule_a_id, []))
        evi_b = list(self.evi_history.get(capsule_b_id, []))
        
        latest_evi_a = evi_a[-1].evi_score if evi_a else 0
        latest_evi_b = evi_b[-1].evi_score if evi_b else 0
        
        return {
            'capsule_a': capsule_a_id,
            'capsule_b': capsule_b_id,
            'fingerprint_similarity': float(similarity),
            'avg_drift_a': float(avg_drift_a),
            'avg_drift_b': float(avg_drift_b),
            'drift_difference': float(abs(avg_drift_a - avg_drift_b)),
            'evi_a': float(latest_evi_a),
            'evi_b': float(latest_evi_b),
            'evi_difference': float(abs(latest_evi_a - latest_evi_b)),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_capsule_summary(self, capsule_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for a capsule"""
        drift_history = list(self.drift_history.get(capsule_id, []))
        evi_history = list(self.evi_history.get(capsule_id, []))
        mds_history = list(self.mds_history.get(capsule_id, []))
        
        return {
            'capsule_id': capsule_id,
            'has_baseline': capsule_id in self.baselines,
            'drift_records': len(drift_history),
            'latest_drift': drift_history[-1].to_dict() if drift_history else None,
            'drift_trend': self.get_drift_trend(capsule_id),
            'evi_records': len(evi_history),
            'latest_evi': evi_history[-1].to_dict() if evi_history else None,
            'mds_records': len(mds_history),
            'discoveries': sum(1 for m in mds_history if m.is_significant()),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # -------------------------------------------------------------------------
    # NEW METRICS FROM V2 SPEC
    # -------------------------------------------------------------------------
    
    def compute_cdi(self, capsule_id: str, window: int = 10) -> float:
        """
        Compute Capsule Drift Index (CDI).
        
        CDI measures the cumulative drift tendency over time:
        CDI = (weighted_avg_drift * trend_factor) / stability_baseline
        
        Args:
            capsule_id: Capsule to analyze
            window: Number of recent records to consider
            
        Returns:
            CDI score (0.0 = stable, >1.0 = drifting)
        """
        drift_history = list(self.drift_history.get(capsule_id, []))
        if len(drift_history) < 2:
            return 0.0
        
        recent = drift_history[-window:]
        scores = [r.drift_score for r in recent]
        
        # Weighted average (more recent = higher weight)
        weights = np.linspace(0.5, 1.0, len(scores))
        weighted_avg = np.average(scores, weights=weights)
        
        # Trend factor: increasing drift = higher factor
        trend = self.get_drift_trend(capsule_id, window)
        trend_factor = 1.5 if trend == "increasing" else (1.0 if trend == "stable" else 0.7)
        
        # Stability baseline (low variance = high stability)
        variance = np.var(scores)
        stability_baseline = max(0.1, 1.0 - variance)
        
        cdi = (weighted_avg * trend_factor) / stability_baseline
        return float(min(2.0, cdi))  # Cap at 2.0
    
    def compute_scs(self, swarm_fingerprints: List['CapsuleFingerprint']) -> float:
        """
        Compute Swarm Coherence Score (SCS).
        
        SCS measures how well-aligned a swarm's members are:
        SCS = avg_pairwise_similarity * (1 - outlier_ratio) * consensus_factor
        
        Args:
            swarm_fingerprints: List of fingerprints for swarm members
            
        Returns:
            SCS score (0.0-1.0, higher = more coherent)
        """
        if len(swarm_fingerprints) < 2:
            return 1.0 if swarm_fingerprints else 0.0
        
        # Pairwise similarities
        similarities = []
        for i, fp1 in enumerate(swarm_fingerprints):
            for fp2 in swarm_fingerprints[i+1:]:
                sim = abs(fp1.similarity(fp2))
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        # Compute centroid
        centroid = np.mean([fp.vector for fp in swarm_fingerprints], axis=0)
        
        # Outlier detection (distance from centroid)
        distances = []
        for fp in swarm_fingerprints:
            dist = 1.0 - abs(cosine_similarity(fp.vector, centroid))
            distances.append(dist)
        
        outlier_threshold = 0.5
        outlier_count = sum(1 for d in distances if d > outlier_threshold)
        outlier_ratio = outlier_count / len(swarm_fingerprints)
        
        # Consensus factor (low variance in similarities)
        variance = np.var(similarities)
        consensus_factor = max(0.1, 1.0 - variance)
        
        scs = avg_similarity * (1 - outlier_ratio) * consensus_factor
        return float(max(0.0, min(1.0, scs)))
    
    def compute_rsq(self, capsule_id: str, window: int = 20) -> float:
        """
        Compute Reflective Stability Quotient (RSQ).
        
        RSQ measures the overall stability of a capsule's reflective capacity:
        RSQ = (EVI_avg * (1 - CDI)) + (discovery_rate * alignment_factor)
        
        Args:
            capsule_id: Capsule to analyze
            window: Number of recent records to consider
            
        Returns:
            RSQ score (0.0-1.0, higher = more stable)
        """
        # Get EVI history
        evi_history = list(self.evi_history.get(capsule_id, []))
        if not evi_history:
            return 0.5  # Default neutral
        
        recent_evi = evi_history[-window:]
        evi_avg = np.mean([e.evi_score for e in recent_evi])
        
        # Get CDI
        cdi = self.compute_cdi(capsule_id, window)
        
        # Discovery rate (significant patterns found)
        mds_history = list(self.mds_history.get(capsule_id, []))
        if mds_history:
            significant = sum(1 for m in mds_history if m.is_significant())
            discovery_rate = significant / len(mds_history)
        else:
            discovery_rate = 0.0
        
        # Alignment factor (EVI confidence)
        confidence_avg = np.mean([e.confidence for e in recent_evi])
        alignment_factor = confidence_avg
        
        # Compute RSQ
        rsq = (evi_avg * (1 - min(1.0, cdi))) + (discovery_rate * alignment_factor * 0.3)
        return float(max(0.0, min(1.0, rsq)))
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'engine_id': self.engine_id,
            'total_scans': self.total_scans,
            'total_anomalies': self.total_anomalies,
            'total_discoveries': self.total_discoveries,
            'known_patterns': len(self.known_patterns),
            'pattern_classes': {k: len(v) for k, v in self.pattern_classes.items()},
            'tracked_capsules': len(self.baselines),
            'created_at': self.created_at.isoformat()
        }


# =============================================================================
# DSL PREDICATES
# =============================================================================

def evi_valid(engine: ReflectiveMatrixEngine, capsule_id: str, threshold: float = 0.5) -> bool:
    """DSL Predicate: Check if capsule has valid EVI"""
    history = engine.evi_history.get(capsule_id)
    if not history:
        return False
    latest = history[-1]
    return latest.is_valid(threshold)


def drift_forecast_breach(
    engine: ReflectiveMatrixEngine,
    capsule_id: str,
    threshold: float = 0.7
) -> bool:
    """DSL Predicate: Check if drift forecast predicts threshold breach"""
    forecast = engine.forecast_drift(capsule_id)
    return forecast.will_breach(threshold)


def pattern_discovered(
    engine: ReflectiveMatrixEngine,
    capsule_id: str,
    significance_threshold: float = 0.3
) -> bool:
    """DSL Predicate: Check if capsule has made significant discovery"""
    history = engine.mds_history.get(capsule_id)
    if not history:
        return False
    
    return any(m.is_significant(significance_threshold) for m in history)


def compare_fingerprint(
    engine: ReflectiveMatrixEngine,
    capsule_a_id: str,
    capsule_b_id: str,
    threshold: float = 0.8
) -> bool:
    """DSL Predicate: Check if two capsules have similar fingerprints"""
    result = engine.compare_capsules(capsule_a_id, capsule_b_id)
    if 'error' in result:
        return False
    return result['fingerprint_similarity'] >= threshold


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=== Reflective Matrix Engine Demo ===\n")
    
    # Create engine
    engine = ReflectiveMatrixEngine("RME-001")
    
    # Create sample fingerprints
    np.random.seed(42)
    
    # Set up baseline
    baseline_fp = CapsuleFingerprint(
        capsule_id="CAP-001",
        vector=np.random.randn(32),
        traits={'stability': 0.9}
    )
    engine.set_baseline("CAP-001", baseline_fp)
    
    # Simulate drift over time
    print("Simulating drift scans...")
    for i in range(20):
        # Add increasing drift
        noise = np.random.randn(32) * (0.1 + i * 0.02)
        current_fp = CapsuleFingerprint(
            capsule_id="CAP-001",
            vector=baseline_fp.vector + noise,
            traits={'stability': 0.9 - i * 0.02}
        )
        
        record = engine.scan_drift("CAP-001", current_fp)
        if i % 5 == 0:
            print(f"  Step {i}: drift={record.drift_score:.4f}, severity={record.severity.value}")
    
    # Compute EVI
    print("\nComputing EVI...")
    peer_fps = [CapsuleFingerprint(
        capsule_id=f"PEER-{i}",
        vector=baseline_fp.vector + np.random.randn(32) * 0.2,
        traits={}
    ) for i in range(5)]
    
    evi = engine.compute_evi("CAP-001", current_fp, peer_fps)
    print(f"  EVI Score: {evi.evi_score:.4f}")
    print(f"  Coherence: {evi.coherence:.4f}")
    print(f"  Stability: {evi.stability:.4f}")
    print(f"  Lineage Fidelity: {evi.lineage_fidelity:.4f}")
    print(f"  Valid: {evi.is_valid()}")
    
    # Compute MDS
    print("\nComputing MDS...")
    pattern = np.random.randn(32) * 2
    mds = engine.compute_mds("CAP-001", pattern, ["PEER-0", "PEER-1"])
    print(f"  MDS Score: {mds.mds_score:.4f}")
    print(f"  Novelty: {mds.novelty_score:.4f}")
    print(f"  Significance: {mds.significance_weight:.4f}")
    print(f"  Pattern Class: {mds.pattern_class}")
    print(f"  Significant: {mds.is_significant()}")
    
    # Forecast drift
    print("\nForecasting drift...")
    forecast = engine.forecast_drift("CAP-001")
    print(f"  Current: {forecast.current_drift:.4f}")
    print(f"  Trend: {forecast.trend}")
    print(f"  Predictions: {[f'{p:.3f}' for p in forecast.predicted_drift[:5]]}")
    print(f"  Will breach 0.7: {forecast.will_breach(0.7)}")
    if forecast.breach_step:
        print(f"  Steps to breach: {forecast.breach_step}")
    
    # Statistics
    print("\nEngine Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
