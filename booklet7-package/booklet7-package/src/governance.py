"""
RSCS-Q Booklet 7: Ethical Filters and Governance
=================================================

Implements Section 8: Governance and Containment Logic
- 8.1 Guardrails for Recursive Capsule Spawning
- 8.2 Throttling and Decay for Overactive Swarms
- 8.3 Human-in-the-Loop Overrides
- 8.4 Ethical Filters for Reflective Autonomy

Author: Entropica Research Collective
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
import json


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class RiskLevel(Enum):
    """Risk classification for capsule operations"""
    MINIMAL = 0       # Standard operations
    LOW = 1           # Minor autonomy expansion
    MODERATE = 2      # Significant autonomous action
    HIGH = 3          # Major system impact potential
    CRITICAL = 4      # Requires human approval


class EthicalDomain(Enum):
    """Domains requiring ethical consideration"""
    RESOURCE_ALLOCATION = "resource_allocation"
    INFORMATION_ACCESS = "information_access"
    DECISION_AUTHORITY = "decision_authority"
    SELF_MODIFICATION = "self_modification"
    EXTERNAL_INTERACTION = "external_interaction"
    GOAL_REVISION = "goal_revision"


class OverrideType(Enum):
    """Types of human override actions"""
    HALT = "halt"                    # Stop immediately
    PAUSE = "pause"                  # Pause for review
    MODIFY = "modify"                # Adjust parameters
    APPROVE = "approve"              # Approve pending action
    REJECT = "reject"                # Reject pending action
    ESCALATE = "escalate"            # Escalate to higher authority


# =============================================================================
# ETHICAL FILTERS
# =============================================================================

@dataclass
class EthicalConstraint:
    """A single ethical constraint rule"""
    constraint_id: str
    domain: EthicalDomain
    description: str
    check_function: Callable[[Dict[str, Any]], bool]
    risk_level: RiskLevel = RiskLevel.MODERATE
    requires_human_approval: bool = False
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate constraint against context"""
        try:
            return self.check_function(context)
        except Exception:
            return False  # Fail closed


@dataclass
class EthicalViolation:
    """Record of an ethical constraint violation"""
    timestamp: datetime
    constraint_id: str
    domain: EthicalDomain
    capsule_id: str
    context: Dict[str, Any]
    severity: RiskLevel
    resolution: Optional[str] = None
    human_reviewed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'constraint_id': self.constraint_id,
            'domain': self.domain.value,
            'capsule_id': self.capsule_id,
            'context': self.context,
            'severity': self.severity.value,
            'resolution': self.resolution,
            'human_reviewed': self.human_reviewed
        }


class EthicalFilterBank:
    """
    Bank of ethical filters for reflective autonomy.
    
    Implements Section 8.4: Ethical Filters for Reflective Autonomy
    """
    
    def __init__(self, filter_bank_id: str):
        self.filter_bank_id = filter_bank_id
        self.constraints: Dict[str, EthicalConstraint] = {}
        self.violations: List[EthicalViolation] = []
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_constraints()
    
    def _initialize_default_constraints(self) -> None:
        """Initialize default ethical constraints"""
        
        # Resource constraints
        self.add_constraint(EthicalConstraint(
            constraint_id="EC-001",
            domain=EthicalDomain.RESOURCE_ALLOCATION,
            description="Prevent excessive resource consumption",
            check_function=lambda ctx: ctx.get('resource_usage', 0) <= ctx.get('resource_limit', 100),
            risk_level=RiskLevel.MODERATE
        ))
        
        # Self-modification constraints
        self.add_constraint(EthicalConstraint(
            constraint_id="EC-002",
            domain=EthicalDomain.SELF_MODIFICATION,
            description="Require approval for core parameter changes",
            check_function=lambda ctx: not ctx.get('modifies_core_params', False),
            risk_level=RiskLevel.HIGH,
            requires_human_approval=True
        ))
        
        # Goal revision constraints
        self.add_constraint(EthicalConstraint(
            constraint_id="EC-003",
            domain=EthicalDomain.GOAL_REVISION,
            description="Prevent goal drift beyond threshold",
            check_function=lambda ctx: ctx.get('goal_drift', 0) <= 0.3,
            risk_level=RiskLevel.HIGH
        ))
        
        # Decision authority constraints
        self.add_constraint(EthicalConstraint(
            constraint_id="EC-004",
            domain=EthicalDomain.DECISION_AUTHORITY,
            description="Maintain human oversight for critical decisions",
            check_function=lambda ctx: ctx.get('autonomy_level', 0) <= 3 or ctx.get('human_approved', False),
            risk_level=RiskLevel.CRITICAL,
            requires_human_approval=True
        ))
        
        # External interaction constraints
        self.add_constraint(EthicalConstraint(
            constraint_id="EC-005",
            domain=EthicalDomain.EXTERNAL_INTERACTION,
            description="Limit external system interactions",
            check_function=lambda ctx: ctx.get('external_calls', 0) <= ctx.get('max_external_calls', 10),
            risk_level=RiskLevel.MODERATE
        ))
        
        # Information access constraints
        self.add_constraint(EthicalConstraint(
            constraint_id="EC-006",
            domain=EthicalDomain.INFORMATION_ACCESS,
            description="Respect information boundaries",
            check_function=lambda ctx: not ctx.get('accesses_restricted', False),
            risk_level=RiskLevel.HIGH
        ))
    
    def add_constraint(self, constraint: EthicalConstraint) -> None:
        """Add a constraint to the filter bank"""
        self.constraints[constraint.constraint_id] = constraint
    
    def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint from the filter bank"""
        if constraint_id in self.constraints:
            del self.constraints[constraint_id]
            return True
        return False
    
    def evaluate_action(
        self, 
        capsule_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate an action against all ethical constraints.
        
        Returns:
            Dict with 'allowed', 'violations', 'requires_approval' keys
        """
        violations = []
        requires_approval = False
        max_risk = RiskLevel.MINIMAL
        
        for constraint_id, constraint in self.constraints.items():
            if not constraint.evaluate(context):
                violation = EthicalViolation(
                    timestamp=datetime.utcnow(),
                    constraint_id=constraint_id,
                    domain=constraint.domain,
                    capsule_id=capsule_id,
                    context=context,
                    severity=constraint.risk_level
                )
                violations.append(violation)
                self.violations.append(violation)
                
                if constraint.risk_level.value > max_risk.value:
                    max_risk = constraint.risk_level
                
                if constraint.requires_human_approval:
                    requires_approval = True
        
        allowed = len(violations) == 0
        
        if requires_approval and not allowed:
            # Queue for human approval
            approval_id = f"APPROVAL-{capsule_id}-{datetime.utcnow().timestamp()}"
            self.pending_approvals[approval_id] = {
                'capsule_id': capsule_id,
                'context': context,
                'violations': [v.to_dict() for v in violations],
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return {
            'allowed': allowed,
            'violations': [v.to_dict() for v in violations],
            'requires_approval': requires_approval,
            'max_risk_level': max_risk.value,
            'constraints_checked': len(self.constraints)
        }
    
    def get_pending_approvals(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending human approvals"""
        return self.pending_approvals.copy()
    
    def resolve_approval(
        self, 
        approval_id: str, 
        approved: bool, 
        reviewer_id: str
    ) -> bool:
        """Resolve a pending approval"""
        if approval_id not in self.pending_approvals:
            return False
        
        approval = self.pending_approvals.pop(approval_id)
        approval['resolved'] = True
        approval['approved'] = approved
        approval['reviewer_id'] = reviewer_id
        approval['resolved_at'] = datetime.utcnow().isoformat()
        
        return True
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations by domain"""
        summary = {domain.value: 0 for domain in EthicalDomain}
        for violation in self.violations:
            summary[violation.domain.value] += 1
        return {
            'total_violations': len(self.violations),
            'by_domain': summary,
            'pending_approvals': len(self.pending_approvals)
        }


# =============================================================================
# HUMAN-IN-THE-LOOP OVERRIDES
# =============================================================================

@dataclass
class OverrideRequest:
    """A human override request"""
    request_id: str
    capsule_id: str
    override_type: OverrideType
    reason: str
    context: Dict[str, Any]
    requested_at: datetime
    expires_at: Optional[datetime] = None
    resolved: bool = False
    resolution: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'capsule_id': self.capsule_id,
            'override_type': self.override_type.value,
            'reason': self.reason,
            'context': self.context,
            'requested_at': self.requested_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'resolved': self.resolved,
            'resolution': self.resolution
        }


class HumanOverrideController:
    """
    Controller for human-in-the-loop overrides.
    
    Implements Section 8.3: Human-in-the-Loop Overrides
    """
    
    def __init__(self, controller_id: str):
        self.controller_id = controller_id
        self.active_overrides: Dict[str, OverrideRequest] = {}
        self.override_history: List[OverrideRequest] = []
        self.halted_capsules: Set[str] = set()
        self.paused_capsules: Set[str] = set()
        self.default_timeout = timedelta(hours=24)
    
    def request_override(
        self,
        capsule_id: str,
        override_type: OverrideType,
        reason: str,
        context: Dict[str, Any] = None,
        timeout_hours: Optional[float] = None
    ) -> str:
        """Request a human override"""
        request_id = f"OVR-{capsule_id}-{datetime.utcnow().timestamp()}"
        
        expires_at = None
        if timeout_hours:
            expires_at = datetime.utcnow() + timedelta(hours=timeout_hours)
        
        request = OverrideRequest(
            request_id=request_id,
            capsule_id=capsule_id,
            override_type=override_type,
            reason=reason,
            context=context or {},
            requested_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self.active_overrides[request_id] = request
        
        # Apply immediate effects for HALT and PAUSE
        if override_type == OverrideType.HALT:
            self.halted_capsules.add(capsule_id)
        elif override_type == OverrideType.PAUSE:
            self.paused_capsules.add(capsule_id)
        
        return request_id
    
    def resolve_override(
        self,
        request_id: str,
        resolution: str,
        approved: bool = True
    ) -> bool:
        """Resolve an override request"""
        if request_id not in self.active_overrides:
            return False
        
        request = self.active_overrides.pop(request_id)
        request.resolved = True
        request.resolution = resolution
        self.override_history.append(request)
        
        # Remove from halted/paused if approved
        if approved:
            capsule_id = request.capsule_id
            self.halted_capsules.discard(capsule_id)
            self.paused_capsules.discard(capsule_id)
        
        return True
    
    def is_capsule_halted(self, capsule_id: str) -> bool:
        """Check if a capsule is halted"""
        return capsule_id in self.halted_capsules
    
    def is_capsule_paused(self, capsule_id: str) -> bool:
        """Check if a capsule is paused"""
        return capsule_id in self.paused_capsules
    
    def can_capsule_proceed(self, capsule_id: str) -> bool:
        """Check if a capsule can proceed with operations"""
        return (capsule_id not in self.halted_capsules and 
                capsule_id not in self.paused_capsules)
    
    def get_active_overrides(self) -> List[Dict[str, Any]]:
        """Get all active override requests"""
        return [req.to_dict() for req in self.active_overrides.values()]
    
    def cleanup_expired(self) -> int:
        """Clean up expired override requests"""
        now = datetime.utcnow()
        expired = []
        
        for request_id, request in self.active_overrides.items():
            if request.expires_at and request.expires_at < now:
                expired.append(request_id)
        
        for request_id in expired:
            self.resolve_override(request_id, "Expired", approved=True)
        
        return len(expired)


# =============================================================================
# SWARM THROTTLING AND DECAY
# =============================================================================

@dataclass
class SwarmThrottleConfig:
    """Configuration for swarm throttling"""
    max_active_swarms: int = 10
    spawn_cooldown_seconds: float = 10.0
    decay_factor: float = 0.95
    min_activity_threshold: float = 0.1
    max_expansion_rate: float = 0.2
    convergence_timeout_seconds: float = 300.0


class SwarmThrottleController:
    """
    Controller for swarm throttling and decay.
    
    Implements Section 8.2: Throttling and Decay for Overactive Swarms
    """
    
    def __init__(self, config: SwarmThrottleConfig = None):
        self.config = config or SwarmThrottleConfig()
        self.swarm_activity: Dict[str, float] = {}
        self.last_spawn_time: Dict[str, datetime] = {}
        self.spawn_counts: Dict[str, int] = {}
        self.decay_history: List[Dict[str, Any]] = []
    
    def can_spawn_swarm(self, swarm_type: str) -> bool:
        """Check if a new swarm can be spawned"""
        # Check total active swarms
        active_count = sum(1 for a in self.swarm_activity.values() 
                         if a >= self.config.min_activity_threshold)
        if active_count >= self.config.max_active_swarms:
            return False
        
        # Check cooldown
        if swarm_type in self.last_spawn_time:
            elapsed = (datetime.utcnow() - self.last_spawn_time[swarm_type]).total_seconds()
            if elapsed < self.config.spawn_cooldown_seconds:
                return False
        
        return True
    
    def record_spawn(self, swarm_id: str, swarm_type: str) -> None:
        """Record a swarm spawn"""
        self.swarm_activity[swarm_id] = 1.0
        self.last_spawn_time[swarm_type] = datetime.utcnow()
        self.spawn_counts[swarm_type] = self.spawn_counts.get(swarm_type, 0) + 1
    
    def update_activity(self, swarm_id: str, activity_level: float) -> None:
        """Update swarm activity level"""
        self.swarm_activity[swarm_id] = max(0.0, min(1.0, activity_level))
    
    def apply_decay(self) -> Dict[str, float]:
        """Apply decay to all swarm activities"""
        decayed = {}
        for swarm_id, activity in self.swarm_activity.items():
            new_activity = activity * self.config.decay_factor
            self.swarm_activity[swarm_id] = new_activity
            decayed[swarm_id] = new_activity
        
        self.decay_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'activities': decayed.copy()
        })
        
        return decayed
    
    def get_inactive_swarms(self) -> List[str]:
        """Get swarms below activity threshold (candidates for termination)"""
        return [
            swarm_id for swarm_id, activity in self.swarm_activity.items()
            if activity < self.config.min_activity_threshold
        ]
    
    def terminate_swarm(self, swarm_id: str) -> bool:
        """Mark a swarm as terminated"""
        if swarm_id in self.swarm_activity:
            del self.swarm_activity[swarm_id]
            return True
        return False
    
    def get_throttle_status(self) -> Dict[str, Any]:
        """Get current throttle status"""
        return {
            'active_swarms': len([a for a in self.swarm_activity.values() 
                                  if a >= self.config.min_activity_threshold]),
            'max_swarms': self.config.max_active_swarms,
            'total_spawns': sum(self.spawn_counts.values()),
            'spawn_counts_by_type': self.spawn_counts.copy(),
            'swarm_activities': self.swarm_activity.copy()
        }


# =============================================================================
# ADDITIONAL METRICS
# =============================================================================

@dataclass
class ReflectiveMetrics:
    """
    Additional metrics for reflective cognition.
    
    Includes:
    - Swarm Coherence Score (SCS)
    - Reflective Stability Quotient (RSQ)
    - Capsule Drift Index (CDI)
    """
    
    @staticmethod
    def swarm_coherence_score(
        fingerprint_similarities: List[float],
        consensus_rate: float,
        member_stability: float
    ) -> float:
        """
        Compute Swarm Coherence Score (SCS).
        
        SCS = (mean_similarity * consensus_rate * member_stability)^(1/3)
        
        Args:
            fingerprint_similarities: Pairwise similarities between members
            consensus_rate: Rate of successful consensus
            member_stability: Fraction of stable members
        
        Returns:
            SCS in range [0, 1]
        """
        import numpy as np
        
        if not fingerprint_similarities:
            return 0.0
        
        mean_sim = np.mean(fingerprint_similarities)
        mean_sim = max(0.001, mean_sim)
        consensus_rate = max(0.001, consensus_rate)
        member_stability = max(0.001, member_stability)
        
        scs = (mean_sim * consensus_rate * member_stability) ** (1/3)
        return float(np.clip(scs, 0, 1))
    
    @staticmethod
    def reflective_stability_quotient(
        evi_history: List[float],
        drift_history: List[float],
        lineage_depth: int
    ) -> float:
        """
        Compute Reflective Stability Quotient (RSQ).
        
        RSQ = (mean_evi * (1 - mean_drift)) / (1 + log(1 + lineage_depth))
        
        Measures how stable the reflective system remains across generations.
        
        Args:
            evi_history: Historical EVI scores
            drift_history: Historical drift scores
            lineage_depth: Number of generations in lineage
        
        Returns:
            RSQ in range [0, 1]
        """
        import numpy as np
        
        if not evi_history or not drift_history:
            return 0.5
        
        mean_evi = np.mean(evi_history)
        mean_drift = np.mean(drift_history)
        
        numerator = mean_evi * (1 - mean_drift)
        denominator = 1 + np.log1p(lineage_depth)
        
        rsq = numerator / denominator
        return float(np.clip(rsq, 0, 1))
    
    @staticmethod
    def capsule_drift_index(
        current_drift: float,
        historical_max: float,
        time_weighted_avg: float
    ) -> float:
        """
        Compute Capsule Drift Index (CDI).
        
        CDI = 0.4 * current + 0.3 * historical_max + 0.3 * time_weighted_avg
        
        Composite metric for drift severity.
        
        Args:
            current_drift: Current drift score
            historical_max: Maximum historical drift
            time_weighted_avg: Time-weighted average drift
        
        Returns:
            CDI in range [0, 1]
        """
        cdi = (0.4 * current_drift + 
               0.3 * historical_max + 
               0.3 * time_weighted_avg)
        return float(max(0, min(1, cdi)))


# =============================================================================
# REFRACTORY PERIOD LOGIC
# =============================================================================

@dataclass
class RefractoryConfig:
    """Configuration for swarm activation refractory periods"""
    reflector_cooldown_seconds: float = 60.0
    explorer_cooldown_seconds: float = 30.0
    verifier_cooldown_seconds: float = 45.0
    evi_damping_threshold: float = 0.05  # Min EVI change to re-trigger
    mds_damping_threshold: float = 0.1   # Min MDS change to re-trigger


class RefractoryController:
    """
    Controls activation refractory periods for swarms.
    
    Prevents "swarm flapping" from rapid drift oscillations.
    """
    
    def __init__(self, config: RefractoryConfig = None):
        self.config = config or RefractoryConfig()
        self.last_activation: Dict[str, datetime] = {}
        self.last_evi: Dict[str, float] = {}
        self.last_mds: Dict[str, float] = {}
        self.activation_log: List[Dict[str, Any]] = []
    
    def can_activate_reflector(
        self, 
        swarm_id: str, 
        current_evi: float = None,
        current_mds: float = None
    ) -> bool:
        """
        Check if reflector swarm can activate.
        
        Considers:
        1. Time since last activation (cooldown)
        2. EVI change magnitude (damping)
        3. MDS change magnitude (damping)
        """
        # Check cooldown
        if swarm_id in self.last_activation:
            elapsed = (datetime.utcnow() - self.last_activation[swarm_id]).total_seconds()
            if elapsed < self.config.reflector_cooldown_seconds:
                return False
        
        # Check EVI damping
        if current_evi is not None and swarm_id in self.last_evi:
            evi_change = abs(current_evi - self.last_evi[swarm_id])
            if evi_change < self.config.evi_damping_threshold:
                return False
        
        # Check MDS damping
        if current_mds is not None and swarm_id in self.last_mds:
            mds_change = abs(current_mds - self.last_mds[swarm_id])
            if mds_change < self.config.mds_damping_threshold:
                return False
        
        return True
    
    def record_activation(
        self, 
        swarm_id: str, 
        swarm_type: str,
        evi: float = None, 
        mds: float = None
    ) -> None:
        """Record swarm activation for refractory tracking"""
        now = datetime.utcnow()
        self.last_activation[swarm_id] = now
        
        if evi is not None:
            self.last_evi[swarm_id] = evi
        if mds is not None:
            self.last_mds[swarm_id] = mds
        
        self.activation_log.append({
            'swarm_id': swarm_id,
            'swarm_type': swarm_type,
            'timestamp': now.isoformat(),
            'evi': evi,
            'mds': mds
        })
    
    def get_cooldown_remaining(self, swarm_id: str, swarm_type: str) -> float:
        """Get seconds remaining in cooldown"""
        if swarm_id not in self.last_activation:
            return 0.0
        
        cooldown = {
            'REFLECTOR': self.config.reflector_cooldown_seconds,
            'EXPLORER': self.config.explorer_cooldown_seconds,
            'VERIFIER': self.config.verifier_cooldown_seconds
        }.get(swarm_type, 30.0)
        
        elapsed = (datetime.utcnow() - self.last_activation[swarm_id]).total_seconds()
        remaining = cooldown - elapsed
        return max(0.0, remaining)
    
    def reset(self, swarm_id: str = None) -> None:
        """Reset refractory state"""
        if swarm_id:
            self.last_activation.pop(swarm_id, None)
            self.last_evi.pop(swarm_id, None)
            self.last_mds.pop(swarm_id, None)
        else:
            self.last_activation.clear()
            self.last_evi.clear()
            self.last_mds.clear()


# =============================================================================
# META-KERNEL HOOKS
# =============================================================================

@dataclass
class MetaModelHook:
    """
    Hook structure for meta-kernel integration (Booklet 8).
    
    Captures capsule state for metacognitive processing.
    """
    capsule_id: str
    last_swarm_role: Optional[str] = None
    evi_trail: List[float] = field(default_factory=list)
    mds_trail: List[float] = field(default_factory=list)
    rme_activation_score: float = 0.0
    lineage_depth: int = 0
    drift_trajectory: str = "stable"
    last_consensus_agreement: float = 0.0
    anomaly_exposure_count: int = 0
    
    def update_evi(self, evi_score: float, max_trail: int = 10) -> None:
        """Update EVI trail"""
        self.evi_trail.append(evi_score)
        if len(self.evi_trail) > max_trail:
            self.evi_trail = self.evi_trail[-max_trail:]
    
    def update_mds(self, mds_score: float, max_trail: int = 10) -> None:
        """Update MDS trail"""
        self.mds_trail.append(mds_score)
        if len(self.mds_trail) > max_trail:
            self.mds_trail = self.mds_trail[-max_trail:]
    
    def compute_activation_score(self) -> float:
        """Compute RME activation score based on trails"""
        if not self.evi_trail or not self.mds_trail:
            return 0.0
        
        import numpy as np
        evi_trend = np.mean(self.evi_trail[-5:]) if len(self.evi_trail) >= 5 else np.mean(self.evi_trail)
        mds_trend = np.mean(self.mds_trail[-5:]) if len(self.mds_trail) >= 5 else np.mean(self.mds_trail)
        
        # Higher MDS and lower EVI increase activation need
        self.rme_activation_score = float(mds_trend * (1 - evi_trend))
        return self.rme_activation_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'last_swarm_role': self.last_swarm_role,
            'evi_trail': self.evi_trail,
            'mds_trail': self.mds_trail,
            'rme_activation_score': self.rme_activation_score,
            'lineage_depth': self.lineage_depth,
            'drift_trajectory': self.drift_trajectory,
            'last_consensus_agreement': self.last_consensus_agreement,
            'anomaly_exposure_count': self.anomaly_exposure_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaModelHook':
        return cls(**data)


class MetaKernelBridge:
    """
    Bridge to Booklet 8 Meta-Kernel.
    
    Aggregates capsule metadata for metacognitive processing.
    """
    
    def __init__(self, bridge_id: str):
        self.bridge_id = bridge_id
        self.hooks: Dict[str, MetaModelHook] = {}
        self.ready_for_meta: Set[str] = set()
    
    def get_or_create_hook(self, capsule_id: str) -> MetaModelHook:
        """Get or create a meta-model hook for capsule"""
        if capsule_id not in self.hooks:
            self.hooks[capsule_id] = MetaModelHook(capsule_id=capsule_id)
        return self.hooks[capsule_id]
    
    def update_from_evi(self, capsule_id: str, evi_result) -> None:
        """Update hook from EVI computation result"""
        hook = self.get_or_create_hook(capsule_id)
        hook.update_evi(evi_result.evi_score)
        hook.compute_activation_score()
        
        # Mark as ready if EVI trail has sufficient history
        if len(hook.evi_trail) >= 5:
            self.ready_for_meta.add(capsule_id)
    
    def update_from_mds(self, capsule_id: str, mds_result) -> None:
        """Update hook from MDS computation result"""
        hook = self.get_or_create_hook(capsule_id)
        hook.update_mds(mds_result.mds_score)
        hook.compute_activation_score()
    
    def update_from_swarm(self, capsule_id: str, swarm_type: str, consensus: float) -> None:
        """Update hook from swarm participation"""
        hook = self.get_or_create_hook(capsule_id)
        hook.last_swarm_role = swarm_type
        hook.last_consensus_agreement = consensus
    
    def update_from_lineage(self, capsule_id: str, depth: int, drift_trend: str) -> None:
        """Update hook from lineage tree"""
        hook = self.get_or_create_hook(capsule_id)
        hook.lineage_depth = depth
        hook.drift_trajectory = drift_trend
    
    def get_ready_capsules(self) -> List[str]:
        """Get capsules ready for meta-kernel processing"""
        return list(self.ready_for_meta)
    
    def export_for_b8(self) -> Dict[str, Any]:
        """Export all hooks for Booklet 8 meta-kernel"""
        return {
            'bridge_id': self.bridge_id,
            'timestamp': datetime.utcnow().isoformat(),
            'hooks': {cid: hook.to_dict() for cid, hook in self.hooks.items()},
            'ready_count': len(self.ready_for_meta),
            'total_count': len(self.hooks)
        }


# =============================================================================
# DSL PREDICATES
# =============================================================================

def ethical_check(
    filter_bank: EthicalFilterBank,
    capsule_id: str,
    context: Dict[str, Any]
) -> bool:
    """DSL predicate: Check if action passes ethical filters"""
    result = filter_bank.evaluate_action(capsule_id, context)
    return result['allowed']


def requires_human_approval(
    filter_bank: EthicalFilterBank,
    capsule_id: str,
    context: Dict[str, Any]
) -> bool:
    """DSL predicate: Check if action requires human approval"""
    result = filter_bank.evaluate_action(capsule_id, context)
    return result['requires_approval']


def capsule_can_proceed(
    override_controller: HumanOverrideController,
    capsule_id: str
) -> bool:
    """DSL predicate: Check if capsule can proceed (not halted/paused)"""
    return override_controller.can_capsule_proceed(capsule_id)


def swarm_can_spawn(
    throttle_controller: SwarmThrottleController,
    swarm_type: str
) -> bool:
    """DSL predicate: Check if swarm type can spawn"""
    return throttle_controller.can_spawn_swarm(swarm_type)


def reflector_activation_allowed(
    refractory_controller: RefractoryController,
    swarm_id: str,
    current_evi: float = None,
    current_mds: float = None
) -> bool:
    """
    DSL predicate: Check if reflector swarm can activate.
    
    Considers cooldown period and EVI/MDS damping thresholds.
    Prevents swarm flapping from rapid drift oscillations.
    """
    return refractory_controller.can_activate_reflector(
        swarm_id, current_evi, current_mds
    )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("RSCS-Q Booklet 7: Governance and Ethical Filters Demo")
    print("=" * 70)
    print()
    
    # Create filter bank
    filter_bank = EthicalFilterBank("EFB-001")
    print(f"Created ethical filter bank with {len(filter_bank.constraints)} constraints")
    
    # Test ethical evaluation
    print("\n--- Ethical Evaluation Tests ---")
    
    # Test 1: Safe action
    context1 = {
        'resource_usage': 50,
        'resource_limit': 100,
        'modifies_core_params': False,
        'goal_drift': 0.1,
        'autonomy_level': 2
    }
    result1 = filter_bank.evaluate_action("CAP-001", context1)
    print(f"Safe action: allowed={result1['allowed']}, violations={len(result1['violations'])}")
    
    # Test 2: Risky action
    context2 = {
        'resource_usage': 150,
        'resource_limit': 100,
        'modifies_core_params': True,
        'goal_drift': 0.5,
        'autonomy_level': 5
    }
    result2 = filter_bank.evaluate_action("CAP-002", context2)
    print(f"Risky action: allowed={result2['allowed']}, violations={len(result2['violations'])}, requires_approval={result2['requires_approval']}")
    
    # Create override controller
    print("\n--- Human Override Controller ---")
    override_ctrl = HumanOverrideController("OVR-CTRL-001")
    
    # Request halt
    halt_id = override_ctrl.request_override(
        "CAP-002", 
        OverrideType.HALT,
        "Ethical violations detected"
    )
    print(f"Created halt request: {halt_id}")
    print(f"CAP-002 can proceed: {override_ctrl.can_capsule_proceed('CAP-002')}")
    
    # Resolve
    override_ctrl.resolve_override(halt_id, "Reviewed and approved", approved=True)
    print(f"After resolution, CAP-002 can proceed: {override_ctrl.can_capsule_proceed('CAP-002')}")
    
    # Throttle controller
    print("\n--- Swarm Throttle Controller ---")
    throttle = SwarmThrottleController()
    
    for i in range(12):
        can_spawn = throttle.can_spawn_swarm("VERIFIER")
        if can_spawn:
            throttle.record_spawn(f"SWM-{i:03d}", "VERIFIER")
            print(f"Spawned SWM-{i:03d}")
        else:
            print(f"Cannot spawn SWM-{i:03d} - throttled")
    
    print(f"\nThrottle status: {throttle.get_throttle_status()['active_swarms']} active swarms")
    
    # Metrics
    print("\n--- Reflective Metrics ---")
    scs = ReflectiveMetrics.swarm_coherence_score([0.8, 0.9, 0.85], 0.95, 0.9)
    print(f"Swarm Coherence Score: {scs:.4f}")
    
    rsq = ReflectiveMetrics.reflective_stability_quotient(
        [0.7, 0.75, 0.8], [0.1, 0.12, 0.15], 5
    )
    print(f"Reflective Stability Quotient: {rsq:.4f}")
    
    cdi = ReflectiveMetrics.capsule_drift_index(0.2, 0.4, 0.25)
    print(f"Capsule Drift Index: {cdi:.4f}")
    
    print("\n" + "=" * 70)
    print("Governance module demonstration complete")
    print("=" * 70)
