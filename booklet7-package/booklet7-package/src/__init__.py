"""
RSCS-Q Booklet 7: Reflective Swarms and Emergent Cognition
============================================================

This package provides the architecture for reflective capsules and 
autonomous swarms within the RSCS-Q cognitive substrate.

Key Components:
- Capsule Lineage System: Parent-child relationships and genealogy
- Swarm Reflector: Autonomous swarm coordination
- Reflective Matrix Engine: EVI, MDS metrics and drift forecasting

Author: Entropica Research Collective
Version: 1.0
"""

# Capsule Lineage System
from .capsule_lineage import (
    CapsuleFingerprint,
    LineageNode,
    CapsuleLineageTree,
    LineageRole,
    CheckInStatus,
    InheritanceMode,
    DriftPolicy,  # New
    # DSL Predicates
    lineage_check,
    capsule_family_drift,
    check_lineage_trajectory,
    lineage_depth,
    # New DSL Predicates (GCP/RIP)
    inherit_capsule,
    escalate_if_diverges,
    track_lineage,
    calculate_drift,
)

# Swarm Reflector System
from .swarm_reflector import (
    Swarm,
    SwarmMember,
    SwarmMessage,
    SwarmCoordinator,
    SwarmType,
    SwarmPhase,
    MessageType,
    TriggerCondition,
    # DSL Predicates
    swarm_consensus_reached,
    swarm_has_outliers,
    trigger_reflective_response,
    reflector_trigger,
)

# Reflective Matrix Engine
from .reflective_matrix_engine import (
    ReflectiveMatrixEngine,
    DriftRecord,
    DriftType,
    DriftSeverity,
    EVIResult,
    MDSResult,
    DriftForecast,
    # DSL Predicates
    evi_valid,
    drift_forecast_breach,
    pattern_discovered,
    compare_fingerprint,
)

# Simulation Harness
from .simulation_harness import (
    B7SimulationHarness,
    SimulationConfig,
    SyntheticLineageGenerator,
    SyntheticSwarmGenerator,
)

# Governance and Ethical Filters
from .governance import (
    EthicalFilterBank,
    EthicalConstraint,
    EthicalViolation,
    EthicalDomain,
    RiskLevel,
    HumanOverrideController,
    OverrideRequest,
    OverrideType,
    SwarmThrottleController,
    SwarmThrottleConfig,
    ReflectiveMetrics,
    # Refractory period
    RefractoryController,
    RefractoryConfig,
    # Meta-kernel hooks
    MetaModelHook,
    MetaKernelBridge,
    # DSL predicates
    ethical_check,
    requires_human_approval,
    capsule_can_proceed,
    swarm_can_spawn,
    reflector_activation_allowed,
)

__version__ = "1.0.0"
__author__ = "Entropica Research Collective"

__all__ = [
    # Capsule Lineage
    "CapsuleFingerprint",
    "LineageNode", 
    "CapsuleLineageTree",
    "LineageRole",
    "CheckInStatus",
    "InheritanceMode",
    "DriftPolicy",
    "lineage_check",
    "capsule_family_drift",
    "check_lineage_trajectory",
    "lineage_depth",
    "inherit_capsule",
    "escalate_if_diverges",
    "track_lineage",
    "calculate_drift",
    # Swarm Reflector
    "Swarm",
    "SwarmMember",
    "SwarmMessage",
    "SwarmCoordinator",
    "SwarmType",
    "SwarmPhase",
    "MessageType",
    "TriggerCondition",
    "swarm_consensus_reached",
    "swarm_has_outliers",
    "trigger_reflective_response",
    "reflector_trigger",
    # Reflective Matrix Engine
    "ReflectiveMatrixEngine",
    "DriftRecord",
    "DriftType",
    "DriftSeverity",
    "EVIResult",
    "MDSResult",
    "DriftForecast",
    "evi_valid",
    "drift_forecast_breach",
    "pattern_discovered",
    "compare_fingerprint",
    # Simulation Harness
    "B7SimulationHarness",
    "SimulationConfig",
    "SyntheticLineageGenerator",
    "SyntheticSwarmGenerator",
    # Governance
    "EthicalFilterBank",
    "EthicalConstraint",
    "EthicalViolation",
    "EthicalDomain",
    "RiskLevel",
    "HumanOverrideController",
    "OverrideRequest",
    "OverrideType",
    "SwarmThrottleController",
    "SwarmThrottleConfig",
    "ReflectiveMetrics",
    "RefractoryController",
    "RefractoryConfig",
    "MetaModelHook",
    "MetaKernelBridge",
    "ethical_check",
    "requires_human_approval",
    "capsule_can_proceed",
    "swarm_can_spawn",
    "reflector_activation_allowed",
]
