"""
RSCS-Q Booklet 7: Capsule Lineage System
=========================================

Implements parent-child capsule relationships, genealogical tracking,
hereditary trait inheritance, and lineage-based reflection.

Key Features:
- Parent-child imprint system
- Hereditary capsule IDs and trait inheritance
- Genealogy trees and recursive monitoring
- Check-in protocols and failure cascades
- Lineage-based anomaly detection

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from collections import defaultdict

# Import B6 components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'booklet6-package', 'src'))

try:
    from capsule_matrix import CapsuleMatrix, CapsuleMatrixConfig, cosine_similarity
except ImportError:
    # Fallback definitions if B6 not available
    def cosine_similarity(a, b):
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# ENUMS
# =============================================================================

class LineageRole(Enum):
    """Role of capsule in lineage hierarchy"""
    ROOT = "root"           # Original ancestor, no parent
    CHILD = "child"         # Has parent, may have children
    LEAF = "leaf"           # Has parent, no children
    ORPHAN = "orphan"       # Parent lost/terminated


class CheckInStatus(Enum):
    """Status of capsule check-in"""
    HEALTHY = "healthy"
    DELAYED = "delayed"
    MISSED = "missed"
    ANOMALOUS = "anomalous"
    TERMINATED = "terminated"


class InheritanceMode(Enum):
    """How traits are inherited from parent"""
    FULL = "full"           # Complete trait copy
    PARTIAL = "partial"     # Selected traits only
    MUTATED = "mutated"     # Traits with variation
    NONE = "none"           # No inheritance


class DriftPolicy(Enum):
    """
    Drift Response Policies (Section 2.4 of Spec)
    
    Defines how the system responds to detected drift.
    """
    STRICT = "strict"       # Immediate halt and escalation
    ADAPTIVE = "adaptive"   # Allow bounded drift with monitoring
    ESCALATE = "escalate"   # Notify parent/swarm for review
    ARCHIVE = "archive"     # Log and continue (for research)


# =============================================================================
# CAPSULE FINGERPRINT
# =============================================================================

@dataclass
class CapsuleFingerprint:
    """
    Compressed signature encoding capsule traits for matching and anomaly detection.
    
    Used for:
    - Lineage similarity comparison
    - Anomaly detection vs. baseline
    - Trait inheritance tracking
    """
    capsule_id: str
    vector: np.ndarray
    traits: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def similarity(self, other: 'CapsuleFingerprint') -> float:
        """Compute similarity to another fingerprint"""
        return cosine_similarity(self.vector, other.vector)
    
    def trait_distance(self, other: 'CapsuleFingerprint') -> float:
        """Compute trait-space distance"""
        common_traits = set(self.traits.keys()) & set(other.traits.keys())
        if not common_traits:
            return 1.0
        
        diffs = [abs(self.traits[t] - other.traits[t]) for t in common_traits]
        return np.mean(diffs)
    
    def hash(self) -> str:
        """Get fingerprint hash"""
        return hashlib.sha256(self.vector.tobytes()).hexdigest()[:16]
    
    def mutate(self, mutation_rate: float = 0.1) -> 'CapsuleFingerprint':
        """Create mutated copy of fingerprint"""
        noise = np.random.randn(len(self.vector)) * mutation_rate
        new_vector = self.vector + noise
        new_traits = {k: v + np.random.randn() * mutation_rate 
                      for k, v in self.traits.items()}
        
        return CapsuleFingerprint(
            capsule_id=f"{self.capsule_id}-mut",
            vector=new_vector,
            traits=new_traits,
            generation=self.generation + 1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'vector': self.vector.tolist(),
            'traits': self.traits,
            'generation': self.generation,
            'hash': self.hash()
        }


# =============================================================================
# LINEAGE NODE
# =============================================================================

@dataclass
class LineageNode:
    """
    A node in the capsule genealogy tree.
    
    Tracks:
    - Parent-child relationships
    - Trait inheritance
    - Check-in status
    - Anomaly history
    """
    capsule_id: str
    fingerprint: CapsuleFingerprint
    
    # Lineage relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    role: LineageRole = LineageRole.ROOT
    generation: int = 0
    
    # Inheritance
    inheritance_mode: InheritanceMode = InheritanceMode.FULL
    inherited_traits: Dict[str, float] = field(default_factory=dict)
    
    # Check-in tracking
    check_in_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    last_check_in: Optional[datetime] = None
    check_in_status: CheckInStatus = CheckInStatus.HEALTHY
    missed_check_ins: int = 0
    max_missed_check_ins: int = 3
    
    # Anomaly tracking
    anomaly_count: int = 0
    anomaly_history: List[Dict[str, Any]] = field(default_factory=list)
    baseline_fingerprint: Optional[CapsuleFingerprint] = None
    escalation_flags: List[str] = field(default_factory=list)
    is_active_flag: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    terminated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.baseline_fingerprint is None:
            self.baseline_fingerprint = self.fingerprint
        if self.last_check_in is None:
            self.last_check_in = self.created_at
    
    # -------------------------------------------------------------------------
    # CHECK-IN MANAGEMENT
    # -------------------------------------------------------------------------
    
    def check_in(self, current_fingerprint: Optional[CapsuleFingerprint] = None) -> CheckInStatus:
        """
        Perform check-in with parent/system.
        
        Args:
            current_fingerprint: Current state fingerprint for anomaly check
            
        Returns:
            CheckInStatus after evaluation
        """
        now = datetime.utcnow()
        self.last_check_in = now
        
        # Check for anomaly if fingerprint provided
        if current_fingerprint is not None:
            similarity = self.baseline_fingerprint.similarity(current_fingerprint)
            if similarity < 0.7:  # Anomaly threshold
                self.check_in_status = CheckInStatus.ANOMALOUS
                self.anomaly_count += 1
                self.anomaly_history.append({
                    'timestamp': now.isoformat(),
                    'similarity': similarity,
                    'fingerprint_hash': current_fingerprint.hash()
                })
                return self.check_in_status
        
        self.check_in_status = CheckInStatus.HEALTHY
        self.missed_check_ins = 0
        return self.check_in_status
    
    def evaluate_check_in_status(self) -> CheckInStatus:
        """Evaluate current check-in status based on time"""
        if self.terminated_at is not None:
            return CheckInStatus.TERMINATED
        
        now = datetime.utcnow()
        time_since_check_in = now - self.last_check_in
        
        if time_since_check_in > self.check_in_interval * 2:
            self.missed_check_ins += 1
            if self.missed_check_ins >= self.max_missed_check_ins:
                self.check_in_status = CheckInStatus.MISSED
            else:
                self.check_in_status = CheckInStatus.DELAYED
        
        return self.check_in_status
    
    def adjust_check_in_frequency(self, factor: float) -> None:
        """Adjust check-in frequency by factor"""
        new_seconds = self.check_in_interval.total_seconds() * factor
        self.check_in_interval = timedelta(seconds=max(5, min(300, new_seconds)))
    
    # -------------------------------------------------------------------------
    # LINEAGE OPERATIONS
    # -------------------------------------------------------------------------
    
    def add_child(self, child_id: str) -> None:
        """Register a child capsule"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
        if self.role == LineageRole.LEAF:
            self.role = LineageRole.CHILD
    
    def remove_child(self, child_id: str) -> None:
        """Remove a child capsule"""
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)
        if not self.children_ids and self.parent_id is not None:
            self.role = LineageRole.LEAF
    
    def is_ancestor_of(self, other_generation: int) -> bool:
        """Check if this node could be ancestor of given generation"""
        return self.generation < other_generation
    
    def terminate(self) -> None:
        """Mark capsule as terminated"""
        self.terminated_at = datetime.utcnow()
        self.check_in_status = CheckInStatus.TERMINATED
    
    def is_active(self) -> bool:
        """Check if capsule is still active"""
        return self.terminated_at is None and self.is_active_flag
    
    # -------------------------------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'fingerprint': self.fingerprint.to_dict(),
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'role': self.role.value,
            'generation': self.generation,
            'inheritance_mode': self.inheritance_mode.value,
            'check_in_status': self.check_in_status.value,
            'anomaly_count': self.anomaly_count,
            'created_at': self.created_at.isoformat(),
            'terminated_at': self.terminated_at.isoformat() if self.terminated_at else None
        }


# =============================================================================
# CAPSULE LINEAGE TREE
# =============================================================================

class CapsuleLineageTree:
    """
    Manages the complete genealogy of capsules.
    
    Provides:
    - Parent-child relationship management
    - Ancestry tracing
    - Lineage-based anomaly detection
    - Check-in monitoring
    - Trait inheritance tracking
    """
    
    def __init__(self, tree_id: str):
        self.tree_id = tree_id
        self.nodes: Dict[str, LineageNode] = {}
        self.root_ids: List[str] = []
        
        # Indices for fast lookup
        self._parent_index: Dict[str, str] = {}  # child_id -> parent_id
        self._children_index: Dict[str, List[str]] = defaultdict(list)  # parent_id -> [child_ids]
        self._generation_index: Dict[int, List[str]] = defaultdict(list)  # gen -> [capsule_ids]
        
        # Event handlers
        self._on_anomaly: Optional[Callable[[LineageNode], None]] = None
        self._on_missed_check_in: Optional[Callable[[LineageNode], None]] = None
        
        # Statistics
        self.total_spawned = 0
        self.total_terminated = 0
        self.created_at = datetime.utcnow()
    
    # -------------------------------------------------------------------------
    # NODE MANAGEMENT
    # -------------------------------------------------------------------------
    
    def add_root(
        self,
        capsule_id: str,
        fingerprint: CapsuleFingerprint,
        **kwargs
    ) -> LineageNode:
        """Add a root capsule (no parent)"""
        node = LineageNode(
            capsule_id=capsule_id,
            fingerprint=fingerprint,
            role=LineageRole.ROOT,
            generation=0,
            **kwargs
        )
        
        self.nodes[capsule_id] = node
        self.root_ids.append(capsule_id)
        self._generation_index[0].append(capsule_id)
        self.total_spawned += 1
        
        return node
    
    def spawn_child(
        self,
        parent_id: str,
        child_id: str,
        inheritance_mode: InheritanceMode = InheritanceMode.FULL,
        mutation_rate: float = 0.0,
        **kwargs
    ) -> Optional[LineageNode]:
        """
        Spawn a child capsule from parent.
        
        Args:
            parent_id: ID of parent capsule
            child_id: ID for new child
            inheritance_mode: How to inherit traits
            mutation_rate: Rate of trait mutation (0-1)
            
        Returns:
            New LineageNode or None if parent not found
        """
        parent = self.nodes.get(parent_id)
        if parent is None:
            return None
        
        # Create child fingerprint based on inheritance mode
        if inheritance_mode == InheritanceMode.FULL:
            child_fingerprint = CapsuleFingerprint(
                capsule_id=child_id,
                vector=parent.fingerprint.vector.copy(),
                traits=parent.fingerprint.traits.copy(),
                generation=parent.generation + 1
            )
        elif inheritance_mode == InheritanceMode.MUTATED:
            child_fingerprint = parent.fingerprint.mutate(mutation_rate)
            child_fingerprint.capsule_id = child_id
        elif inheritance_mode == InheritanceMode.PARTIAL:
            # Inherit 50% of traits randomly
            inherited_traits = {k: v for k, v in parent.fingerprint.traits.items()
                              if np.random.random() > 0.5}
            child_fingerprint = CapsuleFingerprint(
                capsule_id=child_id,
                vector=parent.fingerprint.vector.copy() * 0.5 + np.random.randn(len(parent.fingerprint.vector)) * 0.5,
                traits=inherited_traits,
                generation=parent.generation + 1
            )
        else:  # NONE
            child_fingerprint = CapsuleFingerprint(
                capsule_id=child_id,
                vector=np.random.randn(len(parent.fingerprint.vector)),
                traits={},
                generation=parent.generation + 1
            )
        
        # Create child node
        child_node = LineageNode(
            capsule_id=child_id,
            fingerprint=child_fingerprint,
            parent_id=parent_id,
            role=LineageRole.LEAF,
            generation=parent.generation + 1,
            inheritance_mode=inheritance_mode,
            inherited_traits=parent.fingerprint.traits.copy(),
            **kwargs
        )
        
        # Update indices
        self.nodes[child_id] = child_node
        parent.add_child(child_id)
        self._parent_index[child_id] = parent_id
        self._children_index[parent_id].append(child_id)
        self._generation_index[child_node.generation].append(child_id)
        self.total_spawned += 1
        
        return child_node
    
    def terminate_capsule(self, capsule_id: str, cascade: bool = False) -> List[str]:
        """
        Terminate a capsule.
        
        Args:
            capsule_id: ID of capsule to terminate
            cascade: If True, also terminate all descendants
            
        Returns:
            List of terminated capsule IDs
        """
        terminated = []
        node = self.nodes.get(capsule_id)
        
        if node is None:
            return terminated
        
        node.terminate()
        terminated.append(capsule_id)
        self.total_terminated += 1
        
        if cascade:
            for child_id in node.children_ids.copy():
                terminated.extend(self.terminate_capsule(child_id, cascade=True))
        else:
            # Mark children as orphans
            for child_id in node.children_ids:
                child = self.nodes.get(child_id)
                if child:
                    child.role = LineageRole.ORPHAN
        
        return terminated
    
    def get_node(self, capsule_id: str) -> Optional[LineageNode]:
        """Get a lineage node by ID"""
        return self.nodes.get(capsule_id)
    
    # -------------------------------------------------------------------------
    # ANCESTRY OPERATIONS
    # -------------------------------------------------------------------------
    
    def get_ancestors(self, capsule_id: str) -> List[LineageNode]:
        """Get all ancestors of a capsule (parent, grandparent, etc.)"""
        ancestors = []
        current_id = capsule_id
        
        while current_id in self._parent_index:
            parent_id = self._parent_index[current_id]
            parent = self.nodes.get(parent_id)
            if parent:
                ancestors.append(parent)
            current_id = parent_id
        
        return ancestors
    
    def get_descendants(self, capsule_id: str) -> List[LineageNode]:
        """Get all descendants of a capsule"""
        descendants = []
        to_visit = [capsule_id]
        
        while to_visit:
            current_id = to_visit.pop(0)
            for child_id in self._children_index.get(current_id, []):
                child = self.nodes.get(child_id)
                if child:
                    descendants.append(child)
                    to_visit.append(child_id)
        
        return descendants
    
    def get_siblings(self, capsule_id: str) -> List[LineageNode]:
        """Get siblings of a capsule (same parent)"""
        parent_id = self._parent_index.get(capsule_id)
        if parent_id is None:
            return []
        
        siblings = []
        for child_id in self._children_index.get(parent_id, []):
            if child_id != capsule_id:
                child = self.nodes.get(child_id)
                if child:
                    siblings.append(child)
        
        return siblings
    
    def get_lineage_path(self, capsule_id: str) -> List[str]:
        """Get path from root to capsule"""
        path = [capsule_id]
        current_id = capsule_id
        
        while current_id in self._parent_index:
            parent_id = self._parent_index[current_id]
            path.insert(0, parent_id)
            current_id = parent_id
        
        return path
    
    def get_generation(self, generation: int) -> List[LineageNode]:
        """Get all capsules at a specific generation"""
        return [self.nodes[cid] for cid in self._generation_index.get(generation, [])
                if cid in self.nodes]
    
    def get_max_generation(self) -> int:
        """Get the maximum generation depth"""
        return max(self._generation_index.keys()) if self._generation_index else 0
    
    # -------------------------------------------------------------------------
    # CHECK-IN MONITORING
    # -------------------------------------------------------------------------
    
    def process_check_ins(self) -> Dict[str, CheckInStatus]:
        """Process check-ins for all active capsules"""
        results = {}
        
        for capsule_id, node in self.nodes.items():
            if not node.is_active():
                continue
            
            status = node.evaluate_check_in_status()
            results[capsule_id] = status
            
            # Trigger callbacks
            if status == CheckInStatus.MISSED and self._on_missed_check_in:
                self._on_missed_check_in(node)
            elif status == CheckInStatus.ANOMALOUS and self._on_anomaly:
                self._on_anomaly(node)
        
        return results
    
    def set_anomaly_handler(self, handler: Callable[[LineageNode], None]) -> None:
        """Set callback for anomaly detection"""
        self._on_anomaly = handler
    
    def set_missed_check_in_handler(self, handler: Callable[[LineageNode], None]) -> None:
        """Set callback for missed check-ins"""
        self._on_missed_check_in = handler
    
    # -------------------------------------------------------------------------
    # LINEAGE ANALYSIS
    # -------------------------------------------------------------------------
    
    def compute_lineage_drift(self, capsule_id: str) -> float:
        """
        Compute accumulated drift from root to capsule.
        
        Returns:
            Total drift score (0 = identical to root, higher = more diverged)
        """
        ancestors = self.get_ancestors(capsule_id)
        if not ancestors:
            return 0.0
        
        node = self.nodes.get(capsule_id)
        if node is None:
            return 0.0
        
        root = ancestors[-1]  # Last ancestor is root
        return 1.0 - node.fingerprint.similarity(root.fingerprint)
    
    def find_similar_capsules(
        self,
        fingerprint: CapsuleFingerprint,
        threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """Find capsules similar to given fingerprint"""
        similarities = []
        
        for capsule_id, node in self.nodes.items():
            if not node.is_active():
                continue
            
            sim = node.fingerprint.similarity(fingerprint)
            if sim >= threshold:
                similarities.append((capsule_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def detect_lineage_anomalies(self, threshold: float = 0.5) -> List[LineageNode]:
        """Detect capsules that have drifted significantly from their lineage"""
        anomalies = []
        
        for capsule_id, node in self.nodes.items():
            if not node.is_active() or node.role == LineageRole.ROOT:
                continue
            
            drift = self.compute_lineage_drift(capsule_id)
            if drift > threshold:
                anomalies.append(node)
        
        return anomalies
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tree statistics"""
        active_count = sum(1 for n in self.nodes.values() if n.is_active())
        
        return {
            'tree_id': self.tree_id,
            'total_nodes': len(self.nodes),
            'active_nodes': active_count,
            'terminated_nodes': self.total_terminated,
            'root_count': len(self.root_ids),
            'max_generation': self.get_max_generation(),
            'generation_distribution': {g: len(ids) for g, ids in self._generation_index.items()},
            'created_at': self.created_at.isoformat()
        }
    
    # -------------------------------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Export tree to dictionary"""
        return {
            'tree_id': self.tree_id,
            'nodes': {cid: node.to_dict() for cid, node in self.nodes.items()},
            'root_ids': self.root_ids,
            'statistics': self.get_statistics()
        }
    
    def to_json(self) -> str:
        """Export tree to JSON"""
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# DSL PREDICATES
# =============================================================================

def lineage_check(tree: CapsuleLineageTree, capsule_id: str) -> bool:
    """
    DSL Predicate: Check if capsule has valid lineage.
    
    Returns True if:
    - Capsule exists in tree
    - Capsule is active
    - Parent chain is intact (no orphans)
    """
    node = tree.get_node(capsule_id)
    if node is None or not node.is_active():
        return False
    
    if node.role == LineageRole.ORPHAN:
        return False
    
    return True


def capsule_family_drift(
    tree: CapsuleLineageTree,
    capsule_id: str,
    threshold: float = 0.3
) -> bool:
    """
    DSL Predicate: Check if capsule family shows drift.
    
    Returns True if average drift across siblings exceeds threshold.
    """
    node = tree.get_node(capsule_id)
    if node is None:
        return False
    
    siblings = tree.get_siblings(capsule_id)
    if not siblings:
        return False
    
    # Compute average similarity to siblings
    similarities = [node.fingerprint.similarity(s.fingerprint) for s in siblings]
    avg_similarity = np.mean(similarities)
    
    return (1.0 - avg_similarity) > threshold


def check_lineage_trajectory(
    tree: CapsuleLineageTree,
    capsule_id: str,
    max_drift: float = 0.5
) -> bool:
    """
    DSL Predicate: Check if lineage trajectory is within bounds.
    
    Returns True if accumulated drift from root is within max_drift.
    """
    drift = tree.compute_lineage_drift(capsule_id)
    return drift <= max_drift


def lineage_depth(tree: CapsuleLineageTree, capsule_id: str) -> int:
    """
    DSL Predicate: Get lineage depth (generation number).
    """
    node = tree.get_node(capsule_id)
    return node.generation if node else 0


# =============================================================================
# DSL PREDICATES FROM NOTES (GCP/RIP Protocol)
# =============================================================================

def inherit_capsule(
    tree: CapsuleLineageTree,
    child_id: str,
    parent_id: str,
    mode: InheritanceMode = InheritanceMode.FULL,
    mutation_rate: float = 0.1
) -> Optional['LineageNode']:
    """
    DSL Predicate: Inherit capsule traits from parent.
    
    Implements Reflexive Inheritance Policy (RIP) from spec:
    - Inherited attributes: goal fragments, rubric seeds, thresholds
    - Risk-aware oversight from parent capsules
    
    Args:
        tree: The lineage tree
        child_id: ID for new child capsule
        parent_id: ID of parent to inherit from
        mode: Inheritance mode (FULL, PARTIAL, MUTATED, NONE)
        mutation_rate: Rate of trait mutation for MUTATED mode
    
    Returns:
        The new child node, or None if parent not found
    """
    return tree.spawn_child(parent_id, child_id, mode, mutation_rate)


def escalate_if_diverges(
    tree: CapsuleLineageTree,
    capsule_id: str,
    threshold: float = 0.5,
    policy: DriftPolicy = DriftPolicy.ESCALATE
) -> Dict[str, Any]:
    """
    DSL Predicate: Escalate if capsule diverges beyond threshold.
    
    Implements dynamic escalation via drift detection from notes.
    
    Args:
        tree: The lineage tree
        capsule_id: Capsule to check
        threshold: Drift threshold for escalation
        policy: Response policy (STRICT, ADAPTIVE, ESCALATE, ARCHIVE)
    
    Returns:
        Escalation result with action taken
    """
    node = tree.get_node(capsule_id)
    if node is None:
        return {"escalated": False, "reason": "capsule_not_found"}
    
    # Calculate drift from baseline or parent
    drift = tree.compute_lineage_drift(capsule_id)
    
    result = {
        "capsule_id": capsule_id,
        "drift": drift,
        "threshold": threshold,
        "policy": policy.value,
        "escalated": False,
        "action": None
    }
    
    if drift > threshold:
        result["escalated"] = True
        
        # Apply policy-based response
        if policy == DriftPolicy.STRICT:
            # Immediate halt
            node.escalation_flags.append("drift_halt")
            node.is_active_flag = False
            result["action"] = "halted"
            
        elif policy == DriftPolicy.ADAPTIVE:
            # Increase monitoring frequency
            node.check_in_interval = max(5, node.check_in_interval // 2)
            node.escalation_flags.append("drift_monitored")
            result["action"] = "monitoring_increased"
            
        elif policy == DriftPolicy.ESCALATE:
            # Notify parent
            node.escalation_flags.append("drift_detected")
            node.anomaly_count += 1
            if node.parent_id:
                result["action"] = f"escalated_to_{node.parent_id}"
            else:
                result["action"] = "escalated_to_root"
                
        elif policy == DriftPolicy.ARCHIVE:
            # Log and continue
            node.anomaly_history.append({
                "type": "drift_archived",
                "drift": drift,
                "timestamp": datetime.utcnow().isoformat()
            })
            result["action"] = "archived"
    
    return result


def track_lineage(
    tree: CapsuleLineageTree,
    capsule_id: str
) -> Dict[str, Any]:
    """
    DSL Predicate: Track complete lineage for a capsule.
    
    Returns comprehensive lineage information including:
    - Ancestry chain (GID, PID, CID structure)
    - Generation depth
    - Inheritance path
    - Drift trajectory
    
    Args:
        tree: The lineage tree
        capsule_id: Capsule to track
    
    Returns:
        Complete lineage tracking data
    """
    node = tree.get_node(capsule_id)
    if node is None:
        return {"found": False, "capsule_id": capsule_id}
    
    ancestors = tree.get_ancestors(capsule_id)
    lineage_path = tree.get_lineage_path(capsule_id)
    
    return {
        "found": True,
        "gid": capsule_id,  # Global ID
        "pid": node.parent_id,  # Parent ID
        "cid": node.children_ids,  # Children IDs
        "generation": node.generation,
        "role": node.role.value,
        "lineage_path": lineage_path,
        "ancestor_chain": [a.capsule_id for a in ancestors],
        "drift_from_root": tree.compute_lineage_drift(capsule_id),
        "inheritance_mode": node.inheritance_mode.value,
        "inherited_traits": node.inherited_traits,
        "is_active": node.is_active(),
        "check_in_status": node.check_in_status.value
    }


def calculate_drift(tree: CapsuleLineageTree, capsule_id: str) -> float:
    """
    DSL Predicate: Calculate drift score for capsule.
    
    Wrapper for lineage drift computation.
    """
    return tree.compute_lineage_drift(capsule_id)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=== Capsule Lineage System Demo ===\n")
    
    # Create tree
    tree = CapsuleLineageTree("TREE-001")
    
    # Create root capsule
    np.random.seed(42)
    root_fp = CapsuleFingerprint(
        capsule_id="CAP-ROOT",
        vector=np.random.randn(32),
        traits={'stability': 0.9, 'exploration': 0.3, 'accuracy': 0.85}
    )
    
    root = tree.add_root("CAP-ROOT", root_fp)
    print(f"Created root: {root.capsule_id} (gen {root.generation})")
    
    # Spawn children
    child1 = tree.spawn_child("CAP-ROOT", "CAP-001", InheritanceMode.FULL)
    child2 = tree.spawn_child("CAP-ROOT", "CAP-002", InheritanceMode.MUTATED, mutation_rate=0.2)
    print(f"Spawned children: {child1.capsule_id}, {child2.capsule_id}")
    
    # Spawn grandchildren
    grandchild1 = tree.spawn_child("CAP-001", "CAP-001-A", InheritanceMode.FULL)
    grandchild2 = tree.spawn_child("CAP-001", "CAP-001-B", InheritanceMode.MUTATED, mutation_rate=0.1)
    print(f"Spawned grandchildren: {grandchild1.capsule_id}, {grandchild2.capsule_id}")
    
    # Test ancestry
    print(f"\nAncestors of CAP-001-A: {[a.capsule_id for a in tree.get_ancestors('CAP-001-A')]}")
    print(f"Descendants of CAP-ROOT: {[d.capsule_id for d in tree.get_descendants('CAP-ROOT')]}")
    print(f"Lineage path to CAP-001-B: {tree.get_lineage_path('CAP-001-B')}")
    
    # Test drift detection
    print(f"\nLineage drift for CAP-001-A: {tree.compute_lineage_drift('CAP-001-A'):.4f}")
    print(f"Lineage drift for CAP-002: {tree.compute_lineage_drift('CAP-002'):.4f}")
    
    # Test DSL predicates
    print(f"\nDSL Predicates:")
    print(f"  lineage_check('CAP-001'): {lineage_check(tree, 'CAP-001')}")
    print(f"  capsule_family_drift('CAP-001'): {capsule_family_drift(tree, 'CAP-001')}")
    print(f"  check_lineage_trajectory('CAP-002', 0.5): {check_lineage_trajectory(tree, 'CAP-002', 0.5)}")
    print(f"  lineage_depth('CAP-001-A'): {lineage_depth(tree, 'CAP-001-A')}")
    
    # Statistics
    print(f"\nTree Statistics:")
    stats = tree.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
