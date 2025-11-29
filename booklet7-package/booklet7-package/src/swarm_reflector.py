"""
RSCS-Q Booklet 7: Swarm Reflector System
=========================================

Implements autonomous swarm architectures for reflective cognition:
- Swarm archetypes (Verifier, Explorer, Reflector, Archivist)
- Swarm lifecycle management (spawn, expand, re-align, converge)
- Inter-capsule communication protocols
- Drift-triggered swarm coordination

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib
import json

from capsule_lineage import (
    CapsuleFingerprint, LineageNode, CapsuleLineageTree,
    LineageRole, CheckInStatus, InheritanceMode
)


# =============================================================================
# ENUMS
# =============================================================================

class SwarmType(Enum):
    """Swarm archetype classification"""
    VERIFIER = "verifier"       # Validates capsule outputs and consensus
    EXPLORER = "explorer"       # Discovers new patterns and anomalies
    REFLECTOR = "reflector"     # Performs self-comparison and drift analysis
    ARCHIVIST = "archivist"     # Maintains history and pattern memory
    SYNTHESIZER = "synthesizer" # Combines insights across capsules


class SwarmPhase(Enum):
    """Swarm lifecycle phase"""
    DORMANT = "dormant"         # Not active
    SPAWNING = "spawning"       # Being created
    ACTIVE = "active"           # Normal operation
    EXPANDING = "expanding"     # Growing membership
    REALIGNING = "realigning"   # Adjusting to drift
    CONVERGING = "converging"   # Reaching consensus
    TERMINATED = "terminated"   # Shut down


class MessageType(Enum):
    """Inter-capsule message types"""
    PING = "ping"               # Health check
    COMPARE = "compare"         # Request comparison
    ALERT = "alert"             # Anomaly notification
    CONSENSUS = "consensus"     # Vote request
    BROADCAST = "broadcast"     # General announcement
    HANDOFF = "handoff"         # Transfer responsibility


class TriggerCondition(Enum):
    """Conditions that trigger swarm actions"""
    DRIFT_DETECTED = "drift_detected"
    ANOMALY_CLUSTER = "anomaly_cluster"
    FINGERPRINT_DEVIATION = "fingerprint_deviation"
    RUBRIC_DIVERGENCE = "rubric_divergence"
    LINEAGE_BREAK = "lineage_break"
    CONSENSUS_FAILURE = "consensus_failure"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


# =============================================================================
# SWARM MESSAGE
# =============================================================================

@dataclass
class SwarmMessage:
    """
    Message passed between capsules in a swarm.
    """
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_ids: List[str]  # Empty = broadcast to all
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: int = 3  # Time-to-live (hops)
    requires_ack: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_ids': self.recipient_ids,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'ttl': self.ttl
        }


# =============================================================================
# SWARM MEMBER
# =============================================================================

@dataclass
class SwarmMember:
    """
    A capsule participating in a swarm.
    """
    capsule_id: str
    fingerprint: CapsuleFingerprint
    role: str = "member"  # member, leader, observer
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    contribution_score: float = 0.0
    message_queue: List[SwarmMessage] = field(default_factory=list)
    
    def receive_message(self, message: SwarmMessage) -> None:
        """Queue a message for processing"""
        self.message_queue.append(message)
        self.last_active = datetime.utcnow()
    
    def process_messages(self) -> List[SwarmMessage]:
        """Process and clear message queue"""
        messages = self.message_queue.copy()
        self.message_queue.clear()
        return messages
    
    def is_active(self, timeout_seconds: int = 60) -> bool:
        """Check if member is still active"""
        return (datetime.utcnow() - self.last_active).total_seconds() < timeout_seconds


# =============================================================================
# SWARM
# =============================================================================

@dataclass
class Swarm:
    """
    An autonomous swarm of cooperating capsules.
    
    Swarms coordinate reflective behavior:
    - Pattern discovery
    - Drift consensus
    - Anomaly investigation
    - Memory consolidation
    """
    swarm_id: str
    swarm_type: SwarmType
    phase: SwarmPhase = SwarmPhase.DORMANT
    
    # Membership
    members: Dict[str, SwarmMember] = field(default_factory=dict)
    leader_id: Optional[str] = None
    max_members: int = 50
    min_members: int = 3
    
    # Configuration
    consensus_threshold: float = 0.67  # 2/3 majority
    drift_threshold: float = 0.3
    check_in_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    
    # State
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    trigger_condition: Optional[TriggerCondition] = None
    
    # History
    message_log: List[SwarmMessage] = field(default_factory=list)
    consensus_history: List[Dict[str, Any]] = field(default_factory=list)
    discoveries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    total_messages: int = 0
    consensus_count: int = 0
    anomalies_detected: int = 0
    
    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------
    
    def spawn(self, trigger: TriggerCondition = TriggerCondition.MANUAL) -> None:
        """Activate the swarm"""
        self.phase = SwarmPhase.SPAWNING
        self.trigger_condition = trigger
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        if len(self.members) >= self.min_members:
            self.phase = SwarmPhase.ACTIVE
            self._elect_leader()
    
    def expand(self, member: SwarmMember) -> bool:
        """Add a new member to the swarm"""
        if len(self.members) >= self.max_members:
            return False
        
        if member.capsule_id in self.members:
            return False
        
        self.members[member.capsule_id] = member
        self.last_activity = datetime.utcnow()
        
        if self.phase == SwarmPhase.SPAWNING and len(self.members) >= self.min_members:
            self.phase = SwarmPhase.ACTIVE
            self._elect_leader()
        
        return True
    
    def remove_member(self, capsule_id: str) -> bool:
        """Remove a member from the swarm"""
        if capsule_id not in self.members:
            return False
        
        del self.members[capsule_id]
        
        # Re-elect leader if needed
        if capsule_id == self.leader_id:
            self._elect_leader()
        
        # Check minimum membership
        if len(self.members) < self.min_members:
            self.phase = SwarmPhase.CONVERGING
        
        return True
    
    def realign(self) -> None:
        """Enter realignment phase for drift correction"""
        self.phase = SwarmPhase.REALIGNING
        self.last_activity = datetime.utcnow()
        
        # Broadcast realignment message
        self.broadcast(SwarmMessage(
            message_id=f"realign-{self.swarm_id}-{datetime.utcnow().timestamp()}",
            message_type=MessageType.BROADCAST,
            sender_id=self.leader_id or "system",
            recipient_ids=[],
            payload={'action': 'realign', 'reason': 'drift_detected'}
        ))
    
    def converge(self) -> Dict[str, Any]:
        """
        Enter convergence phase and compute final consensus.
        
        Returns:
            Consensus result dictionary
        """
        self.phase = SwarmPhase.CONVERGING
        self.last_activity = datetime.utcnow()
        
        # Collect fingerprints for consensus
        fingerprints = [m.fingerprint for m in self.members.values()]
        
        if not fingerprints:
            return {'status': 'no_members', 'consensus': None}
        
        # Compute centroid fingerprint
        vectors = np.array([fp.vector for fp in fingerprints])
        centroid = np.mean(vectors, axis=0)
        
        # Compute agreement scores
        agreements = []
        for fp in fingerprints:
            from capsule_lineage import cosine_similarity
            sim = cosine_similarity(fp.vector, centroid)
            agreements.append(sim)
        
        avg_agreement = np.mean(agreements)
        consensus_reached = avg_agreement >= self.consensus_threshold
        
        result = {
            'status': 'consensus' if consensus_reached else 'no_consensus',
            'agreement': float(avg_agreement),
            'threshold': self.consensus_threshold,
            'member_count': len(self.members),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.consensus_history.append(result)
        self.consensus_count += 1
        
        return result
    
    def terminate(self) -> None:
        """Shut down the swarm"""
        self.phase = SwarmPhase.TERMINATED
        self.last_activity = datetime.utcnow()
    
    def _elect_leader(self) -> None:
        """Elect a leader based on contribution score"""
        if not self.members:
            self.leader_id = None
            return
        
        # Elect member with highest contribution score
        leader = max(self.members.values(), key=lambda m: m.contribution_score)
        self.leader_id = leader.capsule_id
        leader.role = "leader"
    
    # -------------------------------------------------------------------------
    # COMMUNICATION
    # -------------------------------------------------------------------------
    
    def broadcast(self, message: SwarmMessage) -> int:
        """
        Broadcast message to all members.
        
        Returns:
            Number of members reached
        """
        count = 0
        for member in self.members.values():
            if member.capsule_id != message.sender_id:
                member.receive_message(message)
                count += 1
        
        self.message_log.append(message)
        self.total_messages += 1
        self.last_activity = datetime.utcnow()
        
        return count
    
    def send_to(self, message: SwarmMessage) -> bool:
        """Send message to specific recipients"""
        sent = False
        for recipient_id in message.recipient_ids:
            member = self.members.get(recipient_id)
            if member:
                member.receive_message(message)
                sent = True
        
        if sent:
            self.message_log.append(message)
            self.total_messages += 1
            self.last_activity = datetime.utcnow()
        
        return sent
    
    def ping_all(self) -> Dict[str, bool]:
        """Ping all members and return response status"""
        results = {}
        ping_msg = SwarmMessage(
            message_id=f"ping-{datetime.utcnow().timestamp()}",
            message_type=MessageType.PING,
            sender_id=self.leader_id or "system",
            recipient_ids=[],
            payload={'timestamp': datetime.utcnow().isoformat()},
            requires_ack=True
        )
        
        for capsule_id, member in self.members.items():
            results[capsule_id] = member.is_active()
        
        self.broadcast(ping_msg)
        return results
    
    # -------------------------------------------------------------------------
    # REFLECTION OPERATIONS
    # -------------------------------------------------------------------------
    
    def compare_fingerprints(self) -> Dict[str, float]:
        """
        Compare all member fingerprints pairwise.
        
        Returns:
            Dictionary mapping pair keys to similarity scores
        """
        comparisons = {}
        members = list(self.members.values())
        
        for i, m1 in enumerate(members):
            for m2 in members[i+1:]:
                key = f"{m1.capsule_id}:{m2.capsule_id}"
                comparisons[key] = m1.fingerprint.similarity(m2.fingerprint)
        
        return comparisons
    
    def detect_outliers(self, threshold: float = 0.5) -> List[str]:
        """
        Detect members whose fingerprints deviate from swarm centroid.
        
        Returns:
            List of outlier capsule IDs
        """
        if len(self.members) < 2:
            return []
        
        # Compute centroid
        vectors = np.array([m.fingerprint.vector for m in self.members.values()])
        centroid = np.mean(vectors, axis=0)
        
        # Find outliers
        outliers = []
        for capsule_id, member in self.members.items():
            from capsule_lineage import cosine_similarity
            sim = cosine_similarity(member.fingerprint.vector, centroid)
            if sim < threshold:
                outliers.append(capsule_id)
        
        return outliers
    
    def request_consensus(self, topic: str, options: List[str]) -> Dict[str, Any]:
        """
        Request consensus vote from all members.
        
        Returns:
            Voting result
        """
        # Simulate voting based on fingerprint similarity to options
        votes = defaultdict(int)
        
        for member in self.members.values():
            # Simple voting: random weighted by contribution
            vote_idx = int(np.random.random() * len(options))
            votes[options[vote_idx]] += 1
        
        total = sum(votes.values())
        winner = max(votes.items(), key=lambda x: x[1]) if votes else (None, 0)
        
        result = {
            'topic': topic,
            'options': options,
            'votes': dict(votes),
            'winner': winner[0],
            'ratio': winner[1] / total if total > 0 else 0,
            'consensus': winner[1] / total >= self.consensus_threshold if total > 0 else False,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.consensus_history.append(result)
        return result
    
    def log_discovery(self, discovery: Dict[str, Any]) -> None:
        """Log a pattern discovery"""
        discovery['timestamp'] = datetime.utcnow().isoformat()
        discovery['swarm_id'] = self.swarm_id
        self.discoveries.append(discovery)
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm statistics"""
        active_members = sum(1 for m in self.members.values() if m.is_active())
        
        return {
            'swarm_id': self.swarm_id,
            'swarm_type': self.swarm_type.value,
            'phase': self.phase.value,
            'total_members': len(self.members),
            'active_members': active_members,
            'leader_id': self.leader_id,
            'total_messages': self.total_messages,
            'consensus_count': self.consensus_count,
            'discoveries': len(self.discoveries),
            'created_at': self.created_at.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export swarm to dictionary"""
        return {
            **self.get_statistics(),
            'members': {cid: {'role': m.role, 'contribution': m.contribution_score}
                       for cid, m in self.members.items()},
            'consensus_history': self.consensus_history[-10:],  # Last 10
            'discoveries': self.discoveries[-10:]  # Last 10
        }


# =============================================================================
# SWARM COORDINATOR
# =============================================================================

class SwarmCoordinator:
    """
    Manages multiple swarms and coordinates inter-swarm communication.
    
    Provides:
    - Swarm lifecycle management
    - Trigger-based swarm spawning
    - Cross-swarm pattern discovery
    - Resource throttling
    """
    
    def __init__(self, coordinator_id: str):
        self.coordinator_id = coordinator_id
        self.swarms: Dict[str, Swarm] = {}
        self.lineage_tree: Optional[CapsuleLineageTree] = None
        
        # Throttling
        self.max_active_swarms: int = 10
        self.spawn_cooldown: timedelta = timedelta(seconds=10)
        self.last_spawn: Optional[datetime] = None
        
        # Event handlers
        self._trigger_handlers: Dict[TriggerCondition, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.total_spawned = 0
        self.total_terminated = 0
        self.created_at = datetime.utcnow()
    
    def set_lineage_tree(self, tree: CapsuleLineageTree) -> None:
        """Connect to a lineage tree for capsule tracking"""
        self.lineage_tree = tree
    
    # -------------------------------------------------------------------------
    # SWARM MANAGEMENT
    # -------------------------------------------------------------------------
    
    def create_swarm(
        self,
        swarm_type: SwarmType,
        trigger: TriggerCondition = TriggerCondition.MANUAL,
        **kwargs
    ) -> Optional[Swarm]:
        """
        Create and spawn a new swarm.
        
        Returns:
            New Swarm or None if throttled
        """
        # Check throttling
        active_count = sum(1 for s in self.swarms.values() 
                         if s.phase not in [SwarmPhase.DORMANT, SwarmPhase.TERMINATED])
        
        if active_count >= self.max_active_swarms:
            return None
        
        if self.last_spawn and (datetime.utcnow() - self.last_spawn) < self.spawn_cooldown:
            return None
        
        # Create swarm
        swarm_id = f"SWM-{swarm_type.value[:3].upper()}-{len(self.swarms):04d}"
        swarm = Swarm(
            swarm_id=swarm_id,
            swarm_type=swarm_type,
            **kwargs
        )
        
        self.swarms[swarm_id] = swarm
        self.total_spawned += 1
        self.last_spawn = datetime.utcnow()
        
        swarm.spawn(trigger)
        
        return swarm
    
    def terminate_swarm(self, swarm_id: str) -> bool:
        """Terminate a swarm"""
        swarm = self.swarms.get(swarm_id)
        if swarm is None:
            return False
        
        swarm.terminate()
        self.total_terminated += 1
        return True
    
    def get_swarm(self, swarm_id: str) -> Optional[Swarm]:
        """Get a swarm by ID"""
        return self.swarms.get(swarm_id)
    
    def get_active_swarms(self) -> List[Swarm]:
        """Get all active swarms"""
        return [s for s in self.swarms.values() 
                if s.phase not in [SwarmPhase.DORMANT, SwarmPhase.TERMINATED]]
    
    # -------------------------------------------------------------------------
    # TRIGGER SYSTEM
    # -------------------------------------------------------------------------
    
    def register_trigger(
        self,
        condition: TriggerCondition,
        handler: Callable[[TriggerCondition, Dict[str, Any]], None]
    ) -> None:
        """Register a handler for a trigger condition"""
        self._trigger_handlers[condition].append(handler)
    
    def fire_trigger(
        self,
        condition: TriggerCondition,
        context: Dict[str, Any]
    ) -> List[Swarm]:
        """
        Fire a trigger and potentially spawn swarms.
        
        Returns:
            List of swarms spawned or activated
        """
        spawned = []
        
        # Call registered handlers
        for handler in self._trigger_handlers[condition]:
            handler(condition, context)
        
        # Auto-spawn swarms based on condition
        swarm_type_map = {
            TriggerCondition.DRIFT_DETECTED: SwarmType.REFLECTOR,
            TriggerCondition.ANOMALY_CLUSTER: SwarmType.EXPLORER,
            TriggerCondition.FINGERPRINT_DEVIATION: SwarmType.VERIFIER,
            TriggerCondition.RUBRIC_DIVERGENCE: SwarmType.VERIFIER,
            TriggerCondition.LINEAGE_BREAK: SwarmType.ARCHIVIST,
            TriggerCondition.CONSENSUS_FAILURE: SwarmType.SYNTHESIZER,
        }
        
        swarm_type = swarm_type_map.get(condition)
        if swarm_type:
            swarm = self.create_swarm(swarm_type, trigger=condition)
            if swarm:
                spawned.append(swarm)
        
        return spawned
    
    # -------------------------------------------------------------------------
    # CROSS-SWARM OPERATIONS
    # -------------------------------------------------------------------------
    
    def broadcast_to_all_swarms(self, message: SwarmMessage) -> int:
        """Broadcast message to all active swarms"""
        count = 0
        for swarm in self.get_active_swarms():
            count += swarm.broadcast(message)
        return count
    
    def aggregate_discoveries(self) -> List[Dict[str, Any]]:
        """Collect discoveries from all swarms"""
        all_discoveries = []
        for swarm in self.swarms.values():
            all_discoveries.extend(swarm.discoveries)
        
        # Sort by timestamp
        all_discoveries.sort(key=lambda d: d.get('timestamp', ''), reverse=True)
        return all_discoveries
    
    def compute_global_consensus(self, topic: str) -> Dict[str, Any]:
        """Request consensus across all active swarms"""
        results = []
        
        for swarm in self.get_active_swarms():
            result = swarm.converge()
            results.append({
                'swarm_id': swarm.swarm_id,
                'agreement': result.get('agreement', 0)
            })
        
        if not results:
            return {'status': 'no_swarms', 'global_agreement': 0}
        
        global_agreement = np.mean([r['agreement'] for r in results])
        
        return {
            'topic': topic,
            'swarm_results': results,
            'global_agreement': float(global_agreement),
            'consensus': global_agreement >= 0.67,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            'coordinator_id': self.coordinator_id,
            'total_swarms': len(self.swarms),
            'active_swarms': len(self.get_active_swarms()),
            'total_spawned': self.total_spawned,
            'total_terminated': self.total_terminated,
            'max_active_swarms': self.max_active_swarms,
            'created_at': self.created_at.isoformat()
        }


# =============================================================================
# DSL PREDICATES
# =============================================================================

def swarm_consensus_reached(swarm: Swarm, threshold: float = 0.67) -> bool:
    """DSL Predicate: Check if swarm has reached consensus"""
    if not swarm.consensus_history:
        return False
    
    latest = swarm.consensus_history[-1]
    return latest.get('agreement', 0) >= threshold


def swarm_has_outliers(swarm: Swarm, threshold: float = 0.5) -> bool:
    """DSL Predicate: Check if swarm has outlier members"""
    outliers = swarm.detect_outliers(threshold)
    return len(outliers) > 0


def trigger_reflective_response(
    coordinator: SwarmCoordinator,
    condition: TriggerCondition,
    context: Dict[str, Any]
) -> bool:
    """DSL Predicate: Trigger a reflective swarm response"""
    swarms = coordinator.fire_trigger(condition, context)
    return len(swarms) > 0


def reflector_trigger(swarm: Swarm, drift_threshold: float = 0.3) -> bool:
    """DSL Predicate: Check if reflector swarm should activate"""
    if swarm.swarm_type != SwarmType.REFLECTOR:
        return False
    
    outliers = swarm.detect_outliers(1.0 - drift_threshold)
    return len(outliers) > 0


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=== Swarm Reflector System Demo ===\n")
    
    # Create coordinator
    coordinator = SwarmCoordinator("COORD-001")
    
    # Create lineage tree
    tree = CapsuleLineageTree("TREE-001")
    coordinator.set_lineage_tree(tree)
    
    # Create an explorer swarm
    np.random.seed(42)
    explorer_swarm = coordinator.create_swarm(
        SwarmType.EXPLORER,
        trigger=TriggerCondition.ANOMALY_CLUSTER
    )
    print(f"Created swarm: {explorer_swarm.swarm_id} ({explorer_swarm.swarm_type.value})")
    
    # Add members
    for i in range(5):
        fp = CapsuleFingerprint(
            capsule_id=f"CAP-{i:03d}",
            vector=np.random.randn(32),
            traits={'exploration': np.random.random()}
        )
        member = SwarmMember(capsule_id=f"CAP-{i:03d}", fingerprint=fp)
        member.contribution_score = np.random.random()
        explorer_swarm.expand(member)
    
    print(f"Added {len(explorer_swarm.members)} members")
    print(f"Leader: {explorer_swarm.leader_id}")
    print(f"Phase: {explorer_swarm.phase.value}")
    
    # Test communication
    msg = SwarmMessage(
        message_id="MSG-001",
        message_type=MessageType.BROADCAST,
        sender_id=explorer_swarm.leader_id,
        recipient_ids=[],
        payload={'test': 'hello swarm'}
    )
    reached = explorer_swarm.broadcast(msg)
    print(f"\nBroadcast reached {reached} members")
    
    # Test comparison
    comparisons = explorer_swarm.compare_fingerprints()
    print(f"Fingerprint comparisons: {len(comparisons)} pairs")
    avg_sim = np.mean(list(comparisons.values()))
    print(f"Average similarity: {avg_sim:.4f}")
    
    # Test outlier detection
    outliers = explorer_swarm.detect_outliers(threshold=0.3)
    print(f"Outliers detected: {len(outliers)}")
    
    # Test consensus
    consensus = explorer_swarm.converge()
    print(f"\nConsensus result: {consensus['status']}")
    print(f"Agreement: {consensus['agreement']:.4f}")
    
    # Coordinator stats
    print(f"\nCoordinator Statistics:")
    stats = coordinator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
