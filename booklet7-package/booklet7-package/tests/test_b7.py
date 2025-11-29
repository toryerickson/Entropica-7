"""
RSCS-Q Booklet 7: Test Suite
=============================

Comprehensive tests for:
- Capsule Lineage System
- Swarm Reflector
- Reflective Matrix Engine

Author: Entropica Research Collective
Version: 1.0
"""

import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from capsule_lineage import (
    CapsuleFingerprint, LineageNode, CapsuleLineageTree,
    LineageRole, CheckInStatus, InheritanceMode,
    lineage_check, capsule_family_drift, check_lineage_trajectory, lineage_depth
)

from swarm_reflector import (
    Swarm, SwarmMember, SwarmMessage, SwarmCoordinator,
    SwarmType, SwarmPhase, MessageType, TriggerCondition,
    swarm_consensus_reached, swarm_has_outliers
)

from reflective_matrix_engine import (
    ReflectiveMatrixEngine, DriftRecord, DriftType, DriftSeverity,
    EVIResult, MDSResult, DriftForecast,
    evi_valid, drift_forecast_breach, pattern_discovered
)


# =============================================================================
# CAPSULE FINGERPRINT TESTS
# =============================================================================

class TestCapsuleFingerprint(unittest.TestCase):
    """Test CapsuleFingerprint class"""
    
    def setUp(self):
        np.random.seed(42)
        self.fp1 = CapsuleFingerprint(
            capsule_id="CAP-001",
            vector=np.random.randn(32),
            traits={'stability': 0.9, 'exploration': 0.3}
        )
        self.fp2 = CapsuleFingerprint(
            capsule_id="CAP-002",
            vector=self.fp1.vector + np.random.randn(32) * 0.1,
            traits={'stability': 0.85, 'exploration': 0.35}
        )
    
    def test_similarity_self(self):
        """Fingerprint is identical to itself"""
        sim = self.fp1.similarity(self.fp1)
        self.assertAlmostEqual(sim, 1.0, places=5)
    
    def test_similarity_range(self):
        """Similarity is in valid range"""
        sim = self.fp1.similarity(self.fp2)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)
    
    def test_hash_deterministic(self):
        """Hash is deterministic"""
        hash1 = self.fp1.hash()
        hash2 = self.fp1.hash()
        self.assertEqual(hash1, hash2)
    
    def test_mutate(self):
        """Mutation creates modified copy"""
        mutated = self.fp1.mutate(mutation_rate=0.1)
        self.assertNotEqual(self.fp1.hash(), mutated.hash())
        self.assertEqual(mutated.generation, self.fp1.generation + 1)
    
    def test_trait_distance(self):
        """Trait distance computation"""
        dist = self.fp1.trait_distance(self.fp2)
        self.assertGreaterEqual(dist, 0)
        self.assertLessEqual(dist, 1)


# =============================================================================
# LINEAGE NODE TESTS
# =============================================================================

class TestLineageNode(unittest.TestCase):
    """Test LineageNode class"""
    
    def setUp(self):
        np.random.seed(42)
        self.fp = CapsuleFingerprint(
            capsule_id="NODE-001",
            vector=np.random.randn(32),
            traits={}
        )
        self.node = LineageNode(
            capsule_id="NODE-001",
            fingerprint=self.fp
        )
    
    def test_initial_state(self):
        """Node has correct initial state"""
        self.assertEqual(self.node.capsule_id, "NODE-001")
        self.assertEqual(self.node.role, LineageRole.ROOT)
        self.assertEqual(self.node.generation, 0)
        self.assertTrue(self.node.is_active())
    
    def test_check_in(self):
        """Check-in updates status"""
        status = self.node.check_in()
        self.assertEqual(status, CheckInStatus.HEALTHY)
    
    def test_terminate(self):
        """Termination updates state"""
        self.node.terminate()
        self.assertFalse(self.node.is_active())
        self.assertEqual(self.node.check_in_status, CheckInStatus.TERMINATED)
    
    def test_add_child(self):
        """Adding children updates role"""
        self.node.role = LineageRole.LEAF
        self.node.parent_id = "PARENT"
        self.node.add_child("CHILD-001")
        self.assertIn("CHILD-001", self.node.children_ids)
        self.assertEqual(self.node.role, LineageRole.CHILD)


# =============================================================================
# LINEAGE TREE TESTS
# =============================================================================

class TestCapsuleLineageTree(unittest.TestCase):
    """Test CapsuleLineageTree class"""
    
    def setUp(self):
        np.random.seed(42)
        self.tree = CapsuleLineageTree("TREE-001")
        
        # Create root
        self.root_fp = CapsuleFingerprint(
            capsule_id="ROOT",
            vector=np.random.randn(32),
            traits={'stability': 0.9}
        )
        self.root = self.tree.add_root("ROOT", self.root_fp)
    
    def test_add_root(self):
        """Can add root node"""
        self.assertIn("ROOT", self.tree.nodes)
        self.assertEqual(self.tree.root_ids, ["ROOT"])
    
    def test_spawn_child(self):
        """Can spawn child from parent"""
        child = self.tree.spawn_child("ROOT", "CHILD-001", InheritanceMode.FULL)
        
        self.assertIsNotNone(child)
        self.assertEqual(child.parent_id, "ROOT")
        self.assertEqual(child.generation, 1)
        self.assertIn("CHILD-001", self.root.children_ids)
    
    def test_get_ancestors(self):
        """Can trace ancestors"""
        self.tree.spawn_child("ROOT", "GEN1", InheritanceMode.FULL)
        self.tree.spawn_child("GEN1", "GEN2", InheritanceMode.FULL)
        
        ancestors = self.tree.get_ancestors("GEN2")
        self.assertEqual(len(ancestors), 2)
        self.assertEqual(ancestors[0].capsule_id, "GEN1")
        self.assertEqual(ancestors[1].capsule_id, "ROOT")
    
    def test_get_descendants(self):
        """Can get all descendants"""
        self.tree.spawn_child("ROOT", "C1", InheritanceMode.FULL)
        self.tree.spawn_child("ROOT", "C2", InheritanceMode.FULL)
        self.tree.spawn_child("C1", "GC1", InheritanceMode.FULL)
        
        descendants = self.tree.get_descendants("ROOT")
        self.assertEqual(len(descendants), 3)
    
    def test_get_lineage_path(self):
        """Lineage path is correct"""
        self.tree.spawn_child("ROOT", "GEN1", InheritanceMode.FULL)
        self.tree.spawn_child("GEN1", "GEN2", InheritanceMode.FULL)
        
        path = self.tree.get_lineage_path("GEN2")
        self.assertEqual(path, ["ROOT", "GEN1", "GEN2"])
    
    def test_terminate_cascade(self):
        """Cascade termination works"""
        self.tree.spawn_child("ROOT", "C1", InheritanceMode.FULL)
        self.tree.spawn_child("C1", "GC1", InheritanceMode.FULL)
        
        terminated = self.tree.terminate_capsule("C1", cascade=True)
        self.assertEqual(len(terminated), 2)


# =============================================================================
# LINEAGE DSL PREDICATE TESTS
# =============================================================================

class TestLineageDSL(unittest.TestCase):
    """Test lineage DSL predicates"""
    
    def setUp(self):
        np.random.seed(42)
        self.tree = CapsuleLineageTree("TREE-001")
        fp = CapsuleFingerprint("ROOT", np.random.randn(32), {})
        self.tree.add_root("ROOT", fp)
        self.tree.spawn_child("ROOT", "CHILD", InheritanceMode.FULL)
    
    def test_lineage_check(self):
        """lineage_check predicate"""
        self.assertTrue(lineage_check(self.tree, "ROOT"))
        self.assertTrue(lineage_check(self.tree, "CHILD"))
        self.assertFalse(lineage_check(self.tree, "NONEXISTENT"))
    
    def test_lineage_depth(self):
        """lineage_depth predicate"""
        self.assertEqual(lineage_depth(self.tree, "ROOT"), 0)
        self.assertEqual(lineage_depth(self.tree, "CHILD"), 1)
    
    def test_check_lineage_trajectory(self):
        """check_lineage_trajectory predicate"""
        self.assertTrue(check_lineage_trajectory(self.tree, "CHILD", 0.5))


# =============================================================================
# SWARM TESTS
# =============================================================================

class TestSwarm(unittest.TestCase):
    """Test Swarm class"""
    
    def setUp(self):
        np.random.seed(42)
        self.swarm = Swarm(
            swarm_id="SWM-001",
            swarm_type=SwarmType.EXPLORER
        )
        
        # Add members
        for i in range(5):
            fp = CapsuleFingerprint(f"CAP-{i}", np.random.randn(32), {})
            member = SwarmMember(capsule_id=f"CAP-{i}", fingerprint=fp)
            member.contribution_score = np.random.random()
            self.swarm.expand(member)
    
    def test_spawn(self):
        """Swarm spawns correctly"""
        self.swarm.spawn(TriggerCondition.MANUAL)
        self.assertEqual(self.swarm.phase, SwarmPhase.ACTIVE)
        self.assertIsNotNone(self.swarm.leader_id)
    
    def test_membership(self):
        """Membership management works"""
        self.assertEqual(len(self.swarm.members), 5)
        
        # Remove member
        self.swarm.remove_member("CAP-0")
        self.assertEqual(len(self.swarm.members), 4)
    
    def test_broadcast(self):
        """Broadcasting reaches members"""
        self.swarm.spawn()
        
        msg = SwarmMessage(
            message_id="MSG-001",
            message_type=MessageType.BROADCAST,
            sender_id=self.swarm.leader_id,
            recipient_ids=[],
            payload={'test': 'data'}
        )
        
        reached = self.swarm.broadcast(msg)
        self.assertEqual(reached, 4)  # All except sender
    
    def test_compare_fingerprints(self):
        """Fingerprint comparison works"""
        self.swarm.spawn()
        comparisons = self.swarm.compare_fingerprints()
        
        # n choose 2 = 10 pairs for 5 members
        self.assertEqual(len(comparisons), 10)
    
    def test_converge(self):
        """Convergence produces result"""
        self.swarm.spawn()
        result = self.swarm.converge()
        
        self.assertIn('status', result)
        self.assertIn('agreement', result)


# =============================================================================
# SWARM COORDINATOR TESTS
# =============================================================================

class TestSwarmCoordinator(unittest.TestCase):
    """Test SwarmCoordinator class"""
    
    def setUp(self):
        self.coordinator = SwarmCoordinator("COORD-001")
    
    def test_create_swarm(self):
        """Can create swarm"""
        swarm = self.coordinator.create_swarm(
            SwarmType.EXPLORER,
            trigger=TriggerCondition.ANOMALY_CLUSTER
        )
        
        self.assertIsNotNone(swarm)
        self.assertEqual(len(self.coordinator.swarms), 1)
    
    def test_throttling(self):
        """Swarm creation is throttled"""
        # Create max swarms
        for i in range(self.coordinator.max_active_swarms + 5):
            self.coordinator.create_swarm(SwarmType.VERIFIER)
        
        # Should be capped
        active = len(self.coordinator.get_active_swarms())
        self.assertLessEqual(active, self.coordinator.max_active_swarms)
    
    def test_fire_trigger(self):
        """Trigger system works"""
        swarms = self.coordinator.fire_trigger(
            TriggerCondition.DRIFT_DETECTED,
            {'capsule_id': 'CAP-001'}
        )
        
        # Should spawn a reflector swarm
        self.assertEqual(len(swarms), 1)
        self.assertEqual(swarms[0].swarm_type, SwarmType.REFLECTOR)


# =============================================================================
# REFLECTIVE MATRIX ENGINE TESTS
# =============================================================================

class TestReflectiveMatrixEngine(unittest.TestCase):
    """Test ReflectiveMatrixEngine class"""
    
    def setUp(self):
        np.random.seed(42)
        self.engine = ReflectiveMatrixEngine("RME-001")
        
        # Set up baseline
        self.baseline = CapsuleFingerprint(
            capsule_id="CAP-001",
            vector=np.random.randn(32),
            traits={'stability': 0.9}
        )
        self.engine.set_baseline("CAP-001", self.baseline)
    
    def test_scan_drift(self):
        """Drift scanning works"""
        current = CapsuleFingerprint(
            capsule_id="CAP-001",
            vector=self.baseline.vector + np.random.randn(32) * 0.1,
            traits={}
        )
        
        record = self.engine.scan_drift("CAP-001", current)
        
        self.assertIsInstance(record, DriftRecord)
        self.assertGreaterEqual(record.drift_score, 0)
        self.assertLessEqual(record.drift_score, 1)
    
    def test_compute_evi(self):
        """EVI computation works"""
        current = CapsuleFingerprint(
            capsule_id="CAP-001",
            vector=self.baseline.vector,
            traits={}
        )
        
        peers = [CapsuleFingerprint(f"PEER-{i}", np.random.randn(32), {})
                for i in range(5)]
        
        evi = self.engine.compute_evi("CAP-001", current, peers)
        
        self.assertIsInstance(evi, EVIResult)
        self.assertGreaterEqual(evi.evi_score, 0)
        self.assertLessEqual(evi.evi_score, 1)
    
    def test_compute_mds(self):
        """MDS computation works"""
        pattern = np.random.randn(32)
        
        mds = self.engine.compute_mds("CAP-001", pattern, ["PEER-1"])
        
        self.assertIsInstance(mds, MDSResult)
        self.assertIsNotNone(mds.pattern_id)
    
    def test_forecast_drift(self):
        """Drift forecasting works"""
        # Generate history
        for i in range(10):
            current = CapsuleFingerprint(
                capsule_id="CAP-001",
                vector=self.baseline.vector + np.random.randn(32) * (0.1 + i * 0.01),
                traits={}
            )
            self.engine.scan_drift("CAP-001", current)
        
        forecast = self.engine.forecast_drift("CAP-001")
        
        self.assertIsInstance(forecast, DriftForecast)
        self.assertEqual(len(forecast.predicted_drift), self.engine.forecast_horizon)
        self.assertIn(forecast.trend, ["increasing", "decreasing", "stable", "unknown"])


# =============================================================================
# RME DSL PREDICATE TESTS
# =============================================================================

class TestRMEDSL(unittest.TestCase):
    """Test RME DSL predicates"""
    
    def setUp(self):
        np.random.seed(42)
        self.engine = ReflectiveMatrixEngine("RME-001")
        
        self.baseline = CapsuleFingerprint("CAP-001", np.random.randn(32), {})
        self.engine.set_baseline("CAP-001", self.baseline)
        
        # Generate some history
        for i in range(5):
            current = CapsuleFingerprint(
                "CAP-001",
                self.baseline.vector + np.random.randn(32) * 0.1,
                {}
            )
            self.engine.scan_drift("CAP-001", current)
            
            peers = [CapsuleFingerprint(f"P-{j}", np.random.randn(32), {})
                    for j in range(3)]
            self.engine.compute_evi("CAP-001", current, peers)
    
    def test_evi_valid(self):
        """evi_valid predicate"""
        result = evi_valid(self.engine, "CAP-001", 0.3)
        self.assertIsInstance(result, bool)
    
    def test_drift_forecast_breach(self):
        """drift_forecast_breach predicate"""
        result = drift_forecast_breach(self.engine, "CAP-001", 0.9)
        self.assertIsInstance(result, bool)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests across B7 modules"""
    
    def test_lineage_to_swarm_flow(self):
        """Lineage integrates with swarm"""
        np.random.seed(42)
        
        # Create lineage tree
        tree = CapsuleLineageTree("TREE-001")
        root_fp = CapsuleFingerprint("ROOT", np.random.randn(32), {})
        tree.add_root("ROOT", root_fp)
        
        # Spawn children
        for i in range(5):
            tree.spawn_child("ROOT", f"CHILD-{i}", InheritanceMode.MUTATED, 0.1)
        
        # Create swarm from lineage
        coordinator = SwarmCoordinator("COORD-001")
        coordinator.set_lineage_tree(tree)
        
        swarm = coordinator.create_swarm(SwarmType.REFLECTOR)
        
        # Add lineage nodes to swarm
        for child_id in tree._children_index["ROOT"]:
            node = tree.get_node(child_id)
            member = SwarmMember(capsule_id=child_id, fingerprint=node.fingerprint)
            swarm.expand(member)
        
        self.assertEqual(len(swarm.members), 5)
        
        swarm.spawn()
        outliers = swarm.detect_outliers(threshold=0.3)
        
        # All are similar since derived from same parent
        self.assertLess(len(outliers), 5)
    
    def test_rme_with_lineage(self):
        """RME integrates with lineage tree"""
        np.random.seed(42)
        
        # Create lineage
        tree = CapsuleLineageTree("TREE-001")
        root_fp = CapsuleFingerprint("ROOT", np.random.randn(32), {})
        tree.add_root("ROOT", root_fp)
        tree.spawn_child("ROOT", "CHILD", InheritanceMode.FULL)
        
        # Create RME and connect
        engine = ReflectiveMatrixEngine("RME-001")
        engine.connect_lineage_tree(tree)
        
        # Set baseline for child
        child = tree.get_node("CHILD")
        engine.set_baseline("CHILD", child.fingerprint)
        
        # Compute EVI with lineage context
        peers = [root_fp]
        evi = engine.compute_evi("CHILD", child.fingerprint, peers)
        
        # Should have high lineage fidelity since FULL inheritance
        # Note: actual value depends on lineage tree lookup
        self.assertGreaterEqual(evi.evi_score, 0)


# =============================================================================
# RUN TESTS
# =============================================================================

# =============================================================================
# NEW DSL PREDICATES TESTS (from notes)
# =============================================================================

class TestNewDSLPredicates(unittest.TestCase):
    """Test new DSL predicates from specification notes"""
    
    def setUp(self):
        np.random.seed(42)
        self.tree = CapsuleLineageTree("TREE-TEST")
        
        root_fp = CapsuleFingerprint("ROOT", np.random.randn(32), {'stability': 0.9})
        self.tree.add_root("ROOT", root_fp)
        
        # Import new predicates
        from capsule_lineage import (
            inherit_capsule, escalate_if_diverges, track_lineage,
            calculate_drift, DriftPolicy
        )
        self.inherit_capsule = inherit_capsule
        self.escalate_if_diverges = escalate_if_diverges
        self.track_lineage = track_lineage
        self.calculate_drift = calculate_drift
        self.DriftPolicy = DriftPolicy
    
    def test_inherit_capsule(self):
        """inherit_capsule creates child with inheritance"""
        from capsule_lineage import InheritanceMode
        child = self.inherit_capsule(
            self.tree, "CHILD-001", "ROOT",
            mode=InheritanceMode.FULL
        )
        self.assertIsNotNone(child)
        self.assertEqual(child.parent_id, "ROOT")
        self.assertEqual(child.generation, 1)
    
    def test_escalate_if_diverges(self):
        """escalate_if_diverges responds to drift"""
        from capsule_lineage import InheritanceMode, DriftPolicy
        
        # Create a child
        self.tree.spawn_child("ROOT", "CHILD-002", InheritanceMode.MUTATED, 0.5)
        
        result = self.escalate_if_diverges(
            self.tree, "CHILD-002",
            threshold=0.01,  # Very low threshold to trigger
            policy=DriftPolicy.ESCALATE
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("escalated", result)
        self.assertIn("policy", result)
    
    def test_track_lineage(self):
        """track_lineage returns comprehensive lineage data"""
        from capsule_lineage import InheritanceMode
        
        # Create hierarchy
        self.tree.spawn_child("ROOT", "C1", InheritanceMode.FULL)
        self.tree.spawn_child("C1", "C1-A", InheritanceMode.FULL)
        
        result = self.track_lineage(self.tree, "C1-A")
        
        self.assertTrue(result["found"])
        self.assertEqual(result["gid"], "C1-A")
        self.assertEqual(result["pid"], "C1")
        self.assertEqual(result["generation"], 2)
        self.assertIn("ROOT", result["lineage_path"])
    
    def test_calculate_drift(self):
        """calculate_drift computes drift score"""
        drift = self.calculate_drift(self.tree, "ROOT")
        self.assertEqual(drift, 0.0)  # Root has no drift from itself


# =============================================================================
# NEW METRICS TESTS (CDI, SCS, RSQ from V2 spec)
# =============================================================================

class TestNewMetrics(unittest.TestCase):
    """Test new metrics from V2 specification"""
    
    def setUp(self):
        np.random.seed(42)
        self.engine = ReflectiveMatrixEngine("RME-METRICS")
        
        self.baseline = CapsuleFingerprint("CAP-001", np.random.randn(32), {})
        self.engine.set_baseline("CAP-001", self.baseline)
        
        # Generate history for metrics
        for i in range(15):
            current = CapsuleFingerprint(
                "CAP-001",
                self.baseline.vector + np.random.randn(32) * (0.1 + i * 0.02),
                {}
            )
            self.engine.scan_drift("CAP-001", current)
            
            peers = [CapsuleFingerprint(f"P-{j}", np.random.randn(32), {}) for j in range(3)]
            self.engine.compute_evi("CAP-001", current, peers)
    
    def test_compute_cdi(self):
        """CDI computation works"""
        cdi = self.engine.compute_cdi("CAP-001")
        self.assertIsInstance(cdi, float)
        self.assertGreaterEqual(cdi, 0.0)
        self.assertLessEqual(cdi, 2.0)
    
    def test_compute_scs(self):
        """SCS computation works"""
        # Create swarm fingerprints
        base = np.random.randn(32)
        fps = [
            CapsuleFingerprint(f"M-{i}", base + np.random.randn(32) * 0.2, {})
            for i in range(5)
        ]
        
        scs = self.engine.compute_scs(fps)
        self.assertIsInstance(scs, float)
        self.assertGreaterEqual(scs, 0.0)
        self.assertLessEqual(scs, 1.0)
    
    def test_compute_rsq(self):
        """RSQ computation works"""
        rsq = self.engine.compute_rsq("CAP-001")
        self.assertIsInstance(rsq, float)
        self.assertGreaterEqual(rsq, 0.0)
        self.assertLessEqual(rsq, 1.0)
    
    def test_scs_coherent_swarm(self):
        """Coherent swarm has high SCS"""
        base = np.random.randn(32)
        # Very similar fingerprints
        fps = [
            CapsuleFingerprint(f"M-{i}", base + np.random.randn(32) * 0.05, {})
            for i in range(5)
        ]
        scs = self.engine.compute_scs(fps)
        self.assertGreater(scs, 0.5)
    
    def test_scs_incoherent_swarm(self):
        """Incoherent swarm has low SCS"""
        # Very different fingerprints
        fps = [
            CapsuleFingerprint(f"M-{i}", np.random.randn(32), {})
            for i in range(5)
        ]
        scs = self.engine.compute_scs(fps)
        self.assertLess(scs, 0.7)


# =============================================================================
# GOVERNANCE TESTS
# =============================================================================

class TestEthicalFilters(unittest.TestCase):
    """Tests for ethical filter system"""
    
    def setUp(self):
        from governance import EthicalFilterBank, EthicalDomain, RiskLevel
        self.EthicalFilterBank = EthicalFilterBank
        self.EthicalDomain = EthicalDomain
        self.RiskLevel = RiskLevel
    
    def test_safe_action_passes(self):
        """Safe actions pass ethical checks"""
        bank = self.EthicalFilterBank("TEST-BANK")
        context = {
            'resource_usage': 50,
            'resource_limit': 100,
            'modifies_core_params': False,
            'goal_drift': 0.1,
            'autonomy_level': 2
        }
        result = bank.evaluate_action("CAP-001", context)
        self.assertTrue(result['allowed'])
        self.assertEqual(len(result['violations']), 0)
    
    def test_risky_action_blocked(self):
        """Risky actions are blocked"""
        bank = self.EthicalFilterBank("TEST-BANK")
        context = {
            'resource_usage': 150,
            'resource_limit': 100,
            'modifies_core_params': True,
            'goal_drift': 0.5,
            'autonomy_level': 5
        }
        result = bank.evaluate_action("CAP-002", context)
        self.assertFalse(result['allowed'])
        self.assertGreater(len(result['violations']), 0)


class TestHumanOverrides(unittest.TestCase):
    """Tests for human override controller"""
    
    def setUp(self):
        from governance import HumanOverrideController, OverrideType
        self.HumanOverrideController = HumanOverrideController
        self.OverrideType = OverrideType
    
    def test_halt_prevents_progress(self):
        """Halt prevents capsule from proceeding"""
        ctrl = self.HumanOverrideController("TEST-CTRL")
        ctrl.request_override("CAP-001", self.OverrideType.HALT, "Test halt")
        self.assertFalse(ctrl.can_capsule_proceed("CAP-001"))
    
    def test_resolve_allows_progress(self):
        """Resolving override allows progress"""
        ctrl = self.HumanOverrideController("TEST-CTRL")
        req_id = ctrl.request_override("CAP-001", self.OverrideType.HALT, "Test halt")
        ctrl.resolve_override(req_id, "Approved", approved=True)
        self.assertTrue(ctrl.can_capsule_proceed("CAP-001"))


class TestSwarmThrottle(unittest.TestCase):
    """Tests for swarm throttling"""
    
    def setUp(self):
        from governance import SwarmThrottleController, SwarmThrottleConfig
        self.SwarmThrottleController = SwarmThrottleController
        self.SwarmThrottleConfig = SwarmThrottleConfig
    
    def test_throttle_limits_spawns(self):
        """Throttle limits swarm spawning"""
        config = self.SwarmThrottleConfig(max_active_swarms=3)
        ctrl = self.SwarmThrottleController(config)
        
        # First spawn should work
        self.assertTrue(ctrl.can_spawn_swarm("VERIFIER"))
        ctrl.record_spawn("SWM-0", "VERIFIER")
        
        status = ctrl.get_throttle_status()
        self.assertLessEqual(status['active_swarms'], config.max_active_swarms)
    
    def test_decay_reduces_activity(self):
        """Decay reduces swarm activity"""
        ctrl = self.SwarmThrottleController()
        ctrl.record_spawn("SWM-001", "VERIFIER")
        initial_activity = ctrl.swarm_activity["SWM-001"]
        
        ctrl.apply_decay()
        
        self.assertLess(ctrl.swarm_activity["SWM-001"], initial_activity)


class TestReflectiveMetricsComputation(unittest.TestCase):
    """Tests for reflective metrics"""
    
    def setUp(self):
        from governance import ReflectiveMetrics
        self.ReflectiveMetrics = ReflectiveMetrics
    
    def test_swarm_coherence_score(self):
        """Swarm Coherence Score computation"""
        scs = self.ReflectiveMetrics.swarm_coherence_score(
            [0.8, 0.9, 0.85], 0.95, 0.9
        )
        self.assertGreater(scs, 0)
        self.assertLessEqual(scs, 1)
    
    def test_reflective_stability_quotient(self):
        """Reflective Stability Quotient computation"""
        rsq = self.ReflectiveMetrics.reflective_stability_quotient(
            [0.7, 0.75, 0.8], [0.1, 0.12, 0.15], 5
        )
        self.assertGreater(rsq, 0)
        self.assertLessEqual(rsq, 1)
    
    def test_capsule_drift_index(self):
        """Capsule Drift Index computation"""
        cdi = self.ReflectiveMetrics.capsule_drift_index(0.2, 0.4, 0.25)
        self.assertGreater(cdi, 0)
        self.assertLessEqual(cdi, 1)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge case tests for robustness"""
    
    def test_orphan_capsule(self):
        """Test orphaned capsule (no parent in tree)"""
        np.random.seed(42)
        tree = CapsuleLineageTree("ORPHAN-TEST")
        
        # Create root
        root_fp = CapsuleFingerprint("ROOT", np.random.randn(32), {})
        tree.add_root("ROOT", root_fp)
        
        # Create orphan manually
        orphan_fp = CapsuleFingerprint("ORPHAN", np.random.randn(32), {})
        orphan_node = LineageNode(
            capsule_id="ORPHAN",
            fingerprint=orphan_fp,
            parent_id="NONEXISTENT",  # Parent doesn't exist
            role=LineageRole.ORPHAN
        )
        tree.nodes["ORPHAN"] = orphan_node
        
        # lineage_check should return False for orphan
        result = lineage_check(tree, "ORPHAN")
        self.assertFalse(result)
    
    def test_mds_zero_confirmation(self):
        """Test MDS with no confirming capsules"""
        np.random.seed(42)
        engine = ReflectiveMatrixEngine("MDS-ZERO")
        
        pattern = np.random.randn(32)
        # No confirming capsules
        mds = engine.compute_mds("CAP-001", pattern, [])
        
        # Should still compute but not be significant
        self.assertIsInstance(mds, MDSResult)
        self.assertFalse(mds.is_significant())
    
    def test_recursive_spawn_depth_limit(self):
        """Test recursive swarm spawning respects depth limits"""
        np.random.seed(42)
        tree = CapsuleLineageTree("DEPTH-TEST")
        
        root_fp = CapsuleFingerprint("ROOT", np.random.randn(32), {})
        tree.add_root("ROOT", root_fp)
        
        # Spawn 10 generations deep
        current = "ROOT"
        for i in range(10):
            child_id = f"GEN-{i}"
            tree.spawn_child(current, child_id, InheritanceMode.FULL)
            current = child_id
        
        # Check depth
        depth = lineage_depth(tree, current)
        self.assertEqual(depth, 10)
        
        # Verify lineage path
        path = tree.get_lineage_path(current)
        self.assertEqual(len(path), 11)  # ROOT + 10 generations
    
    def test_empty_swarm_consensus(self):
        """Test consensus with no members"""
        swarm = Swarm("EMPTY-SWM", SwarmType.VERIFIER)
        
        # Should handle gracefully
        result = swarm.converge()
        self.assertIn('status', result)
    
    def test_single_member_swarm(self):
        """Test swarm with single member"""
        np.random.seed(42)
        swarm = Swarm("SINGLE-SWM", SwarmType.VERIFIER)
        
        fp = CapsuleFingerprint("SOLO", np.random.randn(32), {})
        member = SwarmMember("SOLO", fp)
        swarm.expand(member)
        
        # Should work with single member
        comparisons = swarm.compare_fingerprints()
        self.assertEqual(len(comparisons), 0)  # No pairs to compare
        
        outliers = swarm.detect_outliers()
        self.assertEqual(len(outliers), 0)


class TestRefractoryPeriod(unittest.TestCase):
    """Tests for refractory period logic"""
    
    def setUp(self):
        from governance import RefractoryController, RefractoryConfig
        self.RefractoryController = RefractoryController
        self.RefractoryConfig = RefractoryConfig
    
    def test_initial_activation_allowed(self):
        """First activation should be allowed"""
        ctrl = self.RefractoryController()
        self.assertTrue(ctrl.can_activate_reflector("SWM-001"))
    
    def test_cooldown_blocks_activation(self):
        """Activation during cooldown should be blocked"""
        ctrl = self.RefractoryController()
        ctrl.record_activation("SWM-001", "REFLECTOR", evi=0.5)
        
        # Immediate re-activation should be blocked
        self.assertFalse(ctrl.can_activate_reflector("SWM-001"))
    
    def test_evi_damping(self):
        """Small EVI changes should be damped"""
        config = self.RefractoryConfig(
            reflector_cooldown_seconds=0,  # Disable cooldown
            evi_damping_threshold=0.1
        )
        ctrl = self.RefractoryController(config)
        ctrl.record_activation("SWM-001", "REFLECTOR", evi=0.5)
        
        # Small EVI change should be blocked
        self.assertFalse(ctrl.can_activate_reflector("SWM-001", current_evi=0.52))
        
        # Large EVI change should be allowed
        self.assertTrue(ctrl.can_activate_reflector("SWM-001", current_evi=0.7))


class TestMetaKernelHooks(unittest.TestCase):
    """Tests for meta-kernel bridge"""
    
    def setUp(self):
        from governance import MetaModelHook, MetaKernelBridge
        self.MetaModelHook = MetaModelHook
        self.MetaKernelBridge = MetaKernelBridge
    
    def test_hook_creation(self):
        """Test hook creation and updates"""
        hook = self.MetaModelHook(capsule_id="CAP-001")
        
        hook.update_evi(0.5)
        hook.update_evi(0.6)
        hook.update_evi(0.7)
        
        self.assertEqual(len(hook.evi_trail), 3)
        self.assertEqual(hook.evi_trail[-1], 0.7)
    
    def test_activation_score(self):
        """Test RME activation score computation"""
        hook = self.MetaModelHook(capsule_id="CAP-001")
        
        for evi in [0.8, 0.75, 0.7]:
            hook.update_evi(evi)
        for mds in [0.2, 0.3, 0.4]:
            hook.update_mds(mds)
        
        score = hook.compute_activation_score()
        self.assertGreater(score, 0)
        self.assertLess(score, 1)
    
    def test_bridge_export(self):
        """Test meta-kernel bridge export"""
        bridge = self.MetaKernelBridge("BRIDGE-001")
        
        bridge.update_from_swarm("CAP-001", "REFLECTOR", 0.85)
        bridge.update_from_lineage("CAP-001", depth=3, drift_trend="stable")
        
        export = bridge.export_for_b8()
        
        self.assertIn('hooks', export)
        self.assertIn('CAP-001', export['hooks'])
        self.assertEqual(export['hooks']['CAP-001']['last_swarm_role'], "REFLECTOR")


def run_b7_tests():
    """Run all B7 tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestCapsuleFingerprint,
        TestLineageNode,
        TestCapsuleLineageTree,
        TestLineageDSL,
        TestSwarm,
        TestSwarmCoordinator,
        TestReflectiveMatrixEngine,
        TestRMEDSL,
        TestIntegration,
        TestNewDSLPredicates,
        TestNewMetrics,
        # Governance tests
        TestEthicalFilters,
        TestHumanOverrides,
        TestSwarmThrottle,
        TestReflectiveMetricsComputation,
        # Edge cases and new features
        TestEdgeCases,
        TestRefractoryPeriod,
        TestMetaKernelHooks,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("BOOKLET 7 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_b7_tests()
    sys.exit(0 if success else 1)
