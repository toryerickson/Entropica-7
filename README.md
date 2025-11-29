# RSCS-Q Booklet 7: Reflective Swarms and Emergent Cognition

[![Tests](https://img.shields.io/badge/tests-63%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/F1--F8-100%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview

Booklet 7 introduces the **Reflective Swarm Layer** for RSCS-Q, bridging goal-directed autonomy (Booklet 6) with metacognitive self-modeling (Booklet 8). This layer enables capsule systems to observe, analyze, and discover patterns in their own behavior through genealogical tracking and swarm-based consensus.

**Key Capabilities:**
- 🧬 **Capsule Lineage System** — Genealogical tracking with GID/PID/CID identifiers
- 🐝 **Autonomous Swarms** — Five archetypes (VERIFIER, EXPLORER, REFLECTOR, ARCHIVIST, SYNTHESIZER)
- 📊 **Emergent Metrics** — EVI, MDS, CDI, SCS, RSQ for validity and discovery scoring
- 🔮 **Drift Forecasting** — Predictive analysis with breach detection
- 🛡️ **Governance Layer** — Ethical filters, human overrides, swarm throttling

## Installation

```bash
git clone https://github.com/your-org/rscs-q-booklet7.git
cd rscs-q-booklet7
pip install numpy
```

**Dependencies:** Python 3.8+, NumPy

## Quick Start

```python
from src import (
    CapsuleLineageTree, CapsuleFingerprint, InheritanceMode,
    Swarm, SwarmType, SwarmCoordinator,
    ReflectiveMatrixEngine,
    lineage_check, swarm_consensus_reached, evi_valid
)
import numpy as np

# Create lineage tree
tree = CapsuleLineageTree("DEMO")
root_fp = CapsuleFingerprint("ROOT", np.random.randn(32), {"stability": 0.9})
tree.add_root("ROOT", root_fp)

# Spawn children with inheritance
tree.spawn_child("ROOT", "CHILD-001", InheritanceMode.MUTATED, mutation_rate=0.1)

# Create reflector swarm
coordinator = SwarmCoordinator("COORD", tree)
swarm = coordinator.create_swarm(SwarmType.REFLECTOR)

# Compute emergent metrics
engine = ReflectiveMatrixEngine("ENGINE")
engine.connect_lineage_tree(tree)
# evi = engine.compute_evi("CHILD-001", child_fp, peer_fingerprints)
```

## Package Structure

```
booklet7-package/
├── src/
│   ├── __init__.py              # Package exports
│   ├── capsule_lineage.py       # Lineage tree and fingerprinting (780 LOC)
│   ├── swarm_reflector.py       # Swarm archetypes and coordination (789 LOC)
│   ├── reflective_matrix_engine.py  # EVI, MDS, drift forecasting (850 LOC)
│   ├── governance.py            # Ethical filters, overrides, throttling (750 LOC)
│   └── simulation_harness.py    # F1-F8 validation framework (500 LOC)
├── tests/
│   └── test_b7.py               # 63 unit tests
├── schemas/
│   ├── capsule_lineage_map.json # Lineage tree schema
│   └── swarm_event_log.json     # Swarm event schema
├── pdf/
│   └── booklet7.pdf             # 17-page documentation
├── tex/
│   └── booklet7.tex             # LaTeX source
└── README.md
```

## Core Concepts

### Capsule Lineage System

Tracks capsule genealogy with four inheritance modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| FULL | Complete trait copy | Exact replication |
| PARTIAL | 50% random inheritance | Variation exploration |
| MUTATED | Gaussian noise added | Controlled evolution |
| NONE | Fresh fingerprint | New lineage branch |

### Swarm Archetypes

| Type | Role | Spawn Trigger |
|------|------|---------------|
| VERIFIER | Validates outputs, builds consensus | CONSENSUS_FAILURE |
| EXPLORER | Discovers patterns, detects anomalies | ANOMALY_CLUSTER |
| REFLECTOR | Self-comparison, drift analysis | DRIFT_DETECTED |
| ARCHIVIST | Maintains history, logs discoveries | LINEAGE_BREAK |
| SYNTHESIZER | Combines insights across swarms | RUBRIC_DIVERGENCE |

### Metrics Reference

| Metric | Formula | Purpose |
|--------|---------|---------|
| **EVI** | ∛(coherence × stability × lineage_fidelity) | Internal validity assessment |
| **MDS** | novelty × significance × confirmation | Pattern discovery scoring |
| **CDI** | 0.4·d_curr + 0.3·d_max + 0.3·d_weighted | Composite drift index |
| **SCS** | ∛(sim × consensus × stability) | Swarm coherence |
| **RSQ** | (EVI × (1-drift)) / (1 + ln(1+depth)) | Reflective stability |

## Validation

### Run Tests

```bash
# Unit tests (63 tests)
python tests/test_b7.py
```

### Run Simulation

```bash
python src/simulation_harness.py
```

### Acceptance Criteria (F1-F8)

| ID | Metric | Target | Achieved |
|----|--------|--------|----------|
| F1 | Lineage Operations | 100% | ✅ 100.00% |
| F2 | Swarm Consensus | ≥60% | ✅ 98.40% |
| F3 | EVI Computation | ≥70% | ✅ 100.00% |
| F4 | MDS Detection | ≥20% | ✅ 100.00% |
| F5 | Drift Forecast | ≥70% | ✅ 100.00% |
| F6 | Coordination | ≥95% | ✅ 100.00% |
| F7 | Trigger System | 100% | ✅ 100.00% |
| F8 | DSL Coverage | ≥90% | ✅ 100.00% |

*Note: Results are simulation-only under controlled conditions.*

## DSL Predicates (21 total)

### Lineage (8 predicates)
```python
lineage_check(tree, capsule_id) -> bool
capsule_family_drift(tree, capsule_id, threshold=0.3) -> bool
check_lineage_trajectory(tree, capsule_id, max_drift=0.5) -> bool
lineage_depth(tree, capsule_id) -> int
inherit_capsule(tree, child_id, parent_id, mode, rate) -> bool
escalate_if_diverges(tree, capsule_id, threshold, policy) -> str
track_lineage(tree, capsule_id) -> dict
calculate_drift(tree, capsule_id) -> float
```

### Swarm (4 predicates)
```python
swarm_consensus_reached(swarm, threshold=0.67) -> bool
swarm_has_outliers(swarm, threshold=0.5) -> bool
trigger_reflective_response(coordinator, condition, context) -> bool
reflector_trigger(swarm, drift_threshold=0.3) -> bool
```

### Reflective Matrix Engine (4 predicates)
```python
evi_valid(engine, capsule_id, threshold=0.5) -> bool
drift_forecast_breach(engine, capsule_id, threshold=0.7) -> bool
pattern_discovered(engine, capsule_id, significance=0.3) -> bool
compare_fingerprint(engine, a, b, threshold=0.8) -> bool
```

### Governance (5 predicates)
```python
ethical_check(filter_bank, capsule_id, context) -> bool
requires_human_approval(filter_bank, capsule_id, context) -> bool
capsule_can_proceed(override_controller, capsule_id) -> bool
swarm_can_spawn(throttle_controller, swarm_type) -> bool
reflector_activation_allowed(refractory, swarm_id, evi, mds) -> bool
```

## Bridge to Booklet 8

B7 exports for meta-kernel integration:

```python
from src import MetaKernelBridge

bridge = MetaKernelBridge("BRIDGE-001")
bridge.update_from_evi(capsule_id, evi_result)
bridge.update_from_mds(capsule_id, mds_result)
bridge.update_from_swarm(capsule_id, "REFLECTOR", consensus=0.92)

# Export for B8
export = bridge.export_for_b8()
```

## Failure Modes and Responses

| Failure Mode | Frequency | Policy Response |
|--------------|-----------|-----------------|
| Consensus failure | ~8% | ESCALATE → spawn SYNTHESIZER |
| Forecast overshoot | ~12% | ADAPTIVE → widen bounds |
| Orphan lineage | ~3% | ARCHIVE → log + continue |
| Swarm flapping | Prevented | Refractory period blocks |
| EVI invalid | ~15% | STRICT → flag for review |

## Documentation

- [**booklet7.pdf**](pdf/booklet7.pdf) — 17-page technical documentation
- [**LaTeX source**](tex/booklet7.tex) — Full source with appendices A-H

## Related Booklets

| Booklet | Title | Relationship |
|---------|-------|--------------|
| B6 | Mission Kernel | Provides CapsuleMatrix, drift detection |
| **B7** | **Reflective Swarms** | **This booklet** |
| B8 | Meta-Kernel | Consumes EVI/MDS, self-modeling |

## License

MIT License

## Citation

```bibtex
@techreport{rscsq-b7-2025,
  title={RSCS-Q Booklet 7: Reflective Swarms and Emergent Cognition},
  author={Entropica Research Collective},
  year={2025},
  institution={RSCS-Q Project}
}
```

---

**Statistics:** 4,700 LOC | 63 tests | 21 DSL predicates | 17-page PDF | 8 appendices
