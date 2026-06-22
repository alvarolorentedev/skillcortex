# SkillLatticeCoder Phase 1 Research Results

## Research phase status

- Phase 1: **complete**
- Phase 1.5: **complete**
- Phase 2.1: **complete**
- Phase 2.2 — Skill Registry / Immune Metadata / Capacity Budget: **complete**

## Executive conclusion

Fresh five-seed inference validates `protected_skill_router` as the new default:

- Protected router execution pass rate: **52.93%**
- Python generation: **54.0%**, equal to the frozen base
- Debugging: **46.4%**, equal to the previous routed lattice
- Test generation: **58.4%**, equal to the oracle lattice
- Improvement over the previous router: **+7.33 percentage points**

The concrete implementation remains available as
`python_only_for_test_generation`; the previous prompt-rule behavior remains
available explicitly as `legacy_rule_router`.

### Validated Mechanism 1: Protective Gating / Base Fallback

The system protects stable base-model capabilities by routing pure Python
generation to the frozen base instead of activating harmful broad Python
adapters.

The validation used fresh inference rather than counterfactual recombination:
`fresh_inference=true`, `counterfactual_recombination=false`,
`requires_training=false`, and `training_invoked=false`.

### Validated Mechanism 2: Failure-Born Skill Neurogenesis

The system can create a new quarantined skill from a repeated failure cluster,
validate it on independent holdout data, and recommend promotion only when it
improves the target without non-target regression.

- Protected router: **52.9%**
- SkillCortex Router V1: **53.6%**
- Fixed target: **0.0% → 50.0%**
- Holdout target: **74.0% → 99.3%**
- Non-target regressions: **0**

`alternating_skill` is the first promoted failure-born skill. Its historical
experiment remains quarantined and non-auto-promoting; production research
activation is limited to the explicit `skillcortex_router_v1` semantic gate.

### Validated Governance Layer 1: Bounded Skill Registry

A growing SkillCortex must track skill origin, status, activation scope,
promotion evidence, rollback metadata, and capacity impact before allowing
further neurogenesis.

## Original five-seed lattice result

The five-seed experiment provides statistically significant evidence that the
routed skill lattice outperforms the compute-matched generic LoRA baseline.

- Routed lattice execution pass rate: **45.60%**
- Generic LoRA execution pass rate: **40.67%**
- Paired absolute improvement: **+4.93 percentage points**
- Cluster-bootstrap 95% confidence interval: **+1.47 to +8.40 points**
- Routed lattice won against generic in **all five seeds**
- Routed lattice used **54.2% fewer active adapter parameters**

The result supports the statistical claim that the lattice beats the generic
adapter. It narrowly misses the preregistered practical-effect threshold of at
least five percentage points by 0.07 points. The generated summary therefore
labels the strict hypothesis `falsified`; this should not be interpreted as
evidence that the lattice has no benefit.

Oracle routing reached **47.60%**, an improvement of **+6.93 points** over
generic with a 95% confidence interval of **+3.60 to +10.27 points**. This
passes both the statistical and five-point practical thresholds. The two-point
gap between oracle and routed lattice identifies routing as the clearest next
optimization target.

## Experiment

### Model and adapters

- Frozen base: `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`
- Source model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Skill adapters: rank 8, 100 iterations each
- Generic adapter: rank 24, 300 iterations
- Seeds: `11`, `22`, `33`, `44`, `55`
- Skill adapter parameters: 311,296 each
- Generic adapter parameters: 933,888

The generic adapter received 300 optimizer iterations to match the combined
100 iterations used by each of the three skill adapters.

### Evaluation benchmark

The benchmark contains 150 execution-backed examples:

- 50 Python-generation tasks
- 50 debugging tasks
- 50 pytest-generation tasks

These are derived from 50 semantic families, each represented once in every
task category. Pytest-generation outputs must pass against the correct
implementation and fail against a paired mutant. Confidence intervals
bootstrap seed × semantic-family units, giving 250 paired observations rather
than treating the 750 task formulations as independent.

### Compared modes

1. Frozen base model
2. Compute-matched generic rank-24 LoRA
3. Single task-specific rank-8 skill
4. Rule-routed sparse skill lattice
5. Oracle lattice using benchmark skill labels

The primary metric is executable-test pass rate. Fuzzy similarity and exact
match are secondary diagnostics.

## Aggregate results

| Mode | Pass rate | Passed | Active parameters | Passes per million active parameters |
|---|---:|---:|---:|---:|
| Base | 33.33% | 250/750 | 0 | n/a |
| Generic LoRA | 40.67% | 305/750 | 933,888 | 0.435 |
| Single skill | 44.27% | 332/750 | 311,296 | 1.422 |
| Routed lattice | 45.60% | 342/750 | 427,513 average | 1.067 |
| Oracle lattice | 47.60% | 357/750 | 518,827 average | 0.917 |

The routed lattice achieves about **2.45×** the generic adapter's execution
performance per active parameter. The single-skill mode is the most
parameter-efficient at about **3.27×** generic, but its absolute pass rate is
1.33 points below the routed lattice.

## Results by task

| Mode | Python generation | Debugging | Test generation |
|---|---:|---:|---:|
| Base | 54.0% | 28.0% | 18.0% |
| Generic LoRA | 37.6% | 34.0% | 50.4% |
| Single skill | 38.0% | 42.0% | 52.8% |
| Routed lattice | 37.6% | 46.4% | 52.8% |
| Oracle lattice | 38.0% | 46.4% | 58.4% |

The aggregate lattice gain comes from debugging and test generation:

- Debugging composition adds 12.4 points over generic.
- Routed test generation adds 2.4 points over generic.
- Oracle test generation adds 8.0 points over generic.
- Fine-tuning regresses Python generation relative to the frozen base.

The Python-generation regression must be addressed before claiming that sparse
skills improve coding ability generally.

## Seed stability

| Seed | Generic | Single skill | Routed lattice | Oracle lattice | Routed delta | Oracle delta |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 42.67% | 48.00% | 48.00% | 48.67% | +5.33 | +6.00 |
| 22 | 39.33% | 46.00% | 47.33% | 48.67% | +8.00 | +9.33 |
| 33 | 42.00% | 42.67% | 44.00% | 46.00% | +2.00 | +4.00 |
| 44 | 40.67% | 43.33% | 44.67% | 44.67% | +4.00 | +4.00 |
| 55 | 38.67% | 41.33% | 44.00% | 50.00% | +5.33 | +11.33 |

The routed lattice beats generic in every seed, though the effect ranges from
two to eight points.

## Interpretation

### Supported

- Specialized adapters improve execution performance over a generic LoRA.
- Routed sparse composition provides a repeatable improvement over generic.
- The routed result is statistically significant and more parameter-efficient.
- Oracle composition exceeds the predefined five-point practical threshold.
- Routing quality materially affects the achievable lattice benefit.

### Not established

- The routed lattice did not clear the strict five-point practical threshold.
- Routed composition only modestly exceeds single-skill selection.
- Sparse skills do not improve every task category.
- This experiment does not establish generalization beyond the synthetic local
  Python benchmark or beyond the 1.5B Qwen model.

## Next experiments

1. Evaluate equal, strongest-skill, protected-pair, harmful-pair blocking, and
   deterministic weighted composition using the existing adapters.
2. If composition does not improve the protected router, move to failure-born
   skill creation.
3. Do not add or retrain skills before composition control is resolved.
4. Preserve the current benchmark and thresholds as a fixed regression suite.

## Reproduction and artifacts

Run the complete experiment:

```bash
python scripts/run_seeds.py \
  --seeds 11 22 33 44 55 \
  --output artifacts/experiments/five-seed
```

Key local artifacts:

- `artifacts/experiments/five-seed/summary.json`
- `artifacts/experiments/five-seed/seed-<seed>/results.jsonl`
- `artifacts/experiments/five-seed/seed-<seed>/summary.json`
- `artifacts/experiments/five-seed/seed-<seed>/report.md`
- `artifacts/experiments/five-seed/seed-<seed>/adapters/`

The artifacts directory is intentionally gitignored because it contains model
weights and large generated results. This document records the stable aggregate
results in version control.
