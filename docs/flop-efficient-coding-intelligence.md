# FLOP-Efficient Coding Intelligence

**Technical assessment, current through June 19, 2026**

## Executive conclusion

The core hypothesis is partly right and partly misframed.

Software engineering is extremely local in its **working set**: a task normally touches a small part of one repository, uses one toolchain, and can be checked by compilers, tests, linters, and runtime behavior. That makes coding unusually suitable for retrieval, external memory, structured tools, and verifier-driven search.

It does **not** follow that the useful knowledge is localized in a corresponding subset of a dense model's parameters. A Python packaging bug can require shell behavior, operating-system semantics, dependency resolution, historical API knowledge, and architectural judgment. Dense networks also store concepts superpositionally. No current evidence shows that a repository task can reliably identify and load arbitrary "knowledge regions" from a dense checkpoint.

The most promising startup architecture is therefore not a miniature model with a disk full of weight fragments. It is:

> **A small, resident coding policy operating over a compiled repository state, with adaptive verifier-driven search and learned escalation.**

For a first product, use an existing 7B–14B quantized model, not a new foundation model. Spend engineering effort on:

1. a persistent repository index derived from language tools;
2. high-recall task localization;
3. compact evidence construction;
4. bounded patch/compile/test loops;
5. a cheap learned judge for cases not fully covered by tests; and
6. escalation only when uncertainty or search failure justifies it.

This can plausibly deliver frontier-like behavior on a large fraction of routine repository work without frontier parameter counts. It will not match frontier models on the worst cases: ambiguous requirements, novel architecture, cross-domain debugging, weakly specified behavior, or problems requiring broad world knowledge.

The model-level research bet worth making after the system hypothesis is validated is a **recurrent/looped code model with explicit state tokens and adaptive depth**, trained on real software trajectories. This attacks the likely missing resource—iterative reasoning depth—without requiring proportionally more unique parameters.

## What already exists, and what is actually novel

Classification:

- **Established**: demonstrated repeatedly, with mature implementations or multiple independent results.
- **Emerging**: demonstrated, but evidence is recent, narrow, or operationally immature.
- **Novel combination**: components exist, but their proposed integration and optimization target are not standard.
- **Unsupported**: attractive hypothesis without evidence strong enough to build a company around.

| Idea | Status in 2026 | Evidence | Assessment |
|---|---|---|---|
| **1. Hot-zone inference** | **Unsupported as stated** | Sparse MoE routing, activation sparsity, adaptive depth, and expert caching exist. Systems such as [MoE-Infinity](https://arxiv.org/abs/2401.14361), [HOBBIT](https://arxiv.org/abs/2411.01433), and [Fate](https://arxiv.org/abs/2502.12224) exploit predictable expert use. | There is no demonstrated mapping from a repository/task description to arbitrary useful regions of a dense model. Replace "parameter hot zones" with measurable task-level routing among explicitly trained modules. |
| **2. Dynamic expert loading** | **Established systems problem; emerging product fit** | Expert offloading and prefetching are active fields. [FloE](https://arxiv.org/abs/2505.05950) identifies PCIe bandwidth as the limiting factor; [local routing consistency](https://arxiv.org/abs/2505.16056) varies substantially by model. | Not novel. On a 24GB discrete GPU, loading experts per token is usually the wrong granularity. Task- or phase-level expert residency may work if routing remains stable for seconds, not milliseconds. |
| **3. Adapter composition** | **Established; repository generation is emerging** | [LoRA](https://arxiv.org/abs/2106.09685), [AdapterFusion](https://arxiv.org/abs/2005.00247), and [Mixture of LoRA Experts](https://arxiv.org/abs/2404.13628) cover adaptation and composition. [Mix-of-Language-Experts](https://arxiv.org/abs/2506.18923) routes language LoRAs. June 2026's [Code2LoRA](https://arxiv.org/abs/2606.06492) generates repository-specific adapters with a hypernetwork. | The broad idea is not novel. The useful open question is whether repository adapters improve issue resolution enough to repay training, invalidation, and composition complexity. Retrieval should be the default until that is proven. |
| **4. Repository-native cognition** | **Established direction; still underexploited** | [RepoGraph](https://arxiv.org/abs/2410.14684), [CodexGraph](https://arxiv.org/abs/2408.03910), [ContextBench](https://arxiv.org/abs/2602.05892), and [Codebase-Memory](https://arxiv.org/abs/2603.27277) represent repositories structurally. Codebase-Memory reports tenfold lower token use than file exploration at some quality cost. | This is the strongest part of the hypothesis, but "knowledge graph" is too vague. The winning representation is likely a typed, incrementally maintained program index plus evidence cache, not a general graph database. |
| **5. Compiler-assisted reasoning** | **Established** | [StepCoder](https://arxiv.org/abs/2402.01391) trains from compiler feedback. [CatCoder](https://arxiv.org/abs/2406.03283) injects static type context. [iSWE](https://arxiv.org/abs/2603.11356) uses language-specific analysis and transformation tools. | Compiler feedback is high-value because it is cheap, structured, and local. It narrows search; it does not establish behavioral correctness. |
| **6. Test-assisted reasoning** | **Established and essential** | [SWE-agent](https://arxiv.org/abs/2405.15793), [OpenHands](https://arxiv.org/abs/2407.16741), [SWE-Gym](https://arxiv.org/abs/2412.21139), and [SWE-Master](https://arxiv.org/abs/2602.03411) all use executable environments. [SWE-Replay](https://arxiv.org/abs/2601.22129) reduces repeated test-time search cost. | Tests are the best available verifier, but they under-specify behavior, can be flaky, and can reward overfitting. The agent must distinguish test evidence from proof. |
| **7. Hierarchical coding agents** | **Established; often overbuilt** | Role-based and multi-agent systems are common. [iSWE](https://arxiv.org/abs/2603.11356) separates localization and editing. However, a 2026 analysis of 9,374 trajectories found that the underlying model drove outcomes more than framework choice and that framework gaps shrank as models improved ([Mehtiyev and Assunção](https://arxiv.org/abs/2604.02547)). | Use separate policies only where they have different inputs, actions, or cost profiles. "Architect/planner/implementer/reviewer" personas sharing one model and context often multiply tokens rather than intelligence. |
| **8. Self-distillation** | **Established; target choice remains open** | [SWE-smith](https://arxiv.org/abs/2504.21798), [SWE-Gym](https://arxiv.org/abs/2412.21139), [SWE-Master](https://arxiv.org/abs/2602.03411), and [Open-SWE-Traces](https://arxiv.org/abs/2606.16038) train from software trajectories and execution feedback. | Distilling generic prose reasoning is not the opportunity. Distill control policies: where to look, what evidence is missing, which test to run, when to branch, and when to stop or escalate. |

### The genuinely new opportunity

No individual component is novel. The defensible architecture-level opportunity is the joint optimization of:

- an external, typed repository state;
- a small resident policy;
- adaptive recurrent or test-time compute;
- verifiers with known coverage;
- phase-level specialization; and
- **resolved tasks per joule/GPU-hour** as the training and product objective.

Current systems usually optimize model loss, benchmark pass rate, token count, or serving throughput separately. Coding permits a stronger objective because the environment exposes causal feedback. A system can learn which expensive reasoning actions actually change the probability of a valid patch.

## Assumptions that do not survive scrutiny

### Repository locality does not imply parameter locality

Task locality is observable in files, symbols, dependencies, and tests. Parameter locality is a property of a trained network and its routing. Dense transformers do not provide stable, semantically named regions such as "Django migrations" or "this repository's authorization architecture."

Sparse MoE models create explicit parameter partitions, but token-level routers are trained to reduce language-model loss, not to maintain a repository capability map. Qwen's [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) demonstrates the attraction of this direction—80B total parameters with 3B activated—but all 80B parameters still need accessible storage, and unpredictable routes complicate local deployment.

The practical replacement is:

- keep **repository facts** outside weights;
- keep **procedural skill** in the model;
- train explicit experts only for stable, measurable partitions such as localization, editing, test selection, or a language family.

### Parameters, FLOPs, memory bandwidth, and reasoning depth are different budgets

A model can use few active parameters and still be impossible to run well locally because inactive parameters must remain resident or be fetched. Conversely, a recurrent model can have few parameters but use substantial FLOPs by repeatedly applying them.

This distinction matters:

- **Unique parameters** primarily determine weight memory and memorized capacity.
- **Activated parameters** influence arithmetic per token.
- **KV cache and context length** determine a growing part of runtime memory.
- **Memory bandwidth** often limits autoregressive decoding.
- **Effective depth/test-time search** determines how much sequential reasoning is attempted.

The startup should optimize the whole task, not advertise one favorable number.

### Cold expert loading is usually a bandwidth problem disguised as an AI architecture

An RTX 4090 has 24GB of VRAM ([NVIDIA specification](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)). A typical PCIe 4.0 x16 host link has roughly 32GB/s theoretical bandwidth; HOBBIT uses this exact class of setup ([paper](https://arxiv.org/abs/2411.01433)). The GPU's internal memory bandwidth is roughly a terabyte per second. Even before software overhead, host transfers are tens of times slower than reads from VRAM.

Research repeatedly identifies expert-loading latency as the obstacle: [FloE](https://arxiv.org/abs/2505.05950), [Fate](https://arxiv.org/abs/2502.12224), and [importance-driven scheduling](https://arxiv.org/abs/2508.18983) all add prediction, caching, quantization, or CPU execution to hide it.

Therefore:

- do not swap experts token by token on a gaming PC;
- if experts are used, select a small set for an entire task phase and keep them resident;
- prefer a fully resident 14B dense model or approximately 30B sparse model over a larger model that continuously crosses PCIe;
- treat Apple unified memory differently: current Mac Studio configurations provide 546GB/s with M4 Max and 819GB/s with M3 Ultra ([Apple specifications](https://www.apple.com/mac-studio/specs/)), making large resident quantized models feasible, though generally with lower raw compute than a top discrete GPU.

### Compilers and tests are verifiers, not substitutes for judgment

Compiler errors provide dense local information about syntax, types, linkage, and contracts. Tests provide behavioral evidence. Neither tells the system whether:

- the requirement was interpreted correctly;
- an untested security or compatibility property was broken;
- the patch fits the architecture;
- the implementation is maintainable; or
- the tests themselves are wrong.

OpenAI stopped recommending SWE-bench Verified in 2026 after finding contamination and flawed tests in an audited subset ([analysis](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)). A verifier-aware system must track what was actually checked, not collapse all green signals into "correct."

### More agents do not create more capability

Multiple agents help when work can be partitioned, explored concurrently, or evaluated independently. They hurt when identical models repeatedly summarize the same context. Multi-agent role labels alone provide no new information, tools, or inductive bias.

The 2026 behavioral study found that successful agents gather context before editing and invest in validation, while architectural reasoning and domain knowledge remain failure sources ([paper](https://arxiv.org/abs/2604.02547)). This argues for better policies and state, not a permanent committee of personas.

## Important approaches missing from the initial list

### 1. Recurrent depth rather than parameter scale

[Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416) argues that many reasoning tasks require depth but not proportionally many unique parameters. [LoopLM](https://arxiv.org/abs/2510.25741) learns adaptive recurrent steps. [MELT](https://arxiv.org/abs/2605.07721) shares KV state across loops to prevent memory growth. [Attractor Models](https://arxiv.org/abs/2605.12466) choose iterations by convergence and report improvements over parameter-matched transformers.

This is more aligned with software than arbitrary expert paging. Debugging is naturally iterative: update a hypothesis after each observation. The limitation is evidence: most recurrent-model results are on synthetic reasoning or general language tasks, not repository engineering. This is a second-stage research program, not the first PoC.

### 2. Learned repository exploration as its own cheap policy

Context retrieval is not a preprocessing detail. It is a sequential decision problem. [ContextBench](https://arxiv.org/abs/2602.05892) measures file-, block-, and line-level evidence retrieval. [FastContext](https://arxiv.org/abs/2606.14066) trains 4B–30B exploration models that return compact file-line evidence.

A 1B–4B explorer can be cheaper than asking the main model to repeatedly grep, read, summarize, forget, and reread. This is a justified specialist because it has a distinct action space and verifiable target: evidence recall at bounded token cost.

### 3. Structured action spaces

Raw shell and text patching are universal but expensive and error-prone. Language-server and compiler APIs can expose:

- symbol definition/reference;
- call hierarchy;
- type hierarchy;
- diagnostics;
- safe rename and refactoring;
- impacted tests;
- build targets; and
- exact source spans.

[iSWE](https://arxiv.org/abs/2603.11356) supports the broader principle: stronger language-aware tools can reduce LLM turns and side effects. The system should present high-level operations, retaining shell access as an escape hatch.

### 4. Coverage-aware verification

The agent needs a verification ledger:

- claim or changed behavior;
- evidence source;
- tests or static checks covering it;
- observed result;
- residual uncertainty.

This is more useful than an undifferentiated critic. It lets a cheap controller spend compute only where coverage is weak and prevents test passing from becoming false certainty.

### 5. Search-state reuse

Naive best-of-N repeats repository discovery N times. [SWE-Replay](https://arxiv.org/abs/2601.22129) branches from useful intermediate states and reports lower cost with maintained or improved performance.

Repository indexing, build setup, localization evidence, and failed hypotheses should be shared across candidates. Candidate diversity should begin at uncertain decisions, not at the initial prompt.

### 6. Small judges and risk-triggered escalation

A specialized small judge can rank generated code competitively with models 5–25 times larger in studied settings ([Crupi et al., 2026](https://arxiv.org/abs/2602.11911)). This does not make model judging trustworthy by itself; judge bias remains documented ([Bias in the Loop](https://arxiv.org/abs/2604.16790)).

Use a judge only after deterministic checks and calibrate it on:

- patch risk;
- likely test overfitting;
- requirement coverage;
- architectural inconsistency; and
- whether escalation is worth its cost.

### 7. Temporal repository state

A static graph omits the most informative repository data:

- recent commits in the subsystem;
- ownership and review history;
- prior failed fixes;
- recurring test failures;
- API migrations;
- issue-to-code links; and
- decisions recorded in documentation.

This state should remain source-linked and invalidatable. Generated summaries without provenance will become stale and confidently wrong.

## Recommended startup architecture

### Architecture: Repository-Native Adaptive Compute

```text
User task
   |
   v
Task classifier + risk estimator
   |
   v
Repository state service ----> symbol/type/call/import/test/change indexes
   |                                      |
   v                                      v
Cheap exploration policy ----------> compact evidence bundle
   |
   v
Resident 7B–14B coding core
   |
   +--> hypothesis
   +--> structured edit
   +--> targeted diagnostic/test
   +--> update task state
   |
   v
Deterministic verifier --> small learned judge --> accept / branch / escalate
```

### 1. Resident reasoning core

Use one strong open 7B–14B code/instruction model in 4–5 bit quantization, permanently resident on the GPU.

Why this size:

- 14B Q4 weights are roughly 7–10GB after quantization overhead, leaving material room in 24GB for KV cache, runtime buffers, and a small auxiliary model.
- The full working set stays on fast memory.
- Multiple short candidates become affordable.
- Fine-tuning and LoRA experiments remain possible on accessible hardware.

Do not require a recurrent architecture for v1 because no repository-proven recurrent checkpoint is currently the safe base. The first custom-model experiment should convert or train a small code model with:

- shared middle blocks;
- adaptive loop count;
- persistent state/register tokens;
- supervision on hypothesis updates and tool feedback; and
- an exit head trained against expected value of another reasoning step.

### 2. Repository state service

Build the smallest useful index:

- definitions and references;
- imports and package/module dependencies;
- calls where statically recoverable;
- type and inheritance relations;
- tests mapped to production symbols using static references and coverage when available;
- build targets and commands;
- recent change history and ownership;
- diagnostics and last observed verification results.

Use existing language servers, compiler metadata, Tree-sitter, test collectors, coverage tools, and Git. Do not invent a universal semantic graph schema before the PoC. Store typed records in SQLite and source files; add a graph database only if measured queries require it.

Every derived fact must include source location, tool, repository revision, and invalidation key.

### 3. Retrieval and context construction

Retrieval should combine:

1. lexical/semantic matches from the issue;
2. symbol and type relationships;
3. import/call distance;
4. changed-file and ownership priors;
5. failing-test traces and diagnostics;
6. examples of the same API used elsewhere; and
7. exploration history from the current attempt.

Return evidence blocks, not whole files:

- exact source span;
- signature and containing type/module;
- one-hop dependencies;
- why the evidence was selected;
- confidence and provenance.

Keep raw code available on demand. Summaries are indexes, not authorities.

### 4. Adaptive engineering loop

The main controller maintains explicit state:

- current hypotheses;
- supporting and contradicting evidence;
- intended edits;
- verification ledger;
- failed attempts;
- unresolved risks; and
- remaining compute budget.

The loop is:

1. localize;
2. state a falsifiable hypothesis;
3. request missing evidence;
4. make the smallest candidate edit;
5. run the cheapest discriminating check;
6. update state;
7. accept, branch from the last useful state, or escalate.

Run checks in increasing cost order:

1. parse/format;
2. local type or compiler diagnostic;
3. focused unit test;
4. impacted test set;
5. broader suite;
6. learned review;
7. expensive model escalation.

### 5. Specialists worth having

Use specialists only where the interface and target differ:

- **Explorer (1B–4B or rules plus embeddings):** maximize relevant evidence recall per token.
- **Main coder (7B–14B):** reason and edit.
- **Judge (1B–4B):** estimate residual failure risk after deterministic verification.

Do not create permanent architect, planner, implementer, and reviewer agents using the same checkpoint. Planning and review can be modes of the main core with isolated context views.

### 6. Adapters

Start with no runtime adapter composition.

Add a language/toolchain LoRA only when an A/B test shows durable gains across held-out repositories. Add a repository adapter only if it beats refreshed retrieval after accounting for:

- training GPU-hours;
- repository churn;
- invalidation;
- adapter storage;
- composition conflicts; and
- cold-start delay.

[Code2LoRA](https://arxiv.org/abs/2606.06492) makes generated repository adapters a credible later experiment, but it is too new to replace retrieval as the default architecture.

### 7. Escalation

Escalate to a larger local or remote model when one of these occurs:

- evidence retrieval remains low-confidence after a fixed budget;
- requirements are ambiguous or contradictory;
- the patch spans architectural boundaries;
- deterministic checks disagree;
- the learned judge is uncertain;
- repeated candidate branches converge on the same failure; or
- the task is high-risk: security, data migration, concurrency, money, or public API compatibility.

Train the escalation policy on expected gain per dollar/second, not model confidence alone.

## Can this reach frontier behavior without frontier parameter counts?

### Yes, on a workload distribution

A smaller system can match or beat a frontier model on many routine tasks when:

- the relevant repository state is retrieved accurately;
- correctness has executable signals;
- the task is narrow enough for bounded search;
- tools expose semantics directly;
- multiple candidates are cheap; and
- the larger model's broad knowledge is mostly unused.

Evidence already supports the components:

- SWE-agent showed that interface design materially changes performance ([paper](https://arxiv.org/abs/2405.15793)).
- SWE-Gym showed large gains from only hundreds of environment trajectories and further gains from verifier-based inference scaling ([paper](https://arxiv.org/abs/2412.21139)).
- Small code judges can improve candidate selection at a fraction of generator cost ([paper](https://arxiv.org/abs/2602.11911)).
- Structured repository memory can cut token and tool use substantially ([Codebase-Memory](https://arxiv.org/abs/2603.27277)).
- Devstral demonstrated that a dense 24B model specialized for agent trajectories can compete with much larger models ([technical report](https://arxiv.org/abs/2509.25193)).

### No, as a universal replacement

Parameter count still buys:

- broad language and library coverage;
- rare API and historical knowledge;
- architectural priors;
- robust interpretation of ambiguous requests;
- transfer to unfamiliar domains;
- stronger judgment when tests are absent; and
- recovery from surprising tool output.

[ProgramBench](https://arxiv.org/abs/2605.03546) is a useful warning: in its full-program reconstruction setting, no evaluated task was fully resolved, including by frontier systems. Locality reduces irrelevant context; it does not remove hard reasoning or missing knowledge.

The correct product claim is:

> Match frontier outcomes on the common, verifiable portion of a customer's repository workload at far lower cost, and escalate the tail.

## Local hardware recommendation

### Single 24GB NVIDIA GPU

Best fit:

- resident 7B–14B Q4/Q5 core;
- 16K–32K practical context, with retrieval preventing routine use of the maximum window;
- optional 1B–4B explorer/judge loaded concurrently or sequentially;
- repository index in CPU RAM/SSD;
- no token-level weight swapping.

A 24B dense model such as Devstral Small can be a strong comparison baseline and may fit quantized, but leaves less runtime headroom. An 80B-total sparse model such as Qwen3-Coder-Next does not fit fully in 24GB at ordinary 4-bit weight sizes, despite activating only 3B parameters. Offloading makes it a systems experiment, not the baseline.

### Mac Studio

Best fit:

- larger fully resident quantized sparse models;
- long-running private repository agents;
- larger context and concurrent auxiliaries;
- experimentation with phase-level expert residency.

Unified memory avoids PCIe copies between CPU and GPU memory, and high-memory configurations can hold far larger checkpoints. The tradeoff is platform-specific kernel maturity and lower peak throughput per euro than top NVIDIA hardware for some inference stacks.

### High-end laptop

Best fit:

- 7B Q4 model;
- aggressively bounded context;
- rule/LSP-first retrieval;
- one candidate at a time;
- remote escalation.

The laptop architecture should be identical at the service boundary. Only budgets change.

## First proof of concept

### Hypothesis

> Externalized repository state plus execution-guided adaptive search lets a 7B–14B model resolve materially more fresh repository tasks per GPU-hour than both a plain small-model agent and a larger quantized model.

This tests the central system claim without spending months training a new model.

### Minimal implementation

Use one language first. Choose Python for infrastructure speed or Java/Rust for stronger compiler signals. Python is cheaper; Java or Rust is a stronger test of compiler-assisted reasoning. Default: **Python for the first six-week experiment, then replicate on Java or Rust before making an architecture claim.**

Build:

- SQLite repository index populated by Tree-sitter, language server/static tools, test collection, Git, and optional coverage;
- a compact command-line agent with structured read/edit/test actions;
- explicit task-state JSON persisted between turns;
- deterministic test orchestration;
- a simple candidate archive that branches after localization rather than restarting;
- metrics emitted for every model and tool action.

Do not build:

- a custom foundation model;
- a graph database;
- arbitrary neural weight paging;
- a general multi-agent platform;
- repository LoRA training;
- an IDE;
- a distributed serving layer.

### Experimental arms

Use the same prompt budget, quantization class, hardware, and task timeout where applicable.

| Arm | System |
|---|---|
| **A — Plain small agent** | 7B–14B model, shell/file tools, ordinary rolling context. |
| **B — Structured retrieval** | A plus repository index and evidence bundles. |
| **C — Execution loop** | B plus explicit hypotheses, targeted compile/test selection, verification ledger, and state reuse. |
| **D — Full efficient system** | C plus a trained explorer or routing policy, cheap judge, bounded branching, and learned escalation. |
| **E — Larger baseline** | Strong 24B-class quantized coding model using the same basic harness as A. |

Arm D is added only after B and C show gains. This prevents a learned router from hiding a weak index or controller.

### Dataset

Use two stages:

1. **Development:** 100 executable tasks generated or collected after the base models' training cutoff, across at least 20 repositories.
2. **Confirmation:** at least 300 untouched tasks from different repositories and a later time window.

Prefer continuously refreshed tasks from [SWE-rebench](https://arxiv.org/abs/2505.20411) or a private equivalent. Include a smaller public comparison on [SWE-Bench Pro](https://arxiv.org/abs/2509.16941) and a multilingual follow-up using [Multi-SWE-bench](https://arxiv.org/abs/2504.02605).

Do not use SWE-bench Verified as the primary decision metric. It is useful only for compatibility with prior work.

Stratify by:

- bug fix versus feature;
- patch size and file count;
- repository size;
- test quality;
- required external knowledge;
- language/toolchain;
- architectural span; and
- whether the issue identifies likely files.

### Metrics

Primary:

- resolved tasks per GPU-hour;
- joules per resolved task;
- wall-clock time per resolved task;
- total generated and prefetched tokens;
- peak VRAM.

Capability:

- task resolve rate;
- file/block/line localization recall and precision;
- first-valid-patch rate;
- regression rate on broader tests;
- success conditional on verifier coverage;
- escalation rate and escalation gain.

Process:

- tool calls;
- files and lines read;
- repeated reads;
- number of candidate branches;
- compile/test executions and duration;
- fraction of context retrieved but unused;
- stale-index or invalidation failures.

### Acceptance thresholds

Proceed to custom-model research only if the confirmation set shows all of:

1. **At least 2× resolved tasks per GPU-hour** for D versus A.
2. **At least 1.25× resolved tasks per GPU-hour** for D versus E.
3. D reaches **at least 90% of E's absolute resolve rate**, or exceeds it.
4. B improves localization F1 and lowers tokens without reducing resolve rate.
5. C provides a statistically credible resolve-rate gain over B, proving that feedback—not retrieval alone—matters.
6. Gains persist on repositories unseen during router/judge training.
7. No more than 20% of D's wins depend on escalation; otherwise the local system is only a front end for the larger model.

These are startup thresholds, not paper-friendly thresholds. Smaller gains do not justify owning a new model architecture.

### Ablations that determine the next architecture

- lexical retrieval versus structural retrieval;
- full files versus evidence spans;
- static dependency edges versus dynamic test/trace edges;
- rolling prose summary versus explicit task state;
- restart candidates versus branch from shared exploration;
- deterministic verification only versus added small judge;
- fixed compute budget versus learned stopping/escalation;
- generic model versus language LoRA;
- dense resident model versus fully resident sparse model;
- later: feed-forward core versus recurrent shared-depth core.

### Falsification criteria

The locality hypothesis is weakened if:

- a larger model with plain tools dominates after cost normalization;
- structural retrieval does not improve evidence recall;
- compiler/test loops mainly consume time without changing outcomes;
- unseen repositories erase the gains;
- most successful tasks require escalation; or
- repository adapters beat refreshed external state by a large margin.

Any of these results is valuable. They prevent an expensive custom-model program built on the wrong bottleneck.

## Ranked technical risks

1. **Localization is not the main bottleneck.** Some failures come from architectural reasoning or domain knowledge even when the right files are known.
2. **Verifier gaming and weak tests.** The system may optimize visible tests while violating intent.
3. **State corruption.** Generated architectural summaries can become stale and poison later reasoning.
4. **Small-model control failure.** The core may have enough coding knowledge but insufficient long-horizon policy stability.
5. **Benchmark contamination.** Public benchmark gains may measure memory rather than engineering.
6. **Sparse-model residency.** Low active FLOPs may not produce low latency on 24GB hardware.
7. **Adapter interference and churn.** Repository adapters may expire faster than they repay training.
8. **Tool heterogeneity.** Language servers and build systems expose inconsistent semantics.
9. **Tail dependence on frontier models.** Escalation may capture most of the value, undermining local economics.
10. **Recurrent training instability.** Looped models are promising but not yet proven for repository-scale agent trajectories.

## Product and research sequence

### Phase 1: prove externalized cognition

Build arms A–C. Establish whether structured repository state and targeted feedback improve tasks/GPU-hour.

### Phase 2: learn control, not knowledge

Train the explorer, test selector, stop/branch policy, judge, and escalation policy from collected trajectories. Keep the generator frozen initially.

### Phase 3: test stable specialization

Measure language/toolchain LoRAs and phase-level experts. Reject any module that does not improve held-out task economics.

### Phase 4: recurrent core

Only after the control/data pipeline works, train a small adaptive-depth code model on:

- hypothesis/evidence updates;
- compiler and test feedback;
- counterfactual failed branches;
- stopping decisions;
- repository exploration traces; and
- final validated patches.

Compare at equal active FLOPs and equal wall-clock budget, not only equal parameters.

### Phase 5: hardware-aware sparse capacity

If expert activation is stable at task or phase granularity, add resident expert sets. Optimize around real memory hierarchy:

- 24GB discrete GPU: compact hot set, no frequent PCIe paging;
- Mac unified memory: larger resident cold capacity;
- cloud: conventional MoE serving and batching.

## Final thesis

The strongest path is not to discover which pieces of a dense frontier model can be switched off. That framing assumes semantic modularity that current models do not expose.

The practical breakthrough is to make the **software environment** carry most of the session-specific intelligence:

- the repository stores facts;
- program analysis exposes structure;
- tests and compilers produce causal feedback;
- a small model supplies procedural judgment;
- adaptive search supplies extra depth only when evidence warrants it;
- a larger model handles the irreducible tail.

This architecture can be dramatically more efficient because it does not ask parameters to repeatedly reconstruct information that the repository and toolchain already know. If that system closes most of the quality gap, recurrent shared-depth models are the next model-level bet. If it does not, dynamic weight loading will not rescue the hypothesis—the missing capability is broader reasoning and knowledge, not merely inefficient activation.

## Primary references

- Jimenez et al., [SWE-bench](https://arxiv.org/abs/2310.06770), 2023/2024.
- Yang et al., [SWE-agent](https://arxiv.org/abs/2405.15793), 2024.
- Wang et al., [OpenHands](https://arxiv.org/abs/2407.16741), 2024/2025.
- Pan et al., [SWE-Gym](https://arxiv.org/abs/2412.21139), 2024/2025.
- Yang et al., [SWE-smith](https://arxiv.org/abs/2504.21798), 2025.
- Badertdinov et al., [SWE-rebench](https://arxiv.org/abs/2505.20411), 2025.
- Deng et al., [SWE-Bench Pro](https://arxiv.org/abs/2509.16941), 2025.
- Zan et al., [Multi-SWE-bench](https://arxiv.org/abs/2504.02605), 2025.
- Saunshi et al., [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416), 2025.
- [LoopLM](https://arxiv.org/abs/2510.25741), 2025.
- [MELT](https://arxiv.org/abs/2605.07721), 2026.
- Fein-Ashley and Rashidinejad, [Attractor Models](https://arxiv.org/abs/2605.12466), 2026.
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021.
- Pfeiffer et al., [AdapterFusion](https://arxiv.org/abs/2005.00247), 2020/2021.
- Zong et al., [Mix-of-Language-Experts](https://arxiv.org/abs/2506.18923), 2025.
- [Code2LoRA](https://arxiv.org/abs/2606.06492), 2026.
- Zhang et al., [RepoCoder](https://arxiv.org/abs/2303.12570), 2023.
- [RepoGraph](https://arxiv.org/abs/2410.14684), 2024/2025.
- [ContextBench](https://arxiv.org/abs/2602.05892), 2026.
- [Codebase-Memory](https://arxiv.org/abs/2603.27277), 2026.
- Ding et al., [SWE-Replay](https://arxiv.org/abs/2601.22129), 2026.
- Crupi et al., [Small Language Model as a Judge](https://arxiv.org/abs/2602.11911), 2026.
- [Devstral technical report](https://arxiv.org/abs/2509.25193), 2025.
- [Qwen3-Coder-Next model card](https://huggingface.co/Qwen/Qwen3-Coder-Next), 2026.
- Shazeer et al., [Sparsely-Gated Mixture of Experts](https://arxiv.org/abs/1701.06538), 2017.
- Xue et al., [MoE-Infinity](https://arxiv.org/abs/2401.14361), 2024.
- [HOBBIT](https://arxiv.org/abs/2411.01433), 2024/2025.
- [FloE](https://arxiv.org/abs/2505.05950), 2025.
- Tang et al., [Programming by Chat](https://arxiv.org/abs/2604.00436), 2026.
- Mehtiyev and Assunção, [Behavioral Drivers of Coding Agent Success and Failure](https://arxiv.org/abs/2604.02547), 2026.
- [ProgramBench](https://arxiv.org/abs/2605.03546), 2026.
