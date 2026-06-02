<!-- ooo:START -->
<!-- ooo:VERSION:0.39.1 -->
# Ouroboros — Specification-First AI Development

> Before telling AI what to build, define what should be built.
> As Socrates asked 2,500 years ago — "What do you truly know?"
> Ouroboros turns that question into an evolutionary AI workflow engine.

Most AI coding fails at the input, not the output. Ouroboros fixes this by
**exposing hidden assumptions before any code is written**.

1. **Socratic Clarity** — Question until ambiguity <= 0.2
2. **Ontological Precision** — Solve the root problem, not symptoms
3. **Evolutionary Loops** — Each evaluation cycle feeds back into better specs

```
Interview -> Seed -> Execute -> Evaluate
    ^                              |
    +------- Evolutionary Loop ----+
```

## ooo Commands

Each command loads its agent/MCP on-demand. Details in each skill file.

| Command | Loads |
|---------|-------|
| `ooo` | — |
| `ooo interview` | `ouroboros:socratic-interviewer` |
| `ooo seed` | `ouroboros:seed-architect` |
| `ooo run` | MCP required |
| `ooo evolve` | MCP: `evolve_step` |
| `ooo evaluate` | `ouroboros:evaluator` |
| `ooo unstuck` | `ouroboros:{persona}` |
| `ooo status` | MCP: `session_status` |
| `ooo setup` | — |
| `ooo help` | — |

## Agents

Loaded on-demand — not preloaded.

**Core**: socratic-interviewer, ontologist, seed-architect, evaluator,
wonder, reflect, advocate, contrarian, judge
**Support**: hacker, simplifier, researcher, architect
<!-- ooo:END -->

## Development Methodology

**TDD is required for all implementation work in this project.**

### Red-Green-Refactor cycle

1. **Red**: Write a failing test that describes the exact behavior needed.
   Confirm it fails for the correct reason (feature not implemented, not a compile error).
2. **Green**: Write the minimal implementation to make the test pass. Nothing more.
3. **Refactor**: Clean up without breaking the test.

### Rules

- No implementation code without a prior failing test.
- A test that fails due to a compile error does not count as Red — fix the build first.
- Each AC maps to at least one test. Write the test, confirm it fails, then implement.
- Metal GPU path: the test must assert GPU execution (not just correct output from CPU fallback).
- Checkpoint or note progress after each Red->Green transition.

### Metal-specific guidance

- For Metal Compute Shader work: write a test that verifies the Metal pipeline is
  actually dispatched (e.g., via a mock/spy on the command encoder, or by checking
  that a CPU-only path would produce wrong timing or wrong results under the test harness).
- CPU fallback must be explicitly disabled in tests that target the Metal path.
