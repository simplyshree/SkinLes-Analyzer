# Development Guide for ChatGPT and Codex

## Repository rules

- Keep the project runnable after every change.
- Prefer small, focused changes over broad rewrites.
- Write or update tests for non-trivial processing, data, or UI-support logic.
- Do not commit datasets, archives, checkpoints, credentials, or generated local caches.
- Do not insert fabricated benchmark values.
- Keep the Streamlit app inference-only; training belongs in scripts or notebooks.
- Preserve the non-diagnostic educational scope.
- Update `README.md`, `ARCHITECTURE.md`, `PRODUCT.md`, or an ADR when behaviour or design changes.
- Run tests and compile checks before handing off.

## Division of work

ChatGPT should own ambiguity reduction, tradeoff decisions, sequencing, and review of whether implementation matches the intended project. Codex should own file edits, tests, wiring, and concrete implementation.

When a request changes the product boundary, dataset contract, frontend architecture, medical claims, or deployment assumptions, stop and update the plan/ADR before implementing the larger change.

## Handoff format

Every meaningful handoff should report:

1. What changed.
2. Which files or interfaces changed.
3. What tests/checks ran and their result.
4. What remains unverified.
5. Any assumptions, risks, or required user action.

## Quality expectations

- Use deterministic seeds for split and model experiments.
- Keep method names and benchmark columns stable unless the change is documented.
- Validate user-controlled paths and uploaded images.
- Prefer explicit errors over silent fallback. A fallback must be visible to the user.
- Do not claim a model is trained or best unless the saved artifact and measured result support it.
- Treat all external data and archive contents as untrusted input.

## When to create an ADR

Create the next ADR when choosing a new frontend/backend boundary, changing the default dataset, introducing a deployment service, adding patient data persistence, or changing the evaluation protocol in a way that affects comparability.
