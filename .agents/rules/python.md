# Python Engineering Rules

These rules govern the development of the LLM Ad Placement Testing project. Follow these guidelines to ensure consistency, testability, and maintainability.

## Code Quality
* Follow PEP 8.
* Use clear, descriptive names.
* Prefer small, focused functions.
* Keep modules cohesive.
* Avoid unnecessary abstraction.
* Prefer composition over inheritance when appropriate.
* Use type hints for public functions, methods, and important internal interfaces.
* Keep functions deterministic where practical.
* Avoid hidden global state.
* Avoid duplicated business logic.
* Keep side effects isolated.

## SOLID Principles
Apply SOLID pragmatically rather than mechanically.

**Single Responsibility Principle**
* A class/module should have one primary responsibility.
* Separate business logic from I/O, persistence, API calls, configuration, and presentation where appropriate.

**Open/Closed Principle**
* New behavior should preferably be added through extension rather than repeatedly modifying stable core logic.
* Use strategies, interfaces, registries, or dependency injection when they genuinely simplify extension.

**Liskov Substitution Principle**
* Implementations of an abstraction must honor the abstraction's expected behavior and contract.
* Do not create subclasses that violate assumptions made by callers.

**Interface Segregation Principle**
* Prefer small, focused interfaces over large interfaces containing unrelated functionality.
* Do not force implementations to depend on methods they do not need.

**Dependency Inversion Principle**
* High-level business logic should not depend directly on concrete infrastructure implementations.
* Depend on abstractions where there is a meaningful reason to support multiple implementations or easier testing.

> [!WARNING]
> Do not introduce interfaces or abstractions solely to satisfy SOLID mechanically. 

## Design Pattern Rules
Use established design patterns when they solve a real problem. Prefer simple solutions first.

Good candidates include:
* Strategy Pattern for interchangeable algorithms or behaviors
* Factory Pattern when object creation involves meaningful selection/configuration logic
* Adapter Pattern when integrating incompatible external interfaces
* Repository Pattern when persistence needs to be abstracted from business logic
* Dependency Injection for replaceable dependencies and testability
* Observer/Event-driven patterns when multiple components need to react to events
* Template Method only when there is a genuine shared algorithmic structure

Do not introduce a design pattern merely because the pattern exists. Before introducing a pattern, consider:
1. Is there actually a recurring design problem?
2. Does the pattern reduce coupling?
3. Does it improve extensibility or testability?
4. Does it make the code easier to understand?
5. Is the added abstraction justified by the complexity?

Prefer straightforward Python code over unnecessary design-pattern-heavy code.

## Dependency and Architecture Rules
* Keep dependencies flowing in a clear direction.
* Avoid circular imports and circular architectural dependencies.
* Keep infrastructure concerns separate from domain/business logic where practical.
* Avoid modules that become "god modules" containing unrelated responsibilities.
* Avoid classes that become "god objects."
* Keep public interfaces small and stable.
* Do not expose implementation details unnecessarily.
* Prefer dependency injection for components that need to be mocked or replaced in tests.
* Keep configuration separate from business logic.
* Do not couple core logic directly to a specific external service when multiple implementations may reasonably exist.

## Testing Rules
* New business logic should have appropriate tests.
* Prefer testing behavior and contracts rather than implementation details.
* Make external dependencies mockable when practical.
* Keep unit tests independent from network calls and external services.
* Add integration tests where component interaction is important.
* Do not sacrifice architectural clarity solely to make testing easier.

## Refactoring Rules
When modifying existing code:
* First understand the existing architecture (consult `docs/architecture.md`).
* Prefer incremental refactoring.
* Do not rewrite working code unnecessarily.
* Preserve existing behavior unless the task explicitly requires changing it.
* Avoid combining unrelated refactoring with feature work.
* If a SOLID/design-pattern violation is minor and does not affect maintainability, do not introduce unnecessary abstraction just to fix it.
* If a change creates significant technical debt, mention it explicitly.

---

## Important Agent Behavior
Before making architectural changes:
1. Read `AGENTS.md`.
2. Read `docs/architecture.md` when the change affects multiple modules or system design.
3. Check these Python rules.
4. Inspect existing implementations before introducing new abstractions.
5. Reuse existing patterns and abstractions when appropriate.
6. Do not create duplicate abstractions for functionality that already exists.

When choosing between two implementations, prefer the one with:
* Lower coupling
* Higher cohesion
* Simpler interfaces
* Better testability
* Fewer unnecessary abstractions
* Clearer ownership of responsibilities
* Easier future extension

> [!IMPORTANT]
> Do not apply SOLID or design patterns dogmatically. The goal is maintainable, understandable, extensible Python code—not maximum abstraction.
