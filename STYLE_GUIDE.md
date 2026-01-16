# Style Guide

This guide documents the Python style conventions used in this repository.

## General
- Follow PEP 8 where practical.
- Prefer explicit over implicit behavior.
- Keep functions small and focused.

## Naming
- Modules and functions: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_CASE`.

## Typing
- Use type hints for public APIs and non-trivial functions.
- Favor `Optional[T]` over `Union[T, None]` for readability.

## Docstrings
- Use concise docstrings for public classes and methods.
- Describe inputs, outputs, and side effects.

## Logging and Errors
- Log meaningful events at INFO level.
- Use DEBUG for high-frequency or diagnostic logs.
- Raise specific exceptions with actionable messages.

## Performance
- Avoid repeated expensive operations in tight loops.
- Use caching where it improves clarity and speed.
