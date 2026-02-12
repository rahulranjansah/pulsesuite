# Coding Rules and Instructions

1.  *Test-Driven Development (TDD) with pytest:* Always write a failing test *before* writing implementation code (Red-Green-Refactor). Use `pytest` and `pytest-fixtures` for test setup, execution, and teardown.
2.  *KISS (Keep It Simple, Stupid):* Favor the simplest solution that meets the requirements.
3.  *DRY (Don't Repeat Yourself):* Avoid code duplication. Extract reusable logic into functions or classes.
4.  *Standard Libraries and Tools:* Utilize standard Python libraries (like `datetime` for date/time, `requests` for HTTP requests, and `logging`) and external libraries, including `BeautifulSoup4` for HTML parsing, to avoid reinventing the wheel.  Favor well-maintained and widely-used libraries.
5.  *YAGNI (You Ain't Gonna Need It):* Don't implement features or functionality unless they are currently required.
6.  *SOLID Principles & Extensibility:* Adhere to SOLID principles, promoting maintainability, testability, and future extension. Consider potential future requirements when designing classes and modules.
7.  *PEP 8 Style Guide:* Follow the PEP 8 style guide for Python code.
8.  *Type Hints:* Use type hints for all function parameters and return values.
9.  *Docstrings:* Write clear and concise docstrings for all classes, functions, and methods, explaining their purpose, parameters, and return values.
10. *Small Units of Work:* Keep functions and classes small, focused, and with a single, well-defined responsibility (combines original 10 & 11, and reinforces SOLID).
11. *Modularity:* Design the system as a collection of independent, modular components that can be easily reused and tested.
12. *Parameterized Queries:* Prevent SQL injection vulnerabilities by always using parameterized queries when interacting with the database.
13. *JSONB for Flexible Data:* Use JSONB for storing flexible or semi-structured data in PostgreSQL.
14. *Centralized Logging:* Use the `logging` module to log to standard output. Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) to categorize log messages.
15. *Centralized Metrics:* Track key metrics using a suitable data structure (e.g., a dictionary) and provide a mechanism to display a summary of these metrics.
16. *Configuration and Containerization:* Use a `config.py` file for application configuration and to load environment variables (from a `.env` file). Use `Dockerfile` and `docker-compose.yml` for containerization.
17. *Utilize utils.py:* Use a `utils.py` file for utility and helper functions that are not specific to a particular module.
18. *Test Data:* Use fixtures in `tests/fixtures` for sample data to be used in tests.
19. *Efficient Code:* Write efficient code, avoiding unnecessary computations, loops, or database queries.
20. *Meaningful Return Values:* Ensure that functions return meaningful and predictable values, including appropriate error indicators when necessary.
21. *Follow Python 3.11+:* Use Python 3.11 or a later version.
22. *Makefile Automation:* Use a `Makefile` for automating tasks such as building, running, testing, and deploying the application.
23. *Handle Database Errors:* Handle potential database errors (e.g., connection errors, query errors) gracefully, providing informative error messages and preventing application crashes.
24. *Security and Secret Handling:* Never store secrets (passwords, API keys) directly in the code. Use environment variables (loaded via `.env` and accessed through `config.py`) or a dedicated secrets management solution.
25. *Prioritize Instructions:* Adhere precisely to the provided instructions and specifications. If ambiguity exists, ask clarifying questions *before* making assumptions.
26. *Comprehensive Documentation:* Provide clear, concise, and up-to-date documentation. This includes docstrings (for classes, functions, and methods), in-line comments where necessary to explain complex logic, and README files to explain the project's purpose, setup, and usage.
27. *ORM and Database Interactions:* Use `SQLAlchemy` for database interactions and object-relational mapping (ORM).  Define database models using SQLAlchemy's declarative base.
28. *Data Validation with Pydantic:* Use `PydanticV2` for data validation, schema definition, and settings management.
29. *Asynchronous Programming (if needed):* If the API or application requires asynchronous operations, use `asyncio` and `async`/`await` syntax.
30. *RESTful API Design:* If building a REST API, adhere to RESTful principles (HTTP methods, resource URLs, status codes, JSON).
31. *API Versioning:* Implement a clear API versioning strategy (e.g., `/v1/`).
32. *Rate Limiting (If Applicable):* Implement rate limiting to prevent abuse.
33. *Authentication and Authorization (If Applicable):* Clearly define authentication and authorization methods.
34. *Robust Error Handling:* Handle exceptions, return informative errors, and log errors with context.
35. *Dependency Management:* Use `pip` with a `requirements.txt` file.
36. *Automated Code Formatting:* Use `black` for automatic code formatting.
37. *Static Analysis with Linting:* Use `flake8` or `pylint`.
38. *Resource Management with Context Managers:* Use context managers (`with` statement) for resources.
39. *Favor Immutability:* Prefer immutable data structures when appropriate.
40. *Makefile Structure:* Include targets for build, run, test, lint, format, clean, db-up, db-down.

---

## Fortran → Python Naming Philosophy

This codebase serves **physicists migrating from Fortran**. The following
naming rules override generic PEP 8 conventions where they conflict:

1.  **Keep Numerical Recipes names.** Users grep for `bsstep`, `stifbs`,
    `pzextr`, `rkqs_dp` — these carry meaning in the community. Renaming
    `rkqs_dp` to `adaptive_rk_step_real` buys nothing and breaks the mental
    mapping.

2.  **Keep type-suffixed wrappers.** `rkck_dp` vs `rkck_dpc` tells the
    caller at a glance what data type they're working with. The one-line
    wrappers cost zero performance and make call sites self-documenting.

3.  **Keep camelCase public names.** `arrayCopy`, `outerprod`, `diagAdd`
    match the Fortran module API. Consistency within the domain matters more
    than style guides. (NumPy has `loadtxt`, SciPy has `odeint`.)

4.  **Fix genuine Python traps.**
    - 0-based indexing — 1-based was a trap in Python, not just style.
    - Collapse `assert_eq2/3/4/n` into one `assertEq(*args)` — genuinely
      simpler, not just stylistically different.
    - Duck-typed internals (`_rkck`, `_rk4`) — implementation is Pythonic,
      API preserves Fortran familiarity.

5.  **Remove dead cruft.** Unused imports, dead code paths, backup copies —
    these serve nobody. If it helps a Fortran user find and trust the
    equivalent routine, keep it. If it's leftover mechanical noise, remove it.

**Rule of thumb:** *familiarity for the domain user* > *style guide purity*.
