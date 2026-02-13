# Contributing

PulseSuite is a community project, all contributions are more than welcome!

## What you can do

### Report bugs

Not only things can break, but also different people have
different use cases for the project. If you find anything that doesn't
work as expected or have suggestions, please open a new issue on our
[issue tracker](https://github.com/rahulranjansah/pulsesuite/issues).

### Participate in the chat

To be added soon.

### Improve the Documentation

Documentation can always be expanded and improved.
The docs are stored in text files under the `docs/source` directory,
and the Python classes and methods also feature inline docs
in the form of docstrings.

### Contribute your research scripts

We would love to give your PulseSuite scripts a home!
Please head to [our `contrib/` directory](https://github.com/rahulranjansah/pulsesuite/tree/main/contrib)
for further information.

### Fix bugs and add new features

Code contributions are welcome! If you are looking for a place to start,
check out the ["good-first-issue" label](https://github.com/rahulranjansah/pulsesuite/labels/good%20first%20issue)
on our issue tracker. Those tasks should be easier to fix than the others
and require less knowledge about the library.

## How to contribute

### Work from GitHub

GitHub makes it very easy to make small contributions
directly from your browser, without having to install any additional software.
To get familiar with the process, you can do
[this interactive GitHub training](https://lab.github.com/githubtraining/introduction-to-github).
Once you have finished it, you can edit the PulseSuite source files
straight from GitHub and open pull requests from your browser.

### Work locally

For more complex contributions, it will be more effective
to set up a local development environment and work from your own computer.

The most important thing is to understand how Git works. Git is a decentralized
version control system that preserves the history of the software, helps
tracking changes and allows for multiple versions of the code to exist
at the same time. To learn more, you can have a look at
[this Getting Started guide](https://docs.github.com/en/get-started/getting-started-with-git).

Finally, you might want to have an Integrated Development Environment (IDE)
that makes editing the source files easier.
If you are hesitant on what IDE or editor to use,
have a look at [PyCharm](https://www.jetbrains.com/pycharm/),
[Visual Studio Code](https://code.visualstudio.com/),
or [Spyder](https://www.spyder-ide.org/).

### Work from a cloud environment

There are some cloud options that give you
the flexibility of a powerful IDE with a terminal,
all from your web browser so you don't have to install anything.
Two popular cloud environments for development are
[the GitHub web editor](https://github.dev/rahulranjansah/pulsesuite)
and [Gitpod](https://gitpod.io/#https://github.com/rahulranjansah/pulsesuite/).

## Command line instructions

In the sections below you can read step-by-step guides to
perform common development tasks on the command line.
All these instructions are meant for UNIX-like operating systems,
for example:

- Linux (any distribution will work)
- macOS
- Windows Subsystem for Linux (WSL)

### Set up a development environment

To set up a development environment you will need Git and Python
up and running. You should only need to do this once!

Start by setting up Git:

1. [Install Git](https://git-scm.com/) on your computer.
2. [Sign up to GitHub](https://github.com/).
3. [Fork PulseSuite](https://help.github.com/articles/fork-a-repo/).
4. [Clone your fork](https://help.github.com/articles/cloning-a-repository/)
   (remote name will be `origin`)
5. Add an `upstream` remote with `git remote add upstream https://github.com/rahulranjansah/pulsesuite.git`
   and fetch its information with `git fetch upstream`
6. Set your `main` branch to track `upstream` using `git branch --set-upstream-to=upstream/main`

Next, set up your development environment. PulseSuite uses
[uv](https://docs.astral.sh/uv/) for dependency management:

6. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
7. Sync all dependencies (core + test + doc) in one step:
   `uv sync --all-extras`

Alternatively, with pip:

6. Create a Python virtual environment using `python -m venv .venv`
7. Activate it using `source .venv/bin/activate`
8. Install the code in development mode using `pip install -e ".[test,doc]"`

And with this, you will be ready to start contributing!

Common development commands:

```console
$ uv run pytest                       # run the test suite
$ uv run pytest -k coulomb            # run a subset of tests
$ uv run ruff check src/ tests/       # lint
$ uv run ruff check --fix src/ tests/ # auto-fix lint issues
$ uv run ruff format src/ tests/      # format code
```

A `justfile` is also provided as an optional convenience wrapper.
If you have [just](https://github.com/casey/just) installed, run
`just --list` to see available commands.

### Pull request workflow

Every time you want to contribute some code or documentation to PulseSuite,
you will need to follow these steps:

1. Make sure that your `main` branch is up to date: `git switch main`
   (or `git checkout main`) and `git pull --rebase upstream main`
2. Create a new branch using `git switch --create descriptive-name-of-my-change`.
3. Make your changes!
4. Commit your changes using `git commit -m 'Add new awesome feature'`.
5. [Push to your fork](https://help.github.com/articles/pushing-to-a-remote/).
6. [Open a pull request!](https://help.github.com/articles/creating-a-pull-request/)

One branch corresponds to one pull request. This means that,
if you keep committing and pushing changes to the same branch,
the pull request will update automatically (you don't need to open a new one!).

When your pull request is merged, you can:

7. Update your `main` branch again: `git switch main` and `git pull --rebase upstream main`
8. Delete your local branch: `git branch --delete descriptive-name-of-my-change`
9. Refresh your fork: `git fetch --prune origin` and `git push origin main`

Remember that, whenever you want to start a new pull request,
you need to start from step 1.

### Build the documentation

Using uv (recommended):

```console
$ cd docs && uv run make html
```

Or with pip:

```console
(.venv) $ pip install -e ".[doc]"
(.venv) $ cd docs
(.venv) $ make html
```

After the build, the HTML output will be in `docs/_build/html`.
You can preview it locally:

```console
$ python -m http.server -d docs/_build/html
Serving HTTP on 0.0.0.0 port 8000 ...
```

And point your browser to <http://0.0.0.0:8000>.

### Code style and naming conventions

PulseSuite is a Python port of a production Fortran codebase. We
**intentionally preserve the original Fortran naming conventions**
rather than converting everything to PEP 8 snake_case. This is a
deliberate design choice, not an oversight.

**Why we do this:**

- **Traceability.** Researchers familiar with the original Fortran code
  can locate the same subroutines and variables by name. A function
  called `InitializeSBE` in Fortran is still `InitializeSBE` in Python.
- **Literature consistency.** Variable names like `Ex`, `Ey`, `Nk`,
  `VC`, `Rho`, and `Ee` match the notation used in the published papers
  (Gulley & Huang 2019, 2022) and textbooks (Haug & Koch). Renaming
  them to `electric_field_x` would obscure the connection to the
  physics.
- **Backward compatibility.** Existing simulation scripts and parameter
  files reference these names. Renaming would break downstream usage.

**What this means in practice:**

- Function names use **PascalCase** to match Fortran subroutine names
  (e.g., `CalcPHost`, `SetParamsGaAs`, `InitializeCoulomb`).
- Physics variables use **short names** from the literature
  (e.g., `Ee`, `Eh`, `ky`, `VC`, `Nk`).
- Module names like `SBEs.py`, `rhoPJ.py`, `PSTD3D/` preserve the
  original Fortran module structure.
- Several `ruff` lint rules are suppressed in `pyproject.toml` for this
  reason (e.g., `E741` for "ambiguous" variable names like `I`, `l`,
  `O` which are standard physics notation).

When contributing new code that extends existing modules, **follow the
conventions of the surrounding code**. For entirely new utilities with
no Fortran counterpart, PEP 8 snake_case is acceptable.

### Make code changes

You want to contribute new features or fix existing behavior? You are awesome!
Before rushing out though, make sure if the new feature you propose
is within the scope of the library (best thing is to ask in
[email](f1rahulranjan@gmail.com))
or that the fix you want to apply has a corresponding issue in the issue tracker.

Apart from all the steps described above, you need to have these extra things in mind:

1. Add tests to your code. You have lots of examples in the `tests/` directory
   to get inspiration from. All new features and fixes should be tested,
   and in the ideal case the coverage rate should increase or stay the same.
2. To check if your code is correct, run `uv run pytest`. To also lint
   and format-check, run `uv run ruff check src/ tests/` and
   `uv run ruff format --check src/ tests/`.
3. Notice that you can run a subset of the tests by
   passing extra arguments to pytest, for example
   `uv run pytest -k "coulomb"`

Parts of this contributing guide have been adapted from the
[poliastro](https://github.com/poliastro/poliastro) contributing guide.

Automatic services will ensure your code works
on all the supported operating systems and Python versions.
