# pytket-qujax

This repository contains the pytket-qujax extension, using CQC's
[pytket](https://tket.quantinuum.com/api-docs/index.html) quantum SDK.
The other pytket extensions can be found [here](https://tket.quantinuum.com/api-docs/extensions)

[Pytket](https://tket.quantinuum.com/api-docs/index.html) is a Python module for interfacing
with CQC tket, a set of quantum programming tools.

[qujax](https://github.com/CQCL/qujax) is a pure [JAX](https://github.com/google/jax)
quantum simulator. pytket-qujax is an extension to [pytket](https://tket.quantinuum.com/api-docs/index.html)
that allows [pytket](https://tket.quantinuum.com/api-docs/index.html) circuits to
be converted to [qujax](https://github.com/CQCL/qujax) for fast (classical) simulation and automatic differentiation.

Some useful links:
- [Documentation](https://tket.quantinuum.com/extensions/pytket-qujax/index.html)
- [PyPI](https://pypi.org/project/pytket-qujax/)
- [qujax](https://github.com/CQCL/qujax)
- [pytket-qujax example notebook (VQE)](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_heisenberg_vqe.ipynb)
- [pytket-qujax example notebook (QAOA with `symbol_map`)](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax_qaoa.ipynb)
- [pytket-qujax example notebook (classifier)](https://github.com/CQCL/pytket/blob/main/examples/pytket-qujax-classification.ipynb)
- [qujax example notebooks](https://github.com/CQCL/qujax/tree/main/examples)


## Getting started

`pytket-qujax` is available for Python 3.9, 3.10 and 3.11, on Linux and MacOS.
To install, run:

```shell
pip install pytket-qujax
```

This will install `pytket` if it isn't already installed, and add new classes
and methods into the `pytket.extensions` namespace.

## Bugs and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-qujax/issues).

## Development

To install an extension in editable mode, simply change to its subdirectory
within the `modules` directory, and run:

```shell
pip install -e .
```

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `develop` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting

All code should be formatted using
[black](https://black.readthedocs.io/en/stable/), with default options. This is
checked on the CI. The CI is currently using version 20.8b1.

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. Because of the way
extension modules embed themselves into the `pytket` namespace this is a little
complicated, but it should be sufficient to run the script `modules/mypy-check`
(passing as a single argument the root directory of the module to test). The
script requires `mypy` 0.800 or above.

#### Linting

We use [pylint](https://pypi.org/project/pylint/) on the CI to check compliance
with a set of style requirements (listed in `.pylintrc`). You should run
`pylint` over any changed files before submitting a PR, to catch any issues.

### Tests

To run the tests for a module:

1. `cd` into that module's `tests` directory;
2. ensure you have installed `pytest`, `hypothesis`, and any modules listed in
the `test-requirements.txt` file (all via `pip`);
3. run `pytest`.

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
