# Third-party code

`stable-worldmodel/` is an unmodified snapshot of
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (MIT license, see
`stable-worldmodel/LICENSE`) taken from the development branch that this work was built on.
It provides the environments, HDF5 dataset readers, the CEM solver, the shooting cost evaluator
and the closed-loop world evaluator. Only run logs were removed from the snapshot.

Install it in editable mode (`pip install -e third_party/stable-worldmodel`); the PyPI release
does not contain the planning subsystem used by `salt/eval.py`.
