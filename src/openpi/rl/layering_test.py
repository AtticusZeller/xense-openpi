"""Layering of the online-RL package: algorithms depend on the shared layers, never the reverse.

Only production modules are checked. Tests may reach across layers: the shared action-space
tests, for one, compare against RLT's TacXense reference.
"""

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[2]  # the directory holding the openpi package
_OPENPI = _SRC / "openpi"
_RL = _OPENPI / "rl"


def _module_name(path: pathlib.Path) -> str:
    return ".".join(path.relative_to(_SRC).with_suffix("").parts)


def _imports(path: pathlib.Path) -> set[str]:
    """Every module ``path`` imports, with ``from a import b`` also counted as ``a.b``."""
    package = _module_name(path).split(".")[:-1]
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            modules.add(module)
            modules.update(f"{module}.{alias.name}" for alias in node.names)
    return modules


def _production_modules(root: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in root.rglob("*.py") if not p.name.endswith("_test.py") and p.name != "conftest.py"]


def _violations(files: list[pathlib.Path], forbidden: str) -> list[str]:
    return sorted(
        f"{_module_name(path)} imports {module}"
        for path in files
        for module in _imports(path)
        if module == forbidden or module.startswith(f"{forbidden}.")
    )


def test_shared_layers_do_not_import_algorithms():
    shared = [p for p in _production_modules(_RL) if _RL / "algos" not in p.parents]
    assert shared, "no shared-layer modules found"
    assert _violations(shared, "openpi.rl.algos") == []


def test_openpi_outside_rl_does_not_import_rl():
    outside = [p for p in _production_modules(_OPENPI) if _RL not in p.parents]
    assert outside, "no modules outside openpi.rl found"
    assert _violations(outside, "openpi.rl") == []
