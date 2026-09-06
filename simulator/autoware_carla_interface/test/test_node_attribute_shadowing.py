# Copyright 2024 Tier IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guard against ``__init__`` attributes shadowing methods of the same class.

``self._initialize_localization`` was once both a bool parameter and the method
that calls ``/api/localization/initialize``; the attribute won, and the node died
with ``TypeError: 'bool' object is not callable`` the moment a mission arrived.
The node cannot be imported without ``rclpy``/Autoware messages, so this checks
the source with ``ast`` instead of instantiating it.
"""

import ast
from pathlib import Path

import pytest

NODE_SOURCES = sorted(
    (Path(__file__).resolve().parents[1] / "src" / "autoware_carla_interface").rglob("*.py")
)


def _shadowed_methods(class_node: ast.ClassDef) -> set:
    """Return method names that ``__init__`` also assigns as ``self.<name> = ...``."""
    methods = {
        n.name
        for n in class_node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    init = next(
        (n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return set()
    assigned = set()
    for node in ast.walk(init):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    return methods & assigned


@pytest.mark.parametrize("path", NODE_SOURCES, ids=lambda p: p.name)
def test_init_does_not_shadow_methods(path):
    tree = ast.parse(path.read_text())
    for class_node in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        shadowed = _shadowed_methods(class_node)
        assert not shadowed, (
            f"{path.name}: {class_node.name}.__init__ assigns {sorted(shadowed)}, "
            "which shadows the method(s) of the same name"
        )
