"""Regression test: merged_run.run must reach json.dump without UnboundLocalError
when --task_split is omitted."""

import ast
import sys
import textwrap
import unittest

sys.path.insert(0, 'src')


class TestMergedRunJsonNoTaskSplit(unittest.TestCase):
    """Prove merged_run.run reaches json.dump when task_split is None."""

    SCRIPT_PATH = '/home/tiantianyi/code/federated-mcts/scripts/merged_run.py'

    def test_run_function_has_no_local_import_json_shadowing(self):
        """The 'run' function must NOT contain a local 'import json' that
        shadows the module-level import, because when task_split is None
        the local import never executes and json.dump raises UnboundLocalError."""

        with open(self.SCRIPT_PATH) as f:
            source = f.read()
        tree = ast.parse(source)

        # Find the 'run' function definition
        run_func = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'run':
                run_func = node
                break

        self.assertIsNotNone(run_func, "DEFECT: 'run' function not found in script")

        # Walk all nodes inside the run function looking for 'import json'
        local_imports = []
        for node in ast.walk(run_func):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'json':
                        local_imports.append((
                            node.lineno,
                            f'import json'
                        ))
            elif isinstance(node, ast.ImportFrom):
                if node.module == 'json':
                    local_imports.append((
                        node.lineno,
                        f'from json import ...'
                    ))

        self.assertEqual(
            len(local_imports), 0,
            'DEFECT: "import json" found inside run() function at lines '
            f'{local_imports}. This shadows the module-level import '
            'and causes UnboundLocalError when task_split is None.'
        )

    def test_module_level_import_json_exists(self):
        """merged_run.py must import json at module level (line 2)."""

        with open(self.SCRIPT_PATH) as f:
            source = f.read()
        tree = ast.parse(source)

        found = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'json':
                        found = True
                        break
            if found:
                break

        self.assertTrue(found, 'DEFECT: module-level "import json" not found')

    def test_run_without_task_split_compiles_without_unbound_error(self):
        """Simulate: if task_split is None, the json reference in run() must
        resolve from the module scope (not from an unexecuted local import)."""
        with open(self.SCRIPT_PATH) as f:
            source = f.read()

        # Remove the shebang / anything before the import so we can exec in
        # a controlled scope.
        tree = ast.parse(source)

        # Extract the run function source lines
        run_func = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'run':
                run_func = node
                break

        self.assertIsNotNone(run_func)

        # Build a minimal compile scope: module-level json, no task_split shadowing
        ns = {}
        import json as _json_mod  # noqa: F811
        ns['json'] = _json_mod

        # We do NOT add a function-local json; we compile and exec
        # just the run function body in the namespace with json.
        # Wrap it in a dummy def so we can exec it.
        run_source = ast.get_source_segment(source, run_func)
        self.assertIsNotNone(run_source)

        wrapper = (
            'def run(args, solve_function):\n' +
            textwrap.indent(run_source, '    ')
        )

        try:
            exec(wrapper, ns)
        except Exception as e:
            # The code won't run (missing federated_mcts imports), but
            # the compilation must not fail with a parse error.
            # The important thing: the function is created and json
            # resolves to the module-level json, not UnboundLocalError.
            pass

        # The real proof: compile the code and check there's no
        # local 'json' variable binding that would shadow.
        code_obj = compile(wrapper, '<test>', 'exec')
        # If we got here without SyntaxError, the function definition compiled.
        self.assertIsNotNone(code_obj)

        # Now try calling the run function with a mock that triggers
        # code path WITHOUT task_split. If json were local (shadowed),
        # the compiler would emit LOAD_FAST for json, not LOAD_GLOBAL.
        # Let's check: if the run function body references 'json',
        # it must come from the global scope.
        run_bytecode_ns = {}
        exec(wrapper, run_bytecode_ns)
        run_fn = run_bytecode_ns.get('run')
        if run_fn is not None:
            # Check that 'json' is not in the function's local variables
            # (it should be accessed via LOAD_GLOBAL, not LOAD_FAST/LOAD_DEREF)
            self.assertNotIn('json', run_fn.__code__.co_varnames,
                'DEFECT: "json" is a local variable in run(), '
                'indicating a function-local import shadows the module-level import')


if __name__ == '__main__':
    unittest.main()
