from __future__ import annotations
import nbformat
from types import SimpleNamespace
from typing import Any, Dict, Optional, Callable
import builtins
import os

def load_notebook_namespace(ipynb_path: str) -> Dict[str, Any]:
    """Extract code cells and exec into a fresh namespace.
    Security note: this executes arbitrary code from the notebook.
    """
    nb = nbformat.read(ipynb_path, as_version=4)
    code = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            code.append(cell.source)
    src = "\n\n".join(code)
    ns: Dict[str, Any] = {"__name__": "__notebook__"}
    exec(compile(src, ipynb_path, "exec"), ns, ns)
    return ns

def run_notebook_pipeline(ipynb_path: str, config_updates: Dict[str, Any],
                          log_cb: Optional[Callable[[str], None]] = None) -> None:
    ns = load_notebook_namespace(ipynb_path)
    if "CONFIG" in ns and isinstance(ns["CONFIG"], dict):
        ns["CONFIG"].update(config_updates)
    else:
        raise RuntimeError("Notebook does not define CONFIG dict.")

    # Optional: set output directory and ensure exists
    outdir = ns["CONFIG"].get("output_dir")
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    if "main" not in ns or not callable(ns["main"]):
        raise RuntimeError("Notebook does not define a callable main().")

    # Run
    ns["main"]()
