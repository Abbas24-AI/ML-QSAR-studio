from __future__ import annotations
import traceback
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from ..backend.config import QSARConfig
from ..backend.pipeline_runner import run_pipeline

class TrainWorker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: QSARConfig, use_notebook: bool, notebook_path: str | None):
        super().__init__()
        self.config = config
        self.use_notebook = use_notebook
        self.notebook_path = notebook_path

    def run(self):
        try:
            def _log(msg: str):
                self.log.emit(msg)
            res = run_pipeline(self.config, log_cb=_log, use_notebook=self.use_notebook, notebook_path=self.notebook_path)
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(traceback.format_exc())

class WorkerThread(QThread):
    def __init__(self, worker: TrainWorker):
        super().__init__()
        self.worker = worker

    def run(self):
        self.worker.run()
