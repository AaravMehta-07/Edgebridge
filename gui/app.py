# Full GUI integrated with backend (PyQt6 + pyqtgraph)
import os
import sys
import json
import csv
import traceback
from pathlib import Path
from datetime import datetime

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QGroupBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QMessageBox, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import pyqtgraph as pg

# backend imports
try:
    from edgebridge.core import Benchmark
    from edgebridge.converters import TorchConverter, onnx_to_tflite, torch_to_tflite
    from edgebridge.optimizer import Optimizer
    import torch
    import_types_missing = False
except Exception:
    Benchmark = None
    TorchConverter = None
    onnx_to_tflite = None
    torch_to_tflite = None
    Optimizer = None
    import_types_missing = True

class Worker(QThread):
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(bool, str)  # success, message

    def __init__(self, task_spec):
        super().__init__()
        self.task = task_spec

    def run(self):
        try:
            self.log("Starting task...")
            model_path = self.task.get("model_path")
            backend = self.task.get("backend")
            do_quant = self.task.get("quant")
            quant_mode = self.task.get("quant_mode")
            do_prune = self.task.get("prune")
            prune_amount = self.task.get("prune_amount")
            do_distill = self.task.get("distill")
            runs = int(self.task.get("runs", 20))
            device = self.task.get("device", "cpu")

            if not model_path:
                raise ValueError("No model selected.")
            ext = Path(model_path).suffix.lower()
            working_model_path = model_path
            used_backend = backend.lower()

            if ext in (".pt", ".pth"):
                self.log("Loading PyTorch model...")
                if import_types_missing:
                    raise RuntimeError("PyTorch is not available in the environment.")
                model = torch.load(model_path, map_location=device)
                if isinstance(model, dict):
                    raise RuntimeError("State dict loaded. Please provide a scripted/traced .pt model.")
                opt = Optimizer(model, sample_input=None)
                if do_prune:
                    self.log(f"Pruning model (amount={prune_amount}) ...")
                    model = opt.prune(amount=prune_amount)
                if do_quant:
                    self.log(f"Quantizing model (mode={quant_mode}) ...")
                    model = opt.quantize(mode=quant_mode)
                # Save TorchScript
                try:
                    scripted = torch.jit.script(model)
                    scripted_path = str(Path(model_path).with_suffix(".optim.torchscript.pt"))
                    scripted.save(scripted_path)
                    working_model_path = scripted_path
                    self.log(f"Saved optimized TorchScript at: {scripted_path}")
                except Exception as e:
                    self.log(f"Could not script model: {e}")
                    raise

                if used_backend == "onnx":
                    self.log("Exporting optimized model to ONNX...")
                    dummy = torch.randn(1, 3, 224, 224)
                    onnx_path = str(Path(model_path).with_suffix(".optim.onnx"))
                    TorchConverter.to_onnx(model, dummy, export_path=onnx_path)
                    working_model_path = onnx_path
                    self.log(f"Saved ONNX at {onnx_path}")

                if used_backend == "tflite":
                    self.log("Converting optimized model to TFLite (via ONNX)...")
                    tmp_onnx = str(Path(model_path).with_suffix(".tmp.onnx"))
                    TorchConverter.to_onnx(model, torch.randn(1,3,224,224), export_path=tmp_onnx)
                    tflite_path = str(Path(model_path).with_suffix(".optim.tflite"))
                    if onnx_to_tflite is None:
                        raise RuntimeError("onnx_to_tflite is not available (missing onnx/onnx-tf/tensorflow).")
                    onnx_to_tflite(tmp_onnx, out_path=tflite_path)
                    working_model_path = tflite_path
                    self.log(f"Saved TFLite at {tflite_path}")

            elif ext == ".onnx":
                self.log("ONNX selected.")
                if used_backend == "tflite":
                    self.log("Converting ONNX -> TFLite...")
                    if onnx_to_tflite is None:
                        raise RuntimeError("onnx_to_tflite requires onnx/onnx-tf/tensorflow installed.")
                    tflite_path = str(Path(model_path).with_suffix(".tflite"))
                    onnx_to_tflite(model_path, out_path=tflite_path)
                    working_model_path = tflite_path
                    self.log(f"Saved TFLite at {tflite_path}")
                if do_quant:
                    self.log("ONNX quantization is not implemented in GUI. Consider converting to TFLite and quantizing there.")

            elif ext == ".tflite":
                self.log("TFLite model selected. Proceeding to benchmark.")
                working_model_path = model_path
            else:
                raise RuntimeError("Unsupported model file type. Supported: .pt .pth .onnx .tflite")

            # Run benchmark
            if Benchmark is None:
                raise RuntimeError("Benchmark backend not available.")
            bench = Benchmark(working_model_path, backend=used_backend, device=device)
            results = bench.run(runs=runs)
            results["model_path"] = working_model_path
            results["timestamp"] = datetime.utcnow().isoformat() + "Z"
            self.result_signal.emit(results)
            self.finished_signal.emit(True, "Completed successfully.")
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"[ERROR] {e}\n{tb}")
            self.finished_signal.emit(False, str(e))

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_signal.emit(f"[{ts}] {msg}")


class EdgeBridgeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EdgeBridge — Model Optimizer")
        self.setWindowIcon(QtGui.QIcon())
        self.setGeometry(120, 60, 1320, 800)

        central = QWidget()
        root_layout = QHBoxLayout()
        central.setLayout(root_layout)
        self.setCentralWidget(central)

        left = QVBoxLayout()
        title = QLabel("EdgeBridge")
        title.setObjectName("title")
        subtitle = QLabel("Model optimizer & benchmarking — Aarav Mehta")
        subtitle.setStyleSheet("color: #9da0a6;")
        left.addWidget(title)
        left.addWidget(subtitle)
        left.addSpacing(8)

        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Select model (.pt / .onnx / .tflite)")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_model)
        h = QHBoxLayout()
        h.addWidget(self.model_path_input)
        h.addWidget(browse_btn)
        left.addLayout(h)

        left.addSpacing(6)
        left.addWidget(QLabel("Backend"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["onnx", "tflite", "torchscript"])
        left.addWidget(self.backend_combo)

        opt_group = QGroupBox("Optimizations")
        opt_layout = QVBoxLayout()
        self.quant_check = QCheckBox("Quantization (PyTorch dynamic/static)")
        quant_mode_layout = QHBoxLayout()
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["dynamic", "static"])
        quant_mode_layout.addWidget(QLabel("Mode:"))
        quant_mode_layout.addWidget(self.quant_combo)
        opt_layout.addWidget(self.quant_check)
        opt_layout.addLayout(quant_mode_layout)

        self.prune_check = QCheckBox("Prune model")
        prune_layout = QHBoxLayout()
        self.prune_spin = QSpinBox()
        self.prune_spin.setRange(1, 80)
        self.prune_spin.setValue(30)
        prune_layout.addWidget(QLabel("Amount (%)"))
        prune_layout.addWidget(self.prune_spin)
        opt_layout.addWidget(self.prune_check)
        opt_layout.addLayout(prune_layout)

        self.distill_check = QCheckBox("Distill (teacher→student) [advanced]")
        opt_layout.addWidget(self.distill_check)
        opt_group.setLayout(opt_layout)
        left.addWidget(opt_group)

        bench_group = QGroupBox("Benchmark")
        bench_layout = QVBoxLayout()
        bench_layout.addWidget(QLabel("Runs (iterations)"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 500)
        self.runs_spin.setValue(20)
        bench_layout.addWidget(self.runs_spin)
        bench_group.setLayout(bench_layout)
        left.addWidget(bench_group)

        self.run_btn = QPushButton("🚀 Run")
        self.run_btn.clicked.connect(self.on_run)
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.clicked.connect(self.on_export)
        self.export_btn.setEnabled(False)
        left.addWidget(self.run_btn)
        left.addWidget(self.export_btn)
        left.addStretch()

        right = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        right.addWidget(self.table, 2)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#141417")
        self.plot.addLegend(offset=(10,10))
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("left", "Latency (ms)")
        self.plot.setLabel("bottom", "Iteration")
        right.addWidget(self.plot, 3)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(140)
        right.addWidget(self.log_box, 1)

        root_layout.addLayout(left, 3)
        root_layout.addLayout(right, 7)

        styles_path = Path(__file__).parent / "styles.qss"
        if styles_path.exists():
            try:
                with open(styles_path, "r") as f:
                    self.setStyleSheet(f.read())
            except Exception:
                pass

        self.worker = None
        self.latest_results = None

    def browse_model(self):
        file_types = "Model files (*.pt *.pth *.onnx *.tflite);;All files (*)"
        fp, _ = QFileDialog.getOpenFileName(self, "Select model file", str(Path.cwd()), file_types)
        if fp:
            self.model_path_input.setText(fp)

    def append_log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def on_run(self):
        model_path = self.model_path_input.text().strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.critical(self, "Error", "Please select a valid model file.")
            return

        task = {
            "model_path": model_path,
            "backend": self.backend_combo.currentText(),
            "quant": self.quant_check.isChecked(),
            "quant_mode": self.quant_combo.currentText(),
            "prune": self.prune_check.isChecked(),
            "prune_amount": self.prune_spin.value() / 100.0,
            "distill": self.distill_check.isChecked(),
            "runs": self.runs_spin.value(),
            "device": "cpu",
        }

        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.table.setRowCount(0)
        self.plot.clear()
        self.log_box.clear()

        self.worker = Worker(task)
        self.worker.log_signal.connect(self.append_log)
        self.worker.result_signal.connect(self.on_result)
        self.worker.finished_signal.connect(self.on_finished)
        self.append_log("Queued task... starting worker thread.")
        self.worker.start()

    def on_result(self, results):
        self.latest_results = results
        self.table.setRowCount(0)
        rows = [
            ("Backend", results.get("backend", ""),""),
            ("Device", results.get("device",""), ""),
            ("Avg Latency (ms)", results.get("avg_latency_ms", ""), "ms"),
            ("Median (ms)", results.get("median_ms",""), "ms"),
            ("P95 (ms)", results.get("p95_ms",""), "ms"),
            ("Throughput (FPS)", results.get("throughput_fps", ""), "fps"),
            ("Model Path", results.get("model_path", ""), ""),
            ("Timestamp (UTC)", results.get("timestamp", ""), ""),
        ]
        for r in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(r[0])))
            self.table.setItem(row, 1, QTableWidgetItem(str(r[1])))
            self.table.setItem(row, 2, QTableWidgetItem(str(r[2])))

        # plot per-iteration timings if available
        timings = results.get("timings_ms", [])
        if timings:
            self.plot.plot(list(range(len(timings))), timings, pen=pg.mkPen('#00cc88', width=2), name="latency")
        else:
            avg = results.get("avg_latency_ms", 0)
            self.plot.plot([0,1], [avg, avg], pen=pg.mkPen('#00cc88', width=2), name="avg_latency")

        self.export_btn.setEnabled(True)

    def on_finished(self, success, msg):
        if success:
            self.append_log("Task finished successfully.")
        else:
            self.append_log(f"Task failed: {msg}")
            QMessageBox.critical(self, "Task failed", msg)
        self.run_btn.setEnabled(True)

    def on_export(self):
        if not self.latest_results:
            QMessageBox.information(self, "No results", "No results to export.")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Export results", f"edgebridge_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "JSON files (*.json);;CSV files (*.csv)")
        if not fp:
            return
        if fp.endswith(".json"):
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(self.latest_results, f, indent=2)
            QMessageBox.information(self, "Exported", f"Results exported to {fp}")
        elif fp.endswith(".csv"):
            with open(fp, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for k, v in self.latest_results.items():
                    writer.writerow([k, v])
            QMessageBox.information(self, "Exported", f"Results exported to {fp}")
        else:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(self.latest_results, f, indent=2)
            QMessageBox.information(self, "Exported", f"Results exported to {fp}")

def main():
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    window = EdgeBridgeGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
