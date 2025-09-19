"""
Optimizer: quantization, pruning, distillation helpers.
This file exposes an Optimizer class that works with PyTorch models.
"""

import copy

def _ensure_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise ImportError("torch is required for optimizer functions") from e
    return torch, nn

class Optimizer:
    def __init__(self, model, sample_input=None):
        self.model = model
        self.sample_input = sample_input

    def quantize(self, mode="dynamic"):
        torch, nn = _ensure_torch()
        if mode == "dynamic":
            q_model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("✅ Applied dynamic quantization")
            return q_model
        elif mode == "static":
            # static quantization requires calibration and qconfig
            self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(self.model, inplace=True)
            if self.sample_input is not None:
                with torch.no_grad():
                    self.model(self.sample_input)
            torch.quantization.convert(self.model, inplace=True)
            print("✅ Applied static quantization")
            return self.model
        else:
            raise ValueError("mode must be 'dynamic' or 'static'")

    def prune(self, amount=0.3):
        torch, nn = _ensure_torch()
        import torch.nn.utils.prune as prune
        pruned = copy.deepcopy(self.model)
        parameters_to_prune = []
        for name, module in pruned.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        # Remove reparameterization to make pruning permanent
        for module, _ in parameters_to_prune:
            try:
                prune.remove(module, 'weight')
            except Exception:
                pass
        print(f"✅ Applied pruning (amount={amount})")
        return pruned

    def distill(self, teacher, student, data_loader, epochs=1, temperature=2.0, alpha=0.5, device="cpu"):
        import torch
        import torch.nn.functional as F
        student = copy.deepcopy(student)
        teacher = copy.deepcopy(teacher)
        student.to(device)
        teacher.to(device)
        teacher.eval()
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            running = 0.0
            for i, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    t_out = teacher(x)
                s_out = student(x)
                # soft loss
                soft_loss = F.kl_div(
                    F.log_softmax(s_out / temperature, dim=1),
                    F.softmax(t_out / temperature, dim=1),
                    reduction="batchmean"
                ) * (temperature ** 2)
                hard_loss = criterion(s_out, y)
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                loss.backward()
                optimizer.step()
                running += float(loss.item())
            avg = running / (i + 1)
            print(f"Epoch {epoch+1}/{epochs} - distill loss: {avg:.4f}")
        print("✅ Distillation completed")
        return student
