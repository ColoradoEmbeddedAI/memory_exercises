"""
models.py -- the teacher and the student, and paths shared by every script.

    Teacher  LeNet300100()      784 -> 300 -> 100 -> 10, ReLU, dropout 0.2
                                after each hidden layer (training only)
    Student  StudentMLP(30)     784 -> 30 -> 10, ReLU

LeNet-300-100 is the fully connected MNIST network from Han et al.'s
pruning paper. The dropout is there so the teacher's soft targets carry
information: a teacher that has memorized the training set puts ~1.0 on the
right class of every training image, and then its targets are almost the
same as the labels (see python/part0_warmup.py).

Parameter counts (Part 0 checks them):
    teacher 784*300+300 + 300*100+100 + 100*10+10 = 266,610
    student 784*30+30 + 30*10+10                  =  23,860  (11.2x smaller)
"""

import os

import torch
import torch.nn as nn

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(LAB_DIR, "checkpoints")
RESULTS_DIR = os.path.join(LAB_DIR, "results")
PTE_DIR = os.path.join(LAB_DIR, "pte")
GENERATED_DIR = os.path.join(LAB_DIR, "generated")
TEACHER_CKPT = os.path.join(CHECKPOINT_DIR, "teacher_lenet300100.pt")


class LeNet300100(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.drop(torch.relu(self.fc2(x)))
        return self.fc3(x)


class StudentMLP(nn.Module):
    def __init__(self, hidden: int = 30):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_teacher() -> LeNet300100:
    """The provided teacher, in eval mode (dropout off)."""
    t = LeNet300100()
    t.load_state_dict(torch.load(TEACHER_CKPT, map_location="cpu"))
    return t.eval()
