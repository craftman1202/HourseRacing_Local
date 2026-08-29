"""Cloud Tasks のインプロセス代替。

タスク名の一意制約を再現するのが要点。Cloud Tasks は同名タスクの再 enqueue を
ALREADY_EXISTS で弾くので、決定的なタスク名を付けるだけで冪等性が得られる（SC-03）。
本番へ移すときは `enqueue` の中身を Cloud Tasks API に差し替える。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .clock import to_utc


class AlreadyExists(Exception):
    """同名タスクが既にキューにある。Cloud Tasks の ALREADY_EXISTS 相当。"""


@dataclass(frozen=True)
class Task:
    name: str
    endpoint: str
    payload: dict
    scheduled_for: datetime


@dataclass
class TaskQueue:
    tasks: dict[str, Task] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)
    enqueue_attempts: int = 0

    def enqueue(self, task: Task) -> bool:
        """登録できたら True、既存なら False。例外にしないのは、
        スケジュール再照合が毎回同じタスクを積もうとするのが正常動作のため。"""
        self.enqueue_attempts += 1
        if task.name in self.tasks:
            return False
        self.tasks[task.name] = Task(task.name, task.endpoint, task.payload,
                                     to_utc(task.scheduled_for))
        return True

    def enqueue_strict(self, task: Task) -> None:
        if not self.enqueue(task):
            raise AlreadyExists(task.name)

    def delete(self, name: str) -> bool:
        if name in self.tasks:
            del self.tasks[name]
            self.deleted.append(name)
            return True
        return False

    def names(self) -> list[str]:
        return sorted(self.tasks)

    def due(self, now: datetime) -> list[Task]:
        return sorted((t for t in self.tasks.values() if t.scheduled_for <= to_utc(now)),
                      key=lambda t: t.scheduled_for)

    def count(self, endpoint: str | None = None) -> int:
        if endpoint is None:
            return len(self.tasks)
        return sum(1 for t in self.tasks.values() if t.endpoint == endpoint)
