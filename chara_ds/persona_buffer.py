"""Thread-safe, append-only buffer of persona/situation lines.

Used so dialogue workers in :mod:`chara_ds.runner` can block on a persona
line that has not been generated yet, while a background producer
(``situation_producer``) keeps growing the buffer using DeepSeek flash.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from .config import PersonaLine


class PersonaBuffer:
    """Append-only list of :class:`PersonaLine` with blocking reads."""

    def __init__(self, initial: Optional[List[PersonaLine]] = None) -> None:
        self._items: List[PersonaLine] = list(initial or [])
        self._cond = threading.Condition()
        self._finished = False

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)

    def snapshot(self) -> List[PersonaLine]:
        with self._cond:
            return list(self._items)

    def extend(self, new_items: List[PersonaLine]) -> None:
        if not new_items:
            return
        with self._cond:
            self._items.extend(new_items)
            self._cond.notify_all()

    def mark_finished(self) -> None:
        """Signal that no more items will ever be added."""
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def is_finished(self) -> bool:
        with self._cond:
            return self._finished

    def wait_for_index(
        self, index: int, timeout: Optional[float] = None
    ) -> Optional[PersonaLine]:
        """Block until the buffer has at least ``index + 1`` entries.

        Returns the entry at ``index``. If the producer signals completion
        before the buffer ever reaches that size, returns ``None``.
        """
        with self._cond:
            while index >= len(self._items):
                if self._finished:
                    return None
                self._cond.wait(timeout=timeout)
                if (
                    timeout is not None
                    and index >= len(self._items)
                    and not self._finished
                ):
                    return None
            return self._items[index]

    def wait_until_at_least(self, count: int) -> int:
        """Block until ``len(self) >= count`` or the producer finishes."""
        with self._cond:
            while len(self._items) < count and not self._finished:
                self._cond.wait()
            return len(self._items)
