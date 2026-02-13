"""Thread-safe request queue for serialized LLM calls."""
from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
import uuid
from typing import Callable, Iterator, Optional


@dataclass
class RequestHandle:
    """Handle for tracking queued request progress and results."""

    request_id: str
    result_queue: queue.Queue[Optional[str]]
    done_event: threading.Event
    error_queue: queue.Queue[str]
    created_at: float


@dataclass
class _Task:
    request_id: str
    generator_factory: Callable[[], Iterator[str]]
    result_queue: queue.Queue[Optional[str]]
    done_event: threading.Event
    error_queue: queue.Queue[str]
    timeout_seconds: int
    created_at: float


class QueueManager:
    """Serialize requests so only one LLM call runs at a time."""

    def __init__(self) -> None:
        """Initialize the request queue and worker thread."""

        self._queue: queue.Queue[_Task] = queue.Queue()
        self._current_request_id: Optional[str] = None
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def enqueue(self, generator_factory: Callable[[], Iterator[str]], timeout_seconds: int) -> RequestHandle:
        """Add a streaming generator task to the queue."""

        request_id = str(uuid.uuid4())
        result_queue: queue.Queue[Optional[str]] = queue.Queue()
        done_event = threading.Event()
        error_queue: queue.Queue[str] = queue.Queue()
        created_at = time.time()

        task = _Task(
            request_id=request_id,
            generator_factory=generator_factory,
            result_queue=result_queue,
            done_event=done_event,
            error_queue=error_queue,
            timeout_seconds=timeout_seconds,
            created_at=created_at,
        )

        self._queue.put(task)

        return RequestHandle(
            request_id=request_id,
            result_queue=result_queue,
            done_event=done_event,
            error_queue=error_queue,
            created_at=created_at,
        )

    def get_position(self, request_id: str) -> int:
        """Return the position of a request in the queue. 0 means in-progress."""

        with self._lock:
            if self._current_request_id == request_id:
                return 0

        with self._queue.mutex:
            for index, task in enumerate(list(self._queue.queue)):
                if task.request_id == request_id:
                    return index + 1

        return -1

    def _worker_loop(self) -> None:
        """Continuously process queued tasks one at a time."""

        while True:
            task = self._queue.get()
            with self._lock:
                self._current_request_id = task.request_id

            start_time = time.perf_counter()
            try:
                for token in task.generator_factory():
                    elapsed = time.perf_counter() - start_time
                    if elapsed > task.timeout_seconds:
                        raise TimeoutError("Request timed out")
                    task.result_queue.put(token)
            except Exception as exc:  # noqa: BLE001
                task.error_queue.put(str(exc))
            finally:
                task.result_queue.put(None)
                task.done_event.set()
                with self._lock:
                    self._current_request_id = None
                self._queue.task_done()
