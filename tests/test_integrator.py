from __future__ import annotations

import time
import unittest

from qwen3_asr_stream.integrator import COMMIT_SETTLE_SEC, SttPublisher


class FakeHub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, payload: dict) -> None:
        self.events.append(payload)


class SettledCommitTests(unittest.TestCase):
    def test_refined_last_replaces_initial_draft(self):
        hub = FakeHub()
        publisher = SttPublisher(hub)
        draft = {"type": "commit", "text": "hello wor", "utterance_id": 3}
        final = {"type": "commit", "text": "hello world", "utterance_id": 3}

        publisher._queue_commit(draft, refining=False)
        publisher._queue_commit(draft, refining=True)
        publisher._queue_commit(final, refining=False)
        time.sleep(COMMIT_SETTLE_SEC * 1.5)

        self.assertEqual([item["text"] for item in hub.events], ["hello world"])

    def test_non_refined_last_flushes_after_settle_window(self):
        hub = FakeHub()
        publisher = SttPublisher(hub)
        publisher._queue_commit(
            {"type": "commit", "text": "short final", "utterance_id": 4},
            refining=False,
        )
        self.assertEqual(hub.events, [])
        time.sleep(COMMIT_SETTLE_SEC * 1.5)
        self.assertEqual([item["text"] for item in hub.events], ["short final"])


if __name__ == "__main__":
    unittest.main()
