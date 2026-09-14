from __future__ import annotations

import unittest

from qwen3_asr_stream.integrator import SttPublisher


class FakeHub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, payload: dict) -> None:
        self.events.append(payload)


class ImmediateCommitTests(unittest.TestCase):
    def test_first_last_publishes_without_waiting(self):
        hub = FakeHub()
        publisher = SttPublisher(hub)
        publisher._queue_commit(
            {"type": "commit", "text": "short final", "utterance_id": 4},
            refining=False,
        )
        self.assertEqual([item["text"] for item in hub.events], ["short final"])

    def test_same_line_refine_does_not_duplicate(self):
        hub = FakeHub()
        publisher = SttPublisher(hub)
        publisher._queue_commit(
            {"type": "commit", "text": "hello world", "utterance_id": 3},
            refining=False,
        )
        publisher._queue_commit(
            {"type": "commit", "text": "hello world", "utterance_id": 3},
            refining=True,
        )
        publisher._queue_commit(
            {"type": "commit", "text": "hello world", "utterance_id": 3},
            refining=False,
        )
        self.assertEqual([item["text"] for item in hub.events], ["hello world"])

    def test_sound_while_speaking_is_not_a_commit(self):
        hub = FakeHub()
        publisher = SttPublisher(hub)

        class State:
            finalized = ""
            unfixed = "[crowing]"
            text = "[crowing]"
            language = ""
            speaking = True
            decoding = False
            utterance_id = 2
            gap_sec = 0.0
            silence_sec = 0.0
            event_label = "crowing"
            event_score = 0.9
            event_top = ("crowing",)
            event_is_companion = True
            non_speech_only = False
            sound_label = "crowing"
            refining = False

        publisher.on_update(State())
        types = [item["type"] for item in hub.events]
        self.assertIn("sound", types)
        self.assertNotIn("commit", types)

    def test_sound_without_speech_is_not_a_commit(self):
        hub = FakeHub()
        publisher = SttPublisher(hub)

        class State:
            finalized = ""
            unfixed = "[typing]"
            text = "[typing]"
            language = ""
            speaking = False
            decoding = False
            utterance_id = 3
            gap_sec = 0.4
            silence_sec = 0.4
            event_label = "typing"
            event_score = 0.8
            event_top = ("typing",)
            event_is_companion = True
            non_speech_only = True
            sound_label = "typing"
            refining = False

        publisher.on_update(State())
        types = [item["type"] for item in hub.events]
        self.assertIn("sound", types)
        self.assertNotIn("commit", types)


if __name__ == "__main__":
    unittest.main()
