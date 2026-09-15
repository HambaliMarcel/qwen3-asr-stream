from __future__ import annotations

import unittest

from qwen3_asr_stream.parse import collapse_loops


class LoopGuardTests(unittest.TestCase):
    def test_two_word_spiral_is_cut(self):
        loop = "Yeah, I've seen it. " + " ".join(["black on"] * 50) + " black"
        text, looped = collapse_loops(loop)
        self.assertTrue(looped)
        self.assertEqual(text, "Yeah, I've seen it. black on black on black")

    def test_single_word_spiral_keeps_two_copies(self):
        text, looped = collapse_loops(", ".join(["baby"] * 25) + ", yeah")
        self.assertTrue(looped)
        self.assertEqual(text, "baby, baby, yeah")

    def test_phrase_spiral_is_cut(self):
        text, looped = collapse_loops(" ".join(["the edge of"] * 30) + " the")
        self.assertTrue(looped)
        self.assertEqual(text, "the edge of the edge of the")

    def test_sung_hooks_survive(self):
        for line in (
            "no love, no love, no love, we don't need it",
            "pesawat pesawat pesawat",
            "na na na na na hey",
            "I'm up on the dance floor. Every time I get up, I'm up on the dance floor.",
            "",
        ):
            self.assertEqual(collapse_loops(line), (line, False))


if __name__ == "__main__":
    unittest.main()
