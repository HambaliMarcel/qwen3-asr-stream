from __future__ import annotations

import unittest

from qwen3_asr_stream.parse import (
    classify_sound_event_text,
    continuation_language,
    has_lexical_speech,
    infer_languages,
    language_plausible,
    strip_asr_markup,
)


class LanguageEvidenceTests(unittest.TestCase):
    def test_indonesian_beats_portuguese_tag(self):
        langs = infer_languages("Tak heran aku desa", "Portuguese")
        self.assertEqual(langs, ["Indonesian"])
        self.assertFalse(language_plausible("Portuguese", "Tak heran aku desa"))

    def test_latin_text_rejects_japanese_tag(self):
        langs = infer_languages("Tahir Wadukodeska", "Japanese")
        self.assertNotIn("Japanese", langs)
        self.assertFalse(language_plausible("Japanese", "Tahir Wadukodeska"))

    def test_real_spanish_is_kept(self):
        langs = infer_languages("besame mucho", "Spanish")
        self.assertEqual(langs, ["Spanish"])

    def test_arabic_script_is_detected(self):
        langs = infer_languages("السلام عليكم", "")
        self.assertEqual(langs, ["Arabic"])

    def test_cantonese_tag_is_kept(self):
        langs = infer_languages("唔該", "Cantonese")
        self.assertEqual(langs, ["Cantonese"])
        self.assertEqual(continuation_language("Cantonese", "唔該"), "Cantonese")

    def test_japanese_tag_is_kept(self):
        langs = infer_languages("います", "Japanese")
        self.assertEqual(langs, ["Japanese"])
        self.assertEqual(continuation_language("Japanese", "います"), "Japanese")

    def test_model_tag_is_not_replaced_by_indonesian_lexicon(self):
        self.assertEqual(continuation_language("Cantonese", "唔該"), "Cantonese")
        self.assertNotEqual(continuation_language("Cantonese", "唔該"), "Indonesian")


class MarkupAndSpeechTests(unittest.TestCase):
    def test_truncated_canton_leak_is_stripped(self):
        lang, text = strip_asr_markup("language Canton 唔該")
        self.assertEqual(lang, "Cantonese")
        self.assertEqual(text, "唔該")

    def test_short_cantonese_is_speech(self):
        self.assertIsNone(classify_sound_event_text("唔該"))
        self.assertTrue(has_lexical_speech("唔該"))

    def test_cough_is_still_non_speech(self):
        self.assertEqual(classify_sound_event_text("咳咳"), "batuk?")
        self.assertFalse(has_lexical_speech("咳咳"))


if __name__ == "__main__":
    unittest.main()
