"""
Unit tests for src/features/preprocessing.py
"""
from src.features.preprocessing import (
    remove_html,
    remove_latex,
    remove_special_characters,
    remove_stopwords_fn,
    clean_text,
    truncate_text,
)


class TestRemoveHtml:

    def test_strips_basic_tags(self):
        result = remove_html("<p>Hello <b>world</b></p>")
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_handles_plain_text(self):
        result = remove_html("No HTML here")
        assert result == "No HTML here"

    def test_handles_empty_string(self):
        result = remove_html("")
        assert result == ""


class TestRemoveLatex:

    def test_removes_inline_math(self):
        result = remove_latex("The value is $x^2 + y^2 = z^2$ in geometry.")
        assert "$" not in result
        assert "geometry" in result

    def test_removes_latex_commands(self):
        result = remove_latex("See \\cite{author2020} for details.")
        assert "\\cite" not in result
        assert "details" in result

    def test_removes_textbf(self):
        result = remove_latex("We use \\textbf{bold} text here.")
        assert "\\textbf" not in result


class TestRemoveSpecialCharacters:

    def test_removes_punctuation(self):
        result = remove_special_characters("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_preserves_letters_and_numbers(self):
        result = remove_special_characters("abc 123 xyz")
        assert "abc" in result
        assert "123" in result

    def test_collapses_whitespace(self):
        result = remove_special_characters("too   many    spaces")
        assert "  " not in result


class TestRemoveStopwords:

    def test_removes_common_stopwords(self):
        result = remove_stopwords_fn("the quick brown fox jumps over the lazy dog")
        assert "the" not in result.split()
        assert "over" not in result.split()
        assert "fox" in result

    def test_preserves_meaningful_words(self):
        result = remove_stopwords_fn("machine learning neural network")
        assert "machine" in result
        assert "learning" in result
        assert "neural" in result


class TestCleanText:

    def test_full_pipeline_on_html_text(self):
        text = "<p>The <b>neural network</b> learns from data.</p>"
        result = clean_text(text, remove_stopwords=True, lemmatize=False)
        assert "<p>" not in result
        assert "neural" in result
        assert "network" in result

    def test_full_pipeline_on_latex_text(self):
        text = "We propose $\\alpha$-divergence with \\cite{ref} for optimization."
        result = clean_text(text, remove_stopwords=True, lemmatize=False)
        assert "$" not in result
        assert "\\cite" not in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_none_like_input_returns_empty(self):
        assert clean_text("   ") == ""

    def test_truncation(self):
        long_text = "word " * 500
        result = clean_text(long_text, lemmatize=False, max_len=100)
        assert len(result) <= 100


class TestTruncateText:

    def test_short_text_unchanged(self):
        text = "short text"
        assert truncate_text(text, 100) == text

    def test_long_text_truncated(self):
        text = "word " * 100
        result = truncate_text(text, 50)
        assert len(result) <= 50

    def test_truncates_at_word_boundary(self):
        text = "hello world foo bar"
        result = truncate_text(text, 11)
        assert not result.endswith("wor")