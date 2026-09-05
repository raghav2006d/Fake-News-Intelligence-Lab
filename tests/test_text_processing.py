from src.text_processing import article_stats, clean_text


def test_clean_text_removes_urls_and_symbols():
    text = "Breaking!!! Visit https://example.com #News @user 123"
    assert clean_text(text) == "breaking visit"


def test_article_stats_counts_words():
    stats = article_stats("This is a test. This is only a test.")
    assert stats["words"] == 9
    assert stats["sentences"] == 2
