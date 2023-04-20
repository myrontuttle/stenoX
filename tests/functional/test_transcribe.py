from pathlib import Path

from stenoX import transcribe


def test_transcribe():
    media_file = "../media/test_file.wav"
    transcribe.transcribe(media_file)
    transcribed = "../media/test_file.txt"
    assert Path(transcribed).is_file()


def test_transcribe_from_youtube():
    media_url = "https://www.youtube.com/watch?v=CpUO3JARjAc&ab_channel=CNBC"
    transcribe.transcribe(media_url)
    transcribed = (
        "../media/Why US Vacation Policies Are So Much Worse Than "
        "Europe’s.txt"
    )
    assert Path(transcribed).is_file()


def test_transcribe_with_grammar():
    media_file = "../media/CommonGrammaticalMistakes.mp4"
    transcribe.transcribe(media_file)
    transcribed = "../media/CommonGrammaticalMistakes.txt"
    assert Path(transcribed).is_file()


def test_summarize_long():
    """Test a long summary"""
    test_file = (
        "../media/LangChain Agents Deep Dive with GPT 35 — " "LangChain 6.txt"
    )
    with open(test_file, "r") as tf:
        transcript = tf.read()
    summaries = transcribe.summarize(transcript)
    assert len(summaries) != 0
