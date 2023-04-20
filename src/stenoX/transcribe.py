from typing import Any, Dict, List, Optional

import datetime
import logging
import math
import os
import subprocess
import time
from pathlib import Path

import click
import openai
import tiktoken
import whisper
from backoff import expo, on_exception
from openai import APIError
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
from ratelimit import RateLimitException, limits
from reretry import retry
from whisperx import align, load_align_model
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
WHISPER_MODEL = "small.en"
WHISPER_DEVICE = "cuda"  # "cpu" or "cuda" if available
LANGUAGE = "en"
STENO_ROLE = {
    "role": "system",
    "content": "You are a helpful meeting assistant.",
}
MINUTE = 60
CHAT_MODEL = "gpt-3.5-turbo"
CH_MAX_TOKENS = 4096
SUMMARY_PROMPT = (
    "Summarize the following conversation for it's "
    "participants. Make sure to point out any follow-ups "
    "required or questions that need to be answered. If there "
    "are any acronyms, provide a glossary with their definitions.\n"
)
OPENAI_KEY = "OPENAI_API_KEY"
HF_TOKEN = "HF_TOKEN"


def get_key_from_env(key: str) -> Optional[str]:
    """
    Get a key from the environment variables.

    Args:
        key: HF_TOKEN (HuggingFace Token) or OPENAI_KEY (OpenAI API key)

    Returns:
        The value of the key if it exists, else None.
    """
    if key not in os.environ:
        logger.critical(
            f"{key} does not exist as environment variable.",
        )
        logger.debug(os.environ)
    return os.getenv(key)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("audio_loc")
def main(audio_loc: str) -> None:
    """Transcribe an audio recording"""
    transcribe(audio_loc)


def transcribe(audio_loc: str) -> None:
    """Transcribe an audio recording"""
    # Check if URL or Local File
    if audio_loc.startswith("http"):
        # Download file from YouTube
        logger.info("Downloading file from YouTube")
        file_loc = download_youtube(audio_loc)
        file_path = Path(file_loc)
    else:
        file_loc = audio_loc
        file_path = Path(audio_loc)

    if not file_path.exists():
        logger.error(f"{file_loc}: No such file")
        logger.error(f"CWD: {os.getcwd()}")
        return None
    # Convert to WAV
    if file_path.suffix != ".wav":
        convert_to_wav(file_loc)
        file_loc = os.path.splitext(file_loc)[0] + ".wav"
    # Transcribe and Diarize
    transcript = transcribe_file(file_loc)
    aligned_segments = align_segments(transcript, file_loc)
    diarization_result = diarize(file_loc)
    results_segments_w_speakers = assign_speakers(
        diarization_result, aligned_segments
    )
    transcript_text = ""
    for seg in results_segments_w_speakers:
        start = str(datetime.timedelta(seconds=round(seg["start"])))
        end = str(datetime.timedelta(seconds=round(seg["end"])))
        transcript_text += (
            f"[{start}-{end}] " f"{seg['speaker']}:{seg['text']}\n\n"
        )
    # Summarize
    summaries = summarize(transcript_text)
    # Write to file
    transcribed = write(file_loc, summaries, transcript_text)
    logger.info(f"Transcription at: {transcribed}")


def download_youtube(url: str) -> str:
    """
    Downloads a YouTube video from the given URL and saves it to media
    directory.

    Args:
        url: The URL of the YouTube video to download.

    Returns:
        File path of the downloaded video.
    """
    try:
        yt = YouTube(url)
    except VideoUnavailable:
        logger.error(f"Video {url} is unavailable.")
    else:
        video_stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
        # Get path to media folder
        media_path = Path(__file__).parents[2] / "media"
        audio_file = video_stream.download(media_path)
        logger.info(f"Downloaded {audio_file}")
        return audio_file


def convert_to_wav(input_file: str) -> None:
    """
    Converts an audio file to WAV format using FFmpeg. The output file will
    be created by replacing the input file extension with ".wav".

    Args:
        input_file: The path of the input audio file to convert.

    Returns:
        None
    """
    output_file = os.path.splitext(input_file)[0] + ".wav"

    command = (
        f'ffmpeg -i "{input_file}" -vn -acodec pcm_s16le -ar 44100'
        f' -ac 1 "{output_file}"'
    )

    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(
            f'Successfully converted "{input_file}" to "{output_file}"'
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            f'Error: {e}, could not convert "{input_file}" to '
            f'"{output_file}"'
        )


def transcribe_file(audio_file: str) -> Dict[str, Any]:
    """
    Transcribe an audio file using a speech-to-text model.

    Args:
        audio_file: Path to the audio file to transcribe.

    Returns:
        A dictionary representing the transcript, including the segments,
        the language code, and the duration of the audio file.
    """
    # transcribe with original whisper
    logger.info("Loading model: " + WHISPER_MODEL)
    model = whisper.load_model(WHISPER_MODEL, WHISPER_DEVICE)
    whisper.DecodingOptions(language=LANGUAGE)
    # Transcribe
    logger.info("Transcribing")
    return model.transcribe(audio_file)


def align_segments(
    transcript: Dict[str, Any],
    audio_file: str,
) -> Dict[str, Any]:
    """
    Align the transcript segments using a pretrained alignment model.
    Args:
        transcript: Dictionary representing the transcript with segments
        audio_file: Path to the audio file containing the audio data.
    Returns:
        A dictionary representing the aligned transcript segments.
    """
    logger.info("Loading alignment model")
    model_a, metadata = load_align_model(
        language_code=LANGUAGE, device=WHISPER_DEVICE
    )
    logger.info("Aligning output")
    result_aligned = align(
        transcript["segments"], model_a, metadata, audio_file, WHISPER_DEVICE
    )
    return result_aligned


def diarize(audio_file: str) -> Dict[str, Any]:
    """
    Perform speaker diarization on an audio file.
    Args:
        audio_file: Path to the audio file to diarize.
    Returns:
        A dictionary representing the diarized audio file,
        including the speaker embeddings and the number of speakers.
    """
    logger.info("Diarizing")
    diarization_pipeline = DiarizationPipeline(
        use_auth_token=get_key_from_env(HF_TOKEN)
    )
    diarization_result = diarization_pipeline(audio_file)
    return diarization_result


def assign_speakers(
    diarization_result: Dict[str, Any], aligned_segments: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Assign speakers to each transcript segment based on the speaker
    diarization result.
    Args:
        diarization_result: Dictionary representing the diarized audio file,
            including the speaker embeddings and the number of speakers.
        aligned_segments: Dictionary representing the aligned transcript
            segments.
    Returns:
        A list of dictionaries representing each segment of the transcript,
        including the start and end times, the spoken text, and the speaker ID.
    """
    logger.info("Assigning speakers to segments")
    result_segments, word_seg = assign_word_speakers(
        diarization_result, aligned_segments["segments"]
    )
    results_segments_w_speakers: List[Dict[str, Any]] = []
    for result_segment in result_segments:
        if (
            len(results_segments_w_speakers) > 0
            and result_segment["speaker"]
            == results_segments_w_speakers[-1]["speaker"]
        ):
            results_segments_w_speakers[-1]["text"] += result_segment["text"]
            results_segments_w_speakers[-1]["end"] = result_segment["end"]
            continue
        results_segments_w_speakers.append(
            {
                "start": result_segment["start"],
                "end": result_segment["end"],
                "speaker": result_segment["speaker"],
                "text": result_segment["text"],
            }
        )
    return results_segments_w_speakers


def summarize(transcript: str) -> List[str]:
    """
    Summarize the transcript using a pretrained summarization model.

    Args:
        transcript: Text of the transcript to summarize.

    Returns:
        A list of strings representing the summary. The list will contain
        multiple summaries if the transcript is too long to fit in a single
        summary.
    """
    logger.info("Summarizing")
    messages = [
        STENO_ROLE,
        {
            "role": "user",
            "content": SUMMARY_PROMPT + "\n" + transcript,
        },
    ]
    token_count = num_tokens_from_messages(messages)
    summaries = []
    if token_count < CH_MAX_TOKENS:
        summaries.append(llm(messages))
        return summaries
    else:
        # Try to split transcript on sentences into roughly equal segments
        # that fit below max tokens
        segments = []
        num_lines = transcript.count("\n")
        segments_needed = math.ceil(token_count / CH_MAX_TOKENS) + 1
        lines_per_segment = int(num_lines / segments_needed)

        logger.info(
            f"Transcript requires {token_count} tokens. Limit is "
            f"{CH_MAX_TOKENS}. Breaking into {segments_needed} segments."
        )
        start_loc = 0
        for n in range(segments_needed):
            current_loc = transcript.find("\n", start_loc + len("\n"))
            for m in range(lines_per_segment - 1):
                current_loc = transcript.find("\n", current_loc + len("\n"))
            if current_loc == -1:
                current_loc = len(transcript)
            segments.append(transcript[start_loc:current_loc])
            start_loc = current_loc
        for idx, seg in enumerate(segments):
            if seg.strip() == "":
                continue
            messages = [
                STENO_ROLE,
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT + "\n" + seg,
                },
            ]
            segment_summary = llm(messages)
            logger.info(f"Segment {idx} summary: " + segment_summary)
            summaries.append(segment_summary)
            time.sleep(1)
        main_summary_msg = [
            STENO_ROLE,
            {
                "role": "user",
                "content": "Summarize the following: "
                + "\n"
                + "\n".join(summaries),
            },
        ]
        # Add main summary to beginning of summaries list
        summaries.insert(0, llm(main_summary_msg))
        return summaries


@retry(APIError, tries=8, delay=1, backoff=2)
@on_exception(expo, RateLimitException, max_tries=8)
@limits(calls=20, period=MINUTE)
def llm(messages: List[Dict[str, str]]) -> str:
    """Sends request to ChatGPT service and returns response."""
    openai.api_key = get_key_from_env(OPENAI_KEY)
    try:
        response = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        return str(response["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(e)
        logger.debug(f" from: {messages}")
        return ""


def num_tokens_from_messages(messages, model=CHAT_MODEL) -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == CHAT_MODEL:  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{
            # role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1
                    # token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for
            model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for
  information on how messages are converted to tokens."""
        )


def write(file_loc: str, summaries: List[str], transcript: str) -> str:
    """
    Writes the summary and transcript to txt file
    Args:
        file_loc: The location of the audio file
        summaries: The list of summaries
        transcript: The transcript

    Returns:
        The path to the file containing the summary and transcript.
    """
    # Write result
    logger.info("Writing result to transcript file")
    transcribed = os.path.splitext(file_loc)[0] + ".txt"
    with open(transcribed, "w") as file:
        file.write("Overall Summary:\n\n")
        for idx, summary in enumerate(summaries):
            file.write(f"Section {idx + 1} Summary:\n")
            file.write(summary + "\n\n")
        file.write("Transcript:\n" + transcript)
    return transcribed


if __name__ == "__main__":
    main()
