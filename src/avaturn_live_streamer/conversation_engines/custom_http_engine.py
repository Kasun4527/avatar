# src/avaturn_live_streamer/conversation_engines/custom_http_engine.py

import sys
sys.path.insert(0, '/home/sinhala_llm/Ashan/avatar/avtr-1/venv/lib/python3.12/site-packages')

import asyncio
import os

import numpy as np
from typeid import TypeID

from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.events import (
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    SetSpeechSpeed,
    Shutdown,
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
_TTS_SAMPLE_RATE = 24000


def _text_to_audio(text: str) -> np.ndarray:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
        ),
    )
    part = response.candidates[0].content.parts[0].inline_data
    pcm_bytes = part.data  # already raw bytes — NO base64 decode
    print(f"[TTS] mime={part.mime_type} raw_len={len(pcm_bytes)}")

    pcm_bytes = pcm_bytes[:len(pcm_bytes) & ~1]
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
    print(f"[TTS] samples={len(arr)} duration={len(arr)/_TTS_SAMPLE_RATE:.1f}s")
    return arr


def _time_stretch(audio_int16: np.ndarray, rate: float) -> np.ndarray:
    """Speed up/slow down audio while preserving pitch (unlike naive
    resampling, which would also shift pitch). rate > 1.0 = faster/shorter,
    rate < 1.0 = slower/longer. No-op at rate == 1.0, the common case, to
    avoid any quality loss from a redundant round-trip through the
    time-stretch algorithm."""
    if abs(rate - 1.0) < 1e-6 or audio_int16.size == 0:
        return audio_int16
    import librosa

    audio_float = audio_int16.astype(np.float32) / 32768.0
    stretched = librosa.effects.time_stretch(audio_float, rate=rate)
    return np.clip(stretched * 32768.0, -32768, 32767).astype(np.int16)


async def _speak(bus, text: str, para_index: int, rate: float) -> None:
    segment_id = str(TypeID("seg"))
    await bus.publish(
        SegmentGenerationStarted(segment_id=segment_id, metadata={"para_index": str(para_index)})
    )
    try:
        audio_int16 = await asyncio.get_event_loop().run_in_executor(None, _text_to_audio, text)
        audio_int16 = await asyncio.get_event_loop().run_in_executor(
            None, _time_stretch, audio_int16, rate
        )
        buffer = SpeechBuffer.from_bytes(audio_int16.tobytes(), _TTS_SAMPLE_RATE)
        await bus.publish(SegmentChunkGenerated(segment_id=segment_id, buffer=buffer))
    finally:
        # Always close the segment, even on TTS/time-stretch failure — the
        # scheduler's write-active-segment stays latched on a failed segment
        # otherwise, and silently rejects every paragraph that comes after it
        # as a "double segment start".
        await bus.publish(SegmentGenerationCompleted(segment_id=segment_id))


async def run(
    bus,
    clocks,
    paragraphs: list[str] | None = None,
    topic: str = "",
    initial_speed: float = 1.0,
) -> None:
    """
    Teaching engine: speaks lesson content paragraph-by-paragraph when the
    session starts, each paragraph tagged with its index so the frontend can
    highlight the matching text as it's spoken.

    paragraphs     — pre-split lesson paragraphs (builders.py does the splitting)
    topic          — lesson topic name (used only for logging here; the
                     "no content" fallback text is already resolved into
                     `paragraphs` by builders.py before this is called)
    initial_speed  — starting playback rate; can change live via SetSpeechSpeed,
                     taking effect starting at the next paragraph (mid-paragraph
                     speed changes aren't supported — a segment's audio is
                     already generated as one unit before it starts playing)
    """
    paragraphs = paragraphs or []
    current_rate = [initial_speed]  # mutable holder the rate-listener task updates live

    async def _rate_listener(sub_bus) -> None:
        async with sub_bus.subscribe(SetSpeechSpeed, Shutdown) as sub:
            sub_bus.ready()
            while True:
                event = await sub.get_next()
                if event is None or isinstance(event, Shutdown):
                    return
                print(f"[TeacherEngine] speed changed to {event.rate}x (takes effect next paragraph)")
                current_rate[0] = event.rate

    async def _speak_loop() -> None:
        await asyncio.sleep(0.5)

        if not paragraphs:
            print(f"[TeacherEngine] no content to speak for topic='{topic}'")
        else:
            print(f"[TeacherEngine] speaking {len(paragraphs)} paragraph(s) for topic='{topic}'")
            for i, para in enumerate(paragraphs):
                try:
                    await _speak(bus, para, para_index=i, rate=current_rate[0])
                except RuntimeError as e:
                    if "timed out" in str(e):
                        print("[TeacherEngine] bus timed out — peer disconnected, exiting cleanly")
                        return
                    raise
                except Exception as e:
                    print(f"[TeacherEngine] Error on paragraph {i}: {e}")
                    import traceback
                    traceback.print_exc()

        # Keep session alive after speaking
        while True:
            await asyncio.sleep(10)

    bus.ready()  # must be first — before any await; accounts for the single incoming clone
    async with asyncio.TaskGroup() as tg:
        tg.create_task(_rate_listener(bus.clone()))
        tg.create_task(_speak_loop())
