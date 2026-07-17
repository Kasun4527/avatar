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
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


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
    print(f"[TTS] samples={len(arr)} duration={len(arr)/24000:.1f}s")
    return arr


async def run(bus, clocks, audio_int16=None, topic=""):
    bus.ready()  # must be first — before any await
    await asyncio.sleep(0.5)

    if audio_int16 is not None and len(audio_int16) > 0:
        audio_bytes = audio_int16.tobytes()
        segment_id = str(TypeID("seg"))
        buffer = SpeechBuffer.from_bytes(audio_bytes, 24000)
        print(f"[TeacherEngine] publishing audio: samples={len(audio_int16)} duration={len(audio_int16)/24000:.1f}s")
        try:
            await bus.publish(SegmentGenerationStarted(segment_id=segment_id))
            await asyncio.sleep(0.1)
            await bus.publish(SegmentChunkGenerated(segment_id=segment_id, buffer=buffer))
            await asyncio.sleep(0.1)
            await bus.publish(SegmentGenerationCompleted(segment_id=segment_id))
        except RuntimeError as e:
            if "timed out" in str(e):
                print("[TeacherEngine] bus timed out — peer disconnected, exiting cleanly")
                return
            raise
    else:
        print(f"[TeacherEngine] no audio to publish")

    while True:
        await asyncio.sleep(10)