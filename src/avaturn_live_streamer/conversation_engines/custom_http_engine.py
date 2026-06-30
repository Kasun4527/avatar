# custom_http_engine.py
import asyncio
import httpx
import numpy as np
import io
from gtts import gTTS
from pydub import AudioSegment
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from typeid import TypeID

CUSTOM_ENDPOINT = "https://cupbearer-pointing-serotonin.ngrok-free.dev"

def _text_to_audio(text: str) -> np.ndarray:
    tts = gTTS(text, lang="si")
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)
    audio_seg = AudioSegment.from_mp3(mp3_buf)
    audio_seg = audio_seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    return np.frombuffer(audio_seg.raw_data, dtype=np.int16)

async def _speak(bus, text):
    segment_id = str(TypeID("seg"))
    await bus.publish(SegmentGenerationStarted(segment_id=segment_id))
    audio_int16 = await asyncio.get_event_loop().run_in_executor(None, _text_to_audio, text)
    buffer = SpeechBuffer.from_bytes(audio_int16.tobytes(), 24000)
    await bus.publish(SegmentChunkGenerated(segment_id=segment_id, buffer=buffer))
    await bus.publish(SegmentGenerationCompleted(segment_id=segment_id))

async def run(bus: EventBus, clocks: StreamClocks) -> None:
    bus.ready()
    await asyncio.sleep(2)
    try:
        print("Avatar speaking Sinhala greeting...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                CUSTOM_ENDPOINT,
                json={
                    "instruction": "ඔබ සිංහල AI සහායකයෙකි. සිංහල භාෂාවෙන් පමණක් පිළිතුරු දෙන්න. ඉංග්‍රීසි භාවිතා නොකරන්න. මෙය ඔබේ හැඳින්වීමයි - ඔබේ නම, ඔබට කළ හැකි දේ, සහ ඔබ කෙසේ උදව් කළ හැකිද කියා විස්තර කරන්න.",
                    "max_new_tokens": 150
                }
            )
            reply = r.json().get("answer", "ආයුබෝවන්! මම ඔබේ AI සහායකයා. ඔබට ඕනෑම ප්‍රශ්නයකට පිළිතුරු දීමට, තොරතුරු සෙවීමට, සහ ඔබේ කටයුතු සඳහා උදව් කිරීමට මට හැකියාව ඇත. ඔබට කෙසේ උදව් කළ හැකිද?").strip()
        print(f"Avatar: {reply}")
        await _speak(bus, reply)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    while True:
        await asyncio.sleep(10)
