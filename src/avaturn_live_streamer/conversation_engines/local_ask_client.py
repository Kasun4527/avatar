import httpx
import traceback

from avaturn_live_streamer.events import (
    UserSpeechReceived,
    TextEchoEnqueueText,
    Shutdown,
)


class LocalAskClient:
    def __init__(self, config):
        print("🔥 LocalAskClient __init__")

        self._config = config

        print("API URL =", getattr(config, "api_url", None))

        self._client = httpx.AsyncClient(
            timeout=120
        )

    async def run(self, bus, clocks):

        print("🚀 LocalAskClient started")

        try:
            bus.ready()
            print("✅ bus.ready()")

        except Exception as e:
            print("❌ bus.ready failed:", e)

        try:

            print("📡 Waiting for events...")

            async with bus.subscribe(
                UserSpeechReceived,
                Shutdown,
            ) as sub:

                print("✅ Subscription created")

                async for event in sub:

                    print("\n====================")
                    print("📨 EVENT RECEIVED")
                    print("TYPE =", type(event))
                    print("EVENT =", event)
                    print("====================\n")

                    if isinstance(event, UserSpeechReceived):

                        print("🎤 UserSpeechReceived detected")

                        text = getattr(event, "text", "")

                        print("TEXT =", repr(text))

                        if not text:
                            print("⚠ Empty speech text")
                            continue

                        try:

                            print("➡ Sending to DeepSeek...")
                            answer = await self.ask(text)

                            print("⬅ DeepSeek replied:")
                            print(answer)

                            print("📤 Publishing response...")

                            await bus.publish(
                                TextEchoEnqueueText(
                                    phrase_id="deepseek",
                                    text=answer,
                                )
                            )

                            print("✅ Published")

                        except Exception as e:

                            print("❌ LLM ERROR")
                            print(e)
                            traceback.print_exc()

                    elif isinstance(event, Shutdown):

                        print("🛑 Shutdown event received")
                        return

        except Exception as e:

            print("❌ RUN LOOP ERROR")
            print(e)
            traceback.print_exc()

    async def ask(self, text: str):

        print("🌐 POST", self._config.api_url)
        print("PROMPT =", text)

        response = await self._client.post(
            self._config.api_url,
            json={
                "instruction": text
            },
        )

        print("HTTP STATUS =", response.status_code)

        response.raise_for_status()

        data = response.json()

        print("RAW RESPONSE =", data)

        answer = data.get("answer", "")

        print("FINAL ANSWER =", answer)

        return answer