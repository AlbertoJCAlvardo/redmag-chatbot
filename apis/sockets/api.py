import os
import httpx
import json

from core.bases.apis import WebSocketApi

class ChatSocketApi(WebSocketApi):
    async def stream_gemini_sse(self, client: httpx.AsyncClient, url: str, params: dict, payload: dict):
        async with client.stream("POST", url, params=params, json=payload, timeout=None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    break

                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                text = ""
                candidates = obj.get("candidates") or []
                if candidates:
                    parts = (candidates[0].get("content") or {}).get("parts") or []
                    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
                if text:
                    yield text

    async def consultar(self, user_message, group_id):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            await self.manager.broadcast_to_group("Error: falta GEMINI_API_KEY en el entorno.", group_id)
            await self.manager.broadcast_to_group("-done-", group_id)
            return

        endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:streamGenerateContent"

        instrucciones = "Eres un chatbot"
        payload = {
            "systemInstruction": {
                "parts": [{"text": instrucciones}]
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": f"Prompt: {user_message}"}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048
            }
        }

        params = {"alt": "sse", "key": api_key}

        async with httpx.AsyncClient() as client:
            try:
                async for chunk in self.stream_gemini_sse(client, endpoint, params, payload):
                    await self.manager.broadcast_to_group(chunk, group_id)

            except httpx.HTTPStatusError as e:
                msg = f"Error HTTP {e.response.status_code}: {e.response.text}"
                await self.manager.broadcast_to_group(msg, group_id)
            except httpx.RequestError as e:
                msg = f"Error de red: {e}"
                await self.manager.broadcast_to_group(msg, group_id)
            finally:
                await self.manager.broadcast_to_group("-done-", group_id)

    async def on_receive(self, data: str):
        chat_id = self.data.get('chat_id')
        await self.consultar(data, chat_id)

    async def on_connect(self):
        pass

    async def on_disconnect(self):
        pass
