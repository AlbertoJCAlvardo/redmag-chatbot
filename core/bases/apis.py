# FastApi
from fastapi import WebSocket, WebSocketDisconnect

# User
from core.websockets.manager import ConnectionManager


class WebSocketApi:
    def __init__(self, websocket: WebSocket, manager: ConnectionManager, **kwargs):
        self.websocket = websocket
        self.manager = manager
        self.data = kwargs

    async def on_connect(self):
        pass

    async def on_receive(self, data: any):
        pass

    async def on_disconnect(self):
        pass

    async def handle_connection(self):
        chat_id = self.data.get('chat_id', 'default')

        await self.manager.connect(self.websocket, chat_id)
        await self.on_connect()
        try:
            while True:
                data = await self.websocket.receive_text()
                await self.on_receive(data)
        except WebSocketDisconnect:
            pass
        finally:
            self.manager.disconnect(self.websocket, chat_id)
            await self.on_disconnect()

