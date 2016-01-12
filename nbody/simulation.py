from autobahn.asyncio.websocket import WebSocketServerProtocol, WebSocketServerFactory
import logging
import json
import random
import asyncio
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBodyServerProtocol(WebSocketServerProtocol):
    def onOpen(self):
        self.factory.register(self)

    def connectionLost(self, reason):
        super(NBodyServerProtocol, self).connectionLost(self, reason)
        self.factory.unregister(self)


class NBodyServerFactory(WebSocketServerFactory):
    def __init__(self, eventloop, *args, **kwargs):
        super(NBodyServerFactory, self).__init__(*args, **kwargs)
        self.clients = set()
        self.eventloop = eventloop
        self.tick()

    def tick(self):
        bodies = [
            {'x': random.gauss(0.0, 1.0), 'y': random.gauss(0.0, 1.0), 'mass': 10**random.gauss(0.1, 0.5)}
            for _ in range(200)
        ]
        message = {'time': time.time(), 'bodies': bodies}
        logger.info('Sending tick: {}'.format(message))
        self.broadcast(json.dumps(message))
        self.eventloop.call_later(1, self.tick)

    def register(self, client):
        self.clients.add(client)

    def unregister(self, client):
        self.clients.remove(client)

    def broadcast(self, message):
        for client in self.clients:
            client.sendMessage(message.encode('utf-8'))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    factory = NBodyServerFactory(
        loop,
        'ws://localhost:9000',
        debug=True,
        debugCodePaths=False
    )
    factory.protocol = NBodyServerProtocol

    coro = loop.create_server(factory, '127.0.0.1', 9000)
    server = loop.run_until_complete(coro)

    loop.run_forever()


