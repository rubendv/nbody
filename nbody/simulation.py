from autobahn.asyncio.websocket import WebSocketServerProtocol, WebSocketServerFactory
import logging
import json
import asyncio
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBodyServerProtocol(WebSocketServerProtocol):
    def onOpen(self):
        self.factory.register(self)

    def connectionLost(self, reason):
        super(NBodyServerProtocol, self).connectionLost(self, reason)
        self.factory.unregister(self)


N = 200
X = 0
Y = 1
DXDT = 2
DYDT = 3
M = 4
ID = 5
DT = 0.1

class NBodyServerFactory(WebSocketServerFactory):


    def __init__(self, eventloop, *args, **kwargs):
        super(NBodyServerFactory, self).__init__(*args, **kwargs)
        self.clients = set()
        self.eventloop = eventloop
        self.bodies = np.array([
            np.random.normal(0.0, 1.0, N), # x
            np.random.normal(0.0, 1.0, N), # y
            np.random.normal(0.0, 0.1, N), # dx/dt
            np.random.normal(0.0, 0.1, N), # dy/dt
            10**np.random.normal(0.0, 1.0, N), # m
            np.arange(N) # id
        ])
        logger.info('Starting simulation')
        self.tick()

    def tick(self):
        message = {'time': time.time(), 'bodies': self.bodies.tolist()}
        logger.debug('Sending message: {}'.format(message))
        self.bodies[[X, Y]] += self.bodies[[DXDT, DYDT]] * DT

        self.broadcast(json.dumps(message))
        self.eventloop.call_later(DT, self.tick)

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

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        server.close()
        loop.close()



