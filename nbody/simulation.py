from autobahn.asyncio.websocket import WebSocketServerProtocol, WebSocketServerFactory
import logging
import json
import asyncio
import time
import numpy as np
import numba

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBodyServerProtocol(WebSocketServerProtocol):
    def onOpen(self):
        self.factory.register(self)

    def connectionLost(self, reason):
        super(NBodyServerProtocol, self).connectionLost(self, reason)
        self.factory.unregister(self)


N = 500
X = 0
Y = 1
DXDT = 2
DYDT = 3
M = 4
ID = 5
DT = 60.0 ** (-1)
G = -0.001


@numba.jit
def calculate_forces(bodies):
    """
    :param bodies: np.ndarray
    """
    forces = np.zeros((bodies.shape[1], )*2 + (2, ))
    for i in range(forces.shape[0]):
        for j in range(forces.shape[1]):
            vector_x = bodies[X, j] - bodies[X, i]
            vector_y = bodies[Y, j] - bodies[Y, i]
            norm = np.sqrt(vector_x**2 + vector_y**2)
            distance_squared = vector_x ** 2 + vector_y ** 2
            if distance_squared < 0.001:
                continue
            vector_x /= norm
            vector_y /= norm
            magnitude = G * bodies[M, i] * bodies[M, j] / distance_squared
            forces[i, j, X] = magnitude * vector_x / bodies[M, i]
            forces[i, j, Y] = magnitude * vector_y / bodies[M, i]

    total_forces = np.sum(forces, axis=0)

    return total_forces


class NBodyServerFactory(WebSocketServerFactory):


    def __init__(self, eventloop, *args, **kwargs):
        super(NBodyServerFactory, self).__init__(*args, **kwargs)
        self.clients = set()
        self.eventloop = eventloop
        self.bodies = np.array([
            np.random.uniform(-6.0, 6.0, N), # x
            np.random.uniform(-6.0, 6.0, N), # y
            np.zeros(N), # np.random.normal(0.0, 0.1, N), # dx/dt
            np.zeros(N), # np.random.normal(0.0, 0.1, N), # dy/dt
            np.random.normal(1.0, 1.0, N), # m
            np.arange(N) # id
        ])
        logger.info('Starting simulation')
        self.times = []
        self.tick()

    def tick(self):
        self.eventloop.call_later(DT, self.tick)
        start = time.time()
        message = {'time': time.time(), 'bodies': self.bodies.tolist()}
        logger.debug('Sending message: {}'.format(message))
        forces = calculate_forces(self.bodies)
        self.bodies[[DXDT, DYDT]] += forces.T * DT
        self.bodies[[X, Y]] += self.bodies[[DXDT, DYDT]] * DT

        self.broadcast(json.dumps(message))
        end = time.time()
        self.times.append(end-start)
        if(len(self.times) >= 1000):
            logger.info('Calculated {} ticks at {} ms/tick'.format(len(self.times), np.mean(self.times)*1000))
            self.times = []
            mean_x = self.bodies[X].mean()
            mean_y = self.bodies[Y].mean()
            if np.abs(mean_x) > 4 or np.abs(mean_y) > 4:
                logger.info('Recentering to {}, {}'.format(mean_x, mean_y))
                self.bodies[X] -= mean_x
                self.bodies[Y] -= mean_y


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



