from threading import Thread, Lock
import socket

from bokeh.server.server import Server
from bokeh.embed import server_document
from tornado.ioloop import IOLoop


mutex = Lock()


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


class BokehServer:
    """Bokeh server (based on Tornado HTTPServer) to synchronize client and server
    documents via websocket.
    """
    def __init__(self, application, websocket_origin=['*']):
        if not callable(application):
            raise ValueError()
        self.application = application
        self.websocket_origin = websocket_origin
        self.thread = Thread(target=self._server_thread)
        self.thread.start()

    def _server_thread(self):
        self.io_loop = IOLoop()
        with mutex:
            self.port = get_free_tcp_port()
            self.server = Server(
                self.application, io_loop=self.io_loop,
                allow_websocket_origin=self.websocket_origin,
                port=self.port)
        self.server.start()
        self.server.io_loop.start()

    def get_script(self):
        return server_document('http://localhost:' + str(self.port))