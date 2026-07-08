import socket
import struct
import threading
import time

DOG_IP = "192.168.137.120"
DOG_PORT = 43893

HEARTBEAT = 0x21040001
STAND_SIT = 0x21010202
NAVI_MODE = 0x21010C03
POSE_MODE = 0x21010D05
MOVE_MODE = 0x21010D06
ADJUST_PITCH = 0x21010130

DOC_PITCH_MIN = -32767
DOC_PITCH_MAX = 32767
INT32_MIN = -2147483648
INT32_MAX = 2147483647


class HeartbeatThread:
    def __init__(self, dog, hz=3):
        self.dog = dog
        self.hz = hz
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _loop(self):
        interval = 1.0 / self.hz
        while self._running:
            self.dog.send_heartbeat()
            time.sleep(interval)


class Lite3UDP:
    def __init__(self, ip=DOG_IP, port=DOG_PORT):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 0)

    def send_simple(self, code, value=0):
        packet = struct.pack("<IiI", code, int(value), 0)
        self.sock.sendto(packet, (self.ip, self.port))

    def send_heartbeat(self):
        self.send_simple(HEARTBEAT)

    def switch_navi_mode(self):
        self.send_simple(NAVI_MODE)

    def stand_sit(self):
        self.send_simple(STAND_SIT)

    def switch_pose_mode(self):
        self.send_simple(POSE_MODE)

    def switch_move_mode(self):
        self.send_simple(MOVE_MODE)

    def send_pitch_axis(self, value: int):
        value = int(value)
        if value < INT32_MIN or value > INT32_MAX:
            raise ValueError(f"pitch value {value} is outside signed int32 range")
        if value < DOC_PITCH_MIN or value > DOC_PITCH_MAX:
            # Intentional: keep compatibility with previous experiments.
            pass
        self.send_simple(ADJUST_PITCH, value)

    def close(self):
        self.sock.close()
