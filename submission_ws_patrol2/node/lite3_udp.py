import socket
import struct

DOG_IP = "192.168.137.120"
DOG_PORT = 43893

HEARTBEAT = 0x21040001
NAVI_MODE = 0x21010C03
POSE_MODE = 0x21010D05
ADJUST_PITCH = 0x21010130


class Lite3UDP:
    def __init__(self, ip=DOG_IP, port=DOG_PORT):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 0)

    def _send_simple(self, code, value=0):
        packet = struct.pack("<IiI", int(code), int(value), 0)
        self.sock.sendto(packet, (self.ip, self.port))

    def send_heartbeat(self):
        self._send_simple(HEARTBEAT)

    def switch_navi_mode(self):
        self._send_simple(NAVI_MODE)

    def switch_pose_mode(self):
        self._send_simple(POSE_MODE)

    def send_pitch_axis(self, value):
        self._send_simple(ADJUST_PITCH, value)

    def close(self):
        self.sock.close()
