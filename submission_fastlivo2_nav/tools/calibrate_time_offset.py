#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import numpy as np
from collections import deque
import signal, sys

class Calib(Node):
    def __init__(self):
        super().__init__('time_calib')
        self.lt, self.it, self.diffs = deque(maxlen=1000), deque(maxlen=1000), deque(maxlen=1000)
        self.create_subscription(PointCloud2, '/livox/lidar', self.l_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.i_cb, 10)
        self.create_timer(5.0, self.stats)
        self.get_logger().info('Press Ctrl-C after 30-60s')
    def l_cb(self, m):
        self.lt.append(m.header.stamp.sec + m.header.stamp.nanosec*1e-9); self.calc()
    def i_cb(self, m):
        self.it.append(m.header.stamp.sec + m.header.stamp.nanosec*1e-9); self.calc()
    def calc(self):
        if len(self.lt)>5 and len(self.it)>5:
            self.diffs.append(self.lt[-1] - self.it[-1])
    def stats(self):
        if len(self.diffs)<10:
            self.get_logger().info(f'pairs={len(self.diffs)}')
            return
        a = np.array(self.diffs)
        self.get_logger().info(f'offset={np.mean(a)*1000:.2f}ms std={np.std(a)*1000:.2f}ms | img_time_offset={np.mean(a):.6f}')

def main():
    rclpy.init()
    n = Calib()
    signal.signal(signal.SIGINT, lambda s,f: (n.stats(), n.destroy_node(), rclpy.shutdown(), sys.exit(0)))
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.stats(); n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
