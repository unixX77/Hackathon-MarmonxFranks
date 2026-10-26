#!/usr/bin/env python3
"""Read-only Franka health inspection. No robot commands or error recovery."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/lego_sorter'))


def main():
    import rclpy
    import yaml
    from lego_sorter.hardware import RobotHealth

    c = yaml.safe_load((ROOT / 'src/lego_sorter/config/hardware.yaml').read_text())
    rclpy.init()
    node = rclpy.create_node('lego_robot_health_readonly')
    health = RobotHealth(node, c)
    try:
        deadline = time.monotonic() + 5
        while True:
            rclpy.spin_once(node, timeout_sec=.05)
            try:
                state = health.check()
            except RuntimeError as exc:
                if time.monotonic() < deadline:
                    continue
                print('NOT READY: ' + str(exc))
                print(json.dumps(health.snapshot(), indent=2))
                return 1
            print('Robot health: active FCI control, no current errors, physical robot time advancing.')
            print(json.dumps(state, indent=2))
            print('Read-only check complete. No motion or recovery commands sent.')
            return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
