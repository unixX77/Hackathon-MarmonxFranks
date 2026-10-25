#!/usr/bin/env python3
#  Copyright (c) 2024 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import rclpy
import math
import time
from rclpy import Parameter
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class FakeGripperStatePublisher(Node):

    def __init__(self):
        super().__init__('fake_gripper_state_publisher')
        self.publisher_ = self.create_publisher(JointState, '~/joint_states', 1)
        timer_period = 0.02  # smooth visualization; no contact/dynamics simulation
        self.width = self.target_width = 0.07
        self.speed = 0.01
        self.previous_time = time.monotonic()
        self.command_subscription = self.create_subscription(
            Float64MultiArray, '~/fake_width_command', self.on_command, 10)

        self.declare_parameter('joint_names', Parameter.Type.STRING_ARRAY)
        self.joint_names = (
            self.get_parameter('joint_names').get_parameter_value().string_array_value
        )

        assert len(self.joint_names) == 2
        self.timer = self.create_timer(timer_period, self.publish_state)

    def on_command(self, message):
        if (len(message.data) != 2 or not all(math.isfinite(v) for v in message.data)
                or not 0 <= message.data[0] <= .08 or not 0 < message.data[1] <= .05):
            self.get_logger().warning('Invalid fake gripper [width_m, speed_m_s] command')
            return
        self.target_width, self.speed = message.data

    def publish_state(self):
        now = time.monotonic()
        dt = min(.1, max(0., now-self.previous_time))
        self.previous_time = now
        delta = self.target_width-self.width
        increment = max(-self.speed*dt, min(self.speed*dt, delta))
        self.width += increment
        joint_states = JointState()
        joint_states.header.stamp = self.get_clock().now().to_msg()
        joint_states.name = self.joint_names
        joint_states.position = [self.width/2, self.width/2]
        joint_states.velocity = [increment/dt/2 if dt > 0 else 0.] * 2
        joint_states.effort = [0.0, 0.0]
        self.publisher_.publish(joint_states)


def main(args=None):
    rclpy.init(args=args)

    state_publisher = FakeGripperStatePublisher()

    rclpy.spin(state_publisher)
    state_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
