import time
import roslib
import rospy
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import math
import numpy as np
import numpy.linalg as la
import transformations as tr
from esekf import load_IMU_Param, ESEKF

class IMU_Subscriber():
    def __init__(self, topic_name, imu_yaml):
        self.topic_name = topic_name
        self.t0 = 0
        self.sub = rospy.Subscriber(topic_name, Imu, self.imu_callback, queue_size=10)
        imu_param = load_IMU_Param(imu_yaml)

        self.init_nominal_state = np.zeros((19,))
        self.init_nominal_state[4] = 1                           # init p, q, v
        self.init_nominal_state[10:13] = imu_param.bias_acc      # init ba
        self.init_nominal_state[13:16] = imu_param.bias_gyro     # init bg
        self.init_nominal_state[16:19] = imu_param.gravity       # init g
        self.estimator = ESEKF(self.init_nominal_state, imu_param)

        # self.sigma_measurement_p = 0.02   # in meters
        # self.sigma_measurement_q = 0.015  # in rad
        # self.sigma_measurement = np.eye(6)
        # self.sigma_measurement[0:3, 0:3] *= self.sigma_measurement_p**2
        # self.sigma_measurement[3:6, 3:6] *= self.sigma_measurement_q**2
        
        self.imu_buf = []
        self.pose_buf = []

    def imu_callback(self,data):
        t = data.header.stamp.to_time()
        if self.t0 == 0:
            self.t0 = t
        t = t- self.t0

        gyro = np.array((data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z), dtype=np.float32)
        acc = np.array((data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z), dtype=np.float32)
        imu_data = np.array((t, gyro[0], gyro[1], gyro[2], acc[0], acc[1], acc[2]))
        
        self.estimator.predict(imu_data)
        pos = self.estimator.nominal_state[:3]
        quat = self.estimator.nominal_state[3:7]
        euler = tr.euler_from_quaternion(quat)

        self.pose_buf.append(self.estimator.nominal_state[1:7].copy())
        if t > 10:
            pose_data = np.array(self.pose_buf, dtype=np.float32)
            np.savez('tmp/pose.npz', pose_data = pose_data)
            rospy.signal_shutdown(0)

        print('%.2f |'%t,'%6.2f %6.2f %6.2f |'%(pos[0], pos[1], pos[2]), '%6.2f %6.2f %6.2f'%(euler[0], euler[1], euler[2]))

if __name__ == '__main__':
    rospy.init_node('ekf_img')
    IMU_Subscriber('/mavros/imu/data', 'data/imu_pix2.yaml')
    rospy.spin()