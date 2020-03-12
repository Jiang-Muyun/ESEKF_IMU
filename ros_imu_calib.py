import math
import time
import roslib
import rospy
import numpy as np
from sensor_msgs.msg import Imu

class IMU_Subscriber():
    def __init__(self, topic_name):
        self.topic_name = topic_name
        self.t0 = 0
        self.sub = rospy.Subscriber(topic_name, Imu, self.imu_callback, queue_size=4)
        
        self.imu_buf = []

    def imu_callback(self,data):
        t = data.header.stamp.to_time()
        if self.t0 == 0:
            self.t0 = t
        t = t - self.t0

        acc = np.array((data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z), dtype=np.float32)
        gyro = np.array((data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z), dtype=np.float32)
        imu_data = np.array((t, gyro[0], gyro[1], gyro[2], acc[0], acc[1], acc[2]))
        
        self.imu_buf.append(imu_data)
        if len(self.imu_buf) == 1000:
            imu_data = np.array(self.imu_buf, dtype=np.float32)
            np.savez('tmp/imu_still.npz', imu_data = imu_data)
            print('gyro', imu_data[:,1:4].mean(0))
            print('acc', imu_data[:,4:].mean(0))
            print('Data Saved')
            rospy.signal_shutdown(0)
        else:
            print(len(self.imu_buf))

if __name__ == '__main__':
    rospy.init_node('ekf_img')
    IMU_Subscriber('/mavros/imu/data_raw')
    rospy.spin()
