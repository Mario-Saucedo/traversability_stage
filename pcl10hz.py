import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class PointCloudRepublisher(Node):

	def __init__(self):
		super().__init__('pointcloud_republisher')
		
		# Create a subscriber to the input pointcloud topic
		self.subscription = self.create_subscription(
			PointCloud2,
			'/traversability/points',
			self.pointcloud_callback,
			10)  # QoS profile depth
		
		# Create a publisher to the output pointcloud topic
		self.publisher = self.create_publisher(
			PointCloud2,
			'/husky1/traversability/points',
			10)  # QoS profile depth
		
		# Timer to control the publishing rate
		self.timer = self.create_timer(0.1, self.timer_callback)  # 0.1 seconds for 10 Hz
		self.latest_msg = None

	def pointcloud_callback(self, msg):
		# Save the latest message received
		self.latest_msg = msg

	def timer_callback(self):
		if self.latest_msg is not None:
			# Republish the latest message at 10 Hz
			self.publisher.publish(self.latest_msg)
			self.get_logger().info('Republished PointCloud message')

if __name__ == '__main__':
	rclpy.init()
	node = PointCloudRepublisher()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()