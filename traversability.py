import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit, prange
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs.msg._point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

# import torchvision.transforms as tf
# from torchvision.models.segmentation import lraspp_mobilenet_v3_large
# from torchvision.models.segmentation.lraspp import LRASPPHead

#Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IN_CHANNELS = 3
OUT_CHANNELS = 6
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
MODEL_PATH = "LRSPP.torch"

MAX_RANGE = 45
FRAME = "world"

print("Using: ", DEVICE)

class CTraversability(Node):
	def __init__(self):
		super().__init__('c_traversability')
		self.bridge = CvBridge()
		
		# Create subscribers for RGB, depth images, and point cloud
		self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
		self.depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
		self.pcl_sub = Subscriber(self, PointCloud2, '/husky1/ouster/points')

		# Create subscribers for camra info
		self.intrinsics_sub = self.create_subscription(
			CameraInfo,
			'/camera/color/camera_info',  # Topic name might vary, adjust as needed
			self.callback_intrinsics,
			1)

		# Create a synchronizer to sync the RGB, depth images, and point cloud
		self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.pcl_sub], queue_size=10, slop=0.1)
		self.ts.registerCallback(self.callback)
		
		# Publishers
		self.traversability_pub = self.create_publisher(Image, '/traversability/image_raw', 1)
		self.segmentation_pub = self.create_publisher(Image, '/traversability/segmentation/image_raw', 1)
		self.normals_pub = self.create_publisher(Image, '/traversability/surface_nromals/image_raw', 1)
		self.trav_pcl_pub = self.create_publisher(PointCloud2, "/traversability/points", 1)
		self.depth_ol_pub = self.create_publisher(Image, "/lidar/aligned_range_to_image/image_raw", 1)

		#-----------Default intrinsics---------------#
		self.height = 720
		self.width = 1280
		self.camera_matrix = np.asarray([635.2752075195312, 0.0, 652.32861328125, 0.0, 634.6666259765625, 358.4922180175781, 0.0, 0.0, 1.0]).reshape((3, 3))
		self.dist_coeffs = np.asarray([-0.05529356002807617, 0.06882049143314362, -0.00044326853821985424, 0.0005935364752076566, -0.021897781640291214])
		
		#-------------Default extrinsics----------------#
		self.tvec = np.asarray([0.10,0.00,-0.10])
		self.rvec,_ = cv2.Rodrigues(np.asarray([[0.0,0.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]]))

		#---------------------Color---------------------#
		self.cmap = plt.cm.jet
		self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]

		################## NETWORK #####################
		# self.Net = lraspp_mobilenet_v3_large()
		# self.Net.classifier = LRASPPHead(40, 960, OUT_CHANNELS, 128)

		# self.Net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) # Load trained model
		self.Net = torch.jit.load('model_lraspp.pt')

		self.Net.to(DEVICE)
		self.Net.eval() # Set to evaluation mode

		print("Network Set")

	def callback_intrinsics(self, data):
		self.height = data.height
		self.width = data.width
		self.camera_matrix = np.asarray(data.k).reshape((3, 3))
		self.dist_coeffs = np.asarray(data.d)
		self.get_logger().info(f"Camera Matrix: {self.camera_matrix}, Dist Coeffs: {self.dist_coeffs}")
		# self.intrinsics_sub.reset()
	
	def callback(self, rgb_msg, depth_msg, pcl_msg):
		# Convert ROS Image messages to OpenCV images
		bgr_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
		depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

		print("New data")

		# Calculate the normals of the depth image
		normals_image, normals_x = self.calculate_normals(depth_image)

		# Apply semantic segmentation to the image
		rgb_image = cv2.cvtColor(bgr_image , cv2.COLOR_BGR2RGB)
		segmentation_image = self.semantic_segmentation(rgb_image)

		# Process the images and point cloudcalculate_normals(self, depth_image)
		traversability_image = self.compute_traversability(segmentation_image, normals_image, point_cloud)
		
		# Convert PointCloud2 to an array
		point_cloud = self.pointcloud2_to_array(pcl_msg)

		# Cobvert pointcloud to pixels
		pixels, _ = cv2.projectPoints(point_cloud, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)

		depth_ol = cv2.Copy(rgb_image)

		points_trav = []

		for idxPixel, pixel in enumerate(pixels):
			if pixel[0][0]<self.width and pixel[0][1]<self.height and pixel[0][0]>=0 and pixel[0][1]>=0 and point_cloud[idxPixel][0]>self.tvec[0]:

				depth = int(np.linalg.norm(point_cloud[idxPixel])*255/MAX_RANGE)
				if depth > 255:
					depth = 255

				color_range = self.cmaplist[depth]
				depth_ol = cv2.circle(depth_ol, tuple([int(pixel[0][0]), int(pixel[0][1])]), 1, (int(color_range[0]*255), int(color_range[1]*255), int(color_range[2]*255)), -1)

				points_trav.append([point_cloud[idxPixel][0], point_cloud[idxPixel][1], point_cloud[idxPixel][2], traversability_image[int(pixel[0][1]),int(pixel[0][0])]])

		header = Header()
		header.frame_id = FRAME
		header.stamp = self.get_clock().now().to_msg()

		fields = [
			PointField('x', 0, PointField.FLOAT32, 1),
			PointField('y', 4, PointField.FLOAT32, 1),
			PointField('z', 8, PointField.FLOAT32, 1),
			PointField('traversability', 12, PointField.FLOAT32, 1)
		]

		if len(points_trav) > 0: 
			point_cloud = pcl2.create_cloud(header, fields, points_trav)
			self.trav_pcl_pub.publish(point_cloud)

		self.depth_ol_pub.publish(self.bridge.cv2_to_imgmsg(depth_ol, 'bgr8'))
		
		# Convert the image back to a ROS Image message
		normals_msg = self.bridge.cv2_to_imgmsg(normals_image, encoding='rgb8')
		segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_image, encoding='mono8')
		traversability_msg = self.bridge.cv2_to_imgmsg(traversability_image, encoding='mono8')
		
		# Publish
		self.normals_pub.publish(normals_msg)
		self.segmentation_pub.publish(segmentation_msg)
		self.traversability_pub.publish(traversability_msg)
			
	#@njit(parallel=True)
	def calculate_normals(self, depth_image):
		# Convert depth_image to float32 to avoid overflow issues
		depth_image = depth_image.astype(np.float32)
		
		height, width = depth_image.shape
		normal_map = np.zeros((height, width, 3), dtype=np.uint8)
		normal_x_channel = np.zeros((height, width, 1), dtype=np.float32)

		for x in range(1, height - 1):
			for y in range(1, width - 1):
				dzdx = (depth_image[x + 1, y] - depth_image[x - 1, y]) / 2.0
				dzdy = (depth_image[x, y + 1] - depth_image[x, y - 1]) / 2.0

				# Handle negative dzdy
				if dzdy < 0:
					dzdy = -dzdy

				normal_vector = [-dzdx, -dzdy, 1.0]
				magnitude = math.sqrt(normal_vector[0] ** 2 + normal_vector[1] ** 2 + normal_vector[2] ** 2)

				if magnitude != 0:
					normalized_vector = [
						255 * normal_vector[0] / magnitude,
						255 * normal_vector[1] / magnitude,
						255 * normal_vector[2] / magnitude
					]
				else:
					normalized_vector = [0, 0, 255]  # Default to a unit vector pointing up

				# Ensure the normalized_vector does not contain NaN or infinite values
				if not any(math.isnan(v) or math.isinf(v) for v in normalized_vector):
					normal_map[x, y] = normalized_vector
					normal_x_channel[x, y] = normalized_vector[0]

		return normal_map, normal_x_channel
	
	#@njit(parallel=True)
	def compute_traversability(self, segmentation_image, normal_map):
		# height, width = segmentation_image.shape
		traversability_map = np.zeros((self.height, self.width), dtype=np.float32)
		
		for row in range(self.height):
			for col in range(self.width):
				normal_value = normal_map[row, col]
				segment_value = segmentation_image[row, col]

				if np.isnan(normal_value) or np.isnan(segment_value) or normal_value == 0 or segment_value <= 1:
					traversability_map[row, col] = 0
				elif abs(normal_value) >= 250:
					traversability_map[row, col] = 51 * segment_value
				else:
					threshold_value = -0.039 * abs(normal_value) + 10.0
					traversability_map[row, col] = 51 * segment_value * np.exp(-threshold_value * 1.5 / (segment_value - 1))

		return traversability_map
		
	def semantic_segmentation(self, rgb_image):
		input_tensor = self.transformRGB(rgb_image)
		input_tensor  = input_tensor .to(DEVICE).unsqueeze(0)

		with torch.no_grad():
			pred = self.Net(input_tensor)#['out'] # make prediction
			pred = pred.softmax(dim=1)

		#pred = tf.Resize((self.height, self.width))(pred[0])
		pred = pred[[0, 5, 4, 3, 2, 1],:,:]

		segmentation_image = torch.argmax(pred, dim=0).cpu().detach().float().numpy() 

		# Resize the image using OpenCV
		segmentation_image = cv2.resize(segmentation_image, (self.height, self.width))
		
		return segmentation_image

	def pointcloud2_to_array(self, point_cloud_msg):
		# Convert PointCloud2 message to a numpy array
		point_cloud = []
		for point in pcl2.read_points(point_cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
			point_cloud.append([point[0], point[1], point[2] if len(point) > 2 else 0])
		return np.array(point_cloud)
	
	def transformRGB(self, image):
		# Resize the image using OpenCV
		image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

		# Convert image to float32 and scale to [0, 1]
		image = image.astype(np.float32) / 255.0

		# Normalize the image
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])
		image = (image - mean) / std

		# Convert HWC (height, width, channel) to CHW (channel, height, width)
		image = np.transpose(image, (2, 0, 1))

		# Convert NumPy array to PyTorch tensor
		tensor = torch.tensor(image, dtype=torch.float32)

		return tensor
		
if __name__ == '__main__':
	rclpy.init()
	node = CTraversability()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()
