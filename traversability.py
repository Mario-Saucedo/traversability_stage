import cv2
import numpy as np
import ros2_numpy as rnp
import matplotlib.pyplot as plt
import math
# from numba import njit, prange
import numba
from numba import cuda
import torch
import struct

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs_py.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
# from sensor_msgs.msg import PointField
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
# MODEL_PATH = "LRSPP.torch"

MAX_RANGE = 15
FRAME = "world"

print("Using: ", DEVICE)

@cuda.jit
def calculate_normals_kernel(depth_image, normal_map, normal_x_channel):
    x, y = cuda.grid(2)
    height, width = depth_image.shape
    
    if 1 <= x < height - 1 and 1 <= y < width - 1:
        dzdx = (depth_image[x + 1, y] - depth_image[x - 1, y]) / 2.0
        dzdy = (depth_image[x, y + 1] - depth_image[x, y - 1]) / 2.0

        if dzdy < 0:
            dzdy = -dzdy

        magnitude = math.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0 ** 2)

        if magnitude != 0:
            normal_map[x, y, 0] = int(255 * dzdx / magnitude)
            normal_map[x, y, 1] = int(255 * dzdy / magnitude)
            normal_map[x, y, 2] = int(255 * 1.0 / magnitude)
            normal_x_channel[x, y] = 255 * dzdx / magnitude
        else:
            normal_map[x, y, 0] = 0
            normal_map[x, y, 1] = 0
            normal_map[x, y, 2] = 255
            normal_x_channel[x, y] = 0
			
@cuda.jit
def compute_traversability_kernel(segmentation_image, normal_map, traversability_map):
    row, col = cuda.grid(2)

    if row < segmentation_image.shape[0] and col < segmentation_image.shape[1]:
        normal_value = normal_map[row, col]
        segment_value = segmentation_image[row, col]

        if math.isnan(normal_value) or math.isnan(segment_value) or normal_value == 0 or segment_value <= 1:
            traversability_map[row, col] = 0
        elif abs(normal_value) >= 250:
            traversability_map[row, col] = 51 * segment_value
        else:
            seg_value = segment_value / 51.0 - 1.0
            exp_part = math.exp(-(10.0 - 10.0 * normal_value) * 4.0 / seg_value)
            traversability_value = 255.0 * (seg_value / 4.0) * exp_part
            traversability_map[row, col] = min(max(int(traversability_value), 0), 255)

class CTraversability(Node):
	def __init__(self):
		super().__init__('c_traversability')
		self.bridge = CvBridge()
		
		# Create subscribers for RGB, depth images, and point cloud
		self.rgb_sub = Subscriber(self, Image, '/husky1/camera/color/image_raw')
		self.depth_sub = Subscriber(self, Image, '/husky1/camera/aligned_depth_to_color/image_raw')
		self.pcl_sub = Subscriber(self, PointCloud2, '/husky1/lidar_points')

		# Create subscribers for camra info
		self.intrinsics_sub = self.create_subscription(
			CameraInfo,
			'/husky1/camera/color/camera_info',  # Topic name might vary, adjust as needed
			self.callback_intrinsics,
			1)

		# Create a synchronizer to sync the RGB, depth images, and point cloud
		self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.pcl_sub], queue_size=1, slop=0.25)
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
		self.tvec = np.asarray([0.01,-0.06,0.02])
		self.rvec,_ = cv2.Rodrigues(np.asarray([[0.0,0.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]]))

		#---------------------Color---------------------#
		self.cmap = plt.cm.jet
		self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]

		################## NETWORK #####################
		# self.Net = lraspp_mobilenet_v3_large()
		# self.Net.classifier = LRASPPHead(40, 960, OUT_CHANNELS, 128)

		# self.Net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) # Load trained model
		self.Net = torch.jit.load('./model_lraspp.pt')

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
		depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

		# Calculate the normals of the depth image
		normals_image, normals_x = self.calculate_normals(depth_image)
		print("Normals...")

		# Apply semantic segmentation to the image
		rgb_image = cv2.cvtColor(bgr_image , cv2.COLOR_BGR2RGB)
		segmentation_image = self.semantic_segmentation(rgb_image)
		print("Segmentation...")

		#print("New data")

		# Process the images and point cloudcalculate_normals(self, depth_image)
		traversability_image = self.compute_traversability(segmentation_image, normals_x)
		dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		traversability_image = cv2.dilate(traversability_image, dilation_kernel, iterations=5)
		print("Traversability YAY!!!!")
		
		# Convert the image back to a ROS Image message
		normals_msg = self.bridge.cv2_to_imgmsg(normals_image, encoding='rgb8')
		segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_image, encoding='32FC1')
		traversability_msg = self.bridge.cv2_to_imgmsg(traversability_image, encoding='32FC1')
		
		# Publish
		self.normals_pub.publish(normals_msg)
		self.segmentation_pub.publish(segmentation_msg)
		self.traversability_pub.publish(traversability_msg)

		# Convert PointCloud2 to an array
		point_cloud = self.pointcloud2_to_array(pcl_msg)

		# Cobvert pointcloud to pixels
		pixels, _ = cv2.projectPoints(point_cloud, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)

		depth_ol = rgb_image.copy()

		points_xyz = []
		points_trav = []
		points_rgb = []

		# Iterate over pixels and process
		for idxPixel, pixel in enumerate(pixels):
			# Extract coordinates from pixel
			px, py = pixel[0]
			
			# Safely convert coordinates to integers
			px = self.safe_convert_to_int(px)
			py = self.safe_convert_to_int(py)
			
			# Check if coordinates are within bounds
			if px < self.width and py < self.height and px >= 0 and py >= 0 and point_cloud[idxPixel][0] > self.tvec[0]:
				depth = int(np.linalg.norm(point_cloud[idxPixel]) * 255 / MAX_RANGE)
				if depth > 255:
					depth = 255
				
				color_range = self.cmaplist[depth]
				depth_ol = cv2.circle(depth_ol, (px, py), 1, (int(color_range[0] * 255), int(color_range[1] * 255), int(color_range[2] * 255)), -1)
				
				points_xyz.append([point_cloud[idxPixel][0], point_cloud[idxPixel][1], point_cloud[idxPixel][2]])
				points_trav.append(traversability_image[py, px])
				points_rgb.append(rgb_image[py, px])
				#print(rgb_image[py, px])

		print("Points...")

		# for idxPixel, pixel in enumerate(pixels):
		# 	if pixel[0][0]<self.width and pixel[0][1]<self.height and pixel[0][0]>=0 and pixel[0][1]>=0 and point_cloud[idxPixel][0]>self.tvec[0]:

		# 		depth = int(np.linalg.norm(point_cloud[idxPixel])*255/MAX_RANGE)
		# 		if depth > 255:
		# 			depth = 255

		# 		color_range = self.cmaplist[depth]
		# 		depth_ol = cv2.circle(depth_ol, tuple([int(pixel[0][0]), int(pixel[0][1])]), 1, (int(color_range[0]*255), int(color_range[1]*255), int(color_range[2]*255)), -1)

		# 		points_xyz.append([point_cloud[idxPixel][0], point_cloud[idxPixel][1], point_cloud[idxPixel][2]])
		# 		points_trav.append(traversability_image[int(pixel[0][1]),int(pixel[0][0])])
		# 		points_rgb.append(rgb_image[int(pixel[0][1]),int(pixel[0][0])])

		# depth_ol, points_xyz, points_trav, points_rgb = self.process_pixels(
		# 	pixels, point_cloud, traversability_image, rgb_image, depth_ol
		# )

		# header = Header()
		# header.frame_id = FRAME
		# header.stamp = self.get_clock().now().to_msg()

		# fields = [
		# 	PointField('x', 0, PointField.FLOAT32, 1),
		# 	PointField('y', 4, PointField.FLOAT32, 1),
		# 	PointField('z', 8, PointField.FLOAT32, 1),
		# 	PointField('traversability', 12, PointField.FLOAT32, 1)
		# ]

		#points_dic = {"xyz" : points_xyz, "intesity": points_trav, "rgb" : points_rgb}

		# data = np.zeros(len(points_trav), dtype=[
		# 	('x', np.float32),
		# 	('y', np.float32),
		# 	('z', np.float32),
		# 	('intensity', np.float32),
		# 	('rgb', np.uint8, (3,))
		# ])

		# data['x'] = points_xyz[:, 0]
		# data['y'] = points_xyz[:, 1]
		# data['z'] = points_xyz[:, 2]
		# data['intensity'] = points_trav
		# data['rgb'] = points_rgb

		# if len(points_trav) > 0: 
		# 	# point_cloud = pcl2.create_cloud(header, fields, points_trav)
		# 	point_cloud = rnp.point_cloud2.array_to_point_cloud2(data, FRAME)
		# 	# point_cloud = rnp.msgify(PointCloud2, data)
		# 	point_cloud.header.frame_id = FRAME
		# 	self.trav_pcl_pub.publish(point_cloud)

		points_xyz = np.array(points_xyz)
		points_trav = np.array(points_trav)
		points_rgb = np.array(points_rgb)

		point_cloud_msg = self.create_point_cloud2(points_xyz, points_trav, points_rgb)
		self.trav_pcl_pub.publish(point_cloud_msg)

		self.depth_ol_pub.publish(self.bridge.cv2_to_imgmsg(depth_ol, 'rgb8'))

	def safe_convert_to_int(self, value):
		"""Safely convert a value to an integer, handling infinities and NaNs."""
		if np.isinf(value) or np.isnan(value):
			return 0  # Default or fallback value
		return int(value)
			
	#@njit(parallel=True)
	# @cuda.jit
	# def calculate_normals(self, depth_image):
	# 	# Convert depth_image to float32 to avoid overflow issues
	# 	depth_image = depth_image.astype(np.float32)
	# 	depth_image = cv2.resize(depth_image, (self.width, self.height))
	# 	#print(depth_image.shape)
		
	# 	#height, width = depth_image.shape
	# 	normal_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
	# 	normal_x_channel = np.zeros((self.height, self.width), dtype=np.float32)

	# 	for x in range(1, self.height - 1):
	# 		for y in range(1, self.width - 1):
	# 			#print(x, "&", y)
	# 			dzdx = (depth_image[x + 1, y] - depth_image[x - 1, y]) / 2.0
	# 			dzdy = (depth_image[x, y + 1] - depth_image[x, y - 1]) / 2.0

	# 			# Handle negative dzdy
	# 			if dzdy < 0:
	# 				dzdy = -dzdy

	# 			normal_vector = [-dzdx, -dzdy, 1.0]
	# 			magnitude = math.sqrt(normal_vector[0] ** 2 + normal_vector[1] ** 2 + normal_vector[2] ** 2)

	# 			if magnitude != 0:
	# 				normalized_vector = [
	# 					255 * normal_vector[0] / magnitude,
	# 					255 * normal_vector[1] / magnitude,
	# 					255 * normal_vector[2] / magnitude
	# 				]
	# 			else:
	# 				normalized_vector = [0, 0, 255]  # Default to a unit vector pointing up

	# 			# Ensure the normalized_vector does not contain NaN or infinite values
	# 			if not any(math.isnan(v) or math.isinf(v) for v in normalized_vector):
	# 				#normal_map[x, y] = normalized_vector
	# 				normal_x_channel[x, y] = normalized_vector[0]

	# 	return normal_map, normal_x_channel

	def calculate_normals(self, depth_image):
		depth_image = depth_image.astype(np.float32)
		depth_image = cv2.resize(depth_image, (self.width, self.height))

		normal_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
		normal_x_channel = np.zeros((self.height, self.width), dtype=np.float32)

		threadsperblock = (16, 16)
		blockspergrid_x = math.ceil(depth_image.shape[0] / threadsperblock[0])
		blockspergrid_y = math.ceil(depth_image.shape[1] / threadsperblock[1])
		blockspergrid = (blockspergrid_x, blockspergrid_y)

		calculate_normals_kernel[blockspergrid, threadsperblock](depth_image, normal_map, normal_x_channel)

		return normal_map, normal_x_channel

	#@njit(parallel=True)
	# @cuda.jit
	# def compute_traversability(self, segmentation_image, normal_map):
	# 	# height, width = segmentation_image.shape
	# 	traversability_map = np.zeros((self.height, self.width), dtype=np.float32)
	# 	# print(segmentation_image.shape)
	# 	# print(normal_map.shape)
		
	# 	for row in range(self.height):
	# 		for col in range(self.width):
	# 			normal_value = normal_map[row, col]
	# 			segment_value = segmentation_image[row, col]

	# 			if np.isnan(normal_value) or np.isnan(segment_value) or normal_value == 0 or segment_value <= 1:
	# 				traversability_map[row, col] = 0
	# 			elif abs(normal_value) >= 250:
	# 				traversability_map[row, col] = 51 * segment_value
	# 			else:
	# 				threshold_value = -0.039 * abs(normal_value) + 10.0
	# 				traversability_map[row, col] = 51 * segment_value * np.exp(-threshold_value * 1.5 / (segment_value - 1))

	# 	return traversability_map

	def compute_traversability(self, segmentation_image, normal_map):
		traversability_map = np.zeros((self.height, self.width), dtype=np.float32)

		# Copy data to the device
		segmentation_image_device = cuda.to_device(segmentation_image)
		normal_map_device = cuda.to_device(normal_map)
		traversability_map_device = cuda.to_device(traversability_map)

		# Define the number of threads in a block and the number of blocks in a grid
		threadsperblock = (16, 16)
		blockspergrid_x = int(np.ceil(segmentation_image.shape[0] / threadsperblock[0]))
		blockspergrid_y = int(np.ceil(segmentation_image.shape[1] / threadsperblock[1]))
		blockspergrid = (blockspergrid_x, blockspergrid_y)

		# Launch the kernel
		compute_traversability_kernel[blockspergrid, threadsperblock](
			segmentation_image_device, normal_map_device, traversability_map_device
		)

		# Copy the result back to the host
		traversability_map = traversability_map_device.copy_to_host()

		return traversability_map
		
	def semantic_segmentation(self, rgb_image):
		input_tensor = self.transformRGB(rgb_image)
		input_tensor  = input_tensor.to(DEVICE).unsqueeze(0)

		with torch.no_grad():
			pred = self.Net(input_tensor)#['out'] # make prediction
			pred = pred.softmax(dim=1)

		# pred = tf.Resize((self.height, self.width))(pred[0])
		# pred = pred[[0, 5, 4, 3, 2, 1],:,:]

		segmentation_image = torch.argmax(torch.squeeze(pred, 0), dim=0).cpu().detach().float().numpy() 
		# print(segmentation_image)

		# Resize the image using OpenCV
		segmentation_image = cv2.resize(segmentation_image, (self.width, self.height))
		
		return segmentation_image

	def pointcloud2_to_array(self, point_cloud_msg):
		# Convert PointCloud2 message to a numpy array
		point_cloud = rnp.point_cloud2.point_cloud2_to_array(point_cloud_msg)
		point_cloud = point_cloud["xyz"]
		point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1)]
		# for point in pcl2.read_points(point_cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
		# 	point_cloud.append([point[0], point[1], point[2] if len(point) > 2 else 0])
		return np.array(point_cloud)
	
	def transformRGB(self, image):
		# Resize the image using OpenCV
		image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

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

		return  tensor#.to(DEVICE).unsqueeze(0)
	
	def create_point_cloud2(self, points_xyz, points_trav, points_rgb):

		# Define data type for ROS PointCloud2
		ros_dtype = PointField.FLOAT32
		dtype = np.float32
		itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes

		# Add a new axis to points_trav to match the dimensions for concatenation
		points_trav = points_trav[:, np.newaxis]  # Shape will be (number_of_points, 1)

		# Convert points_rgb to a single uint32 value per point
		points_rgb_uint32 = np.left_shift(points_rgb[:, 0].astype(np.uint32), 16) + \
							np.left_shift(points_rgb[:, 1].astype(np.uint32), 8) + \
							points_rgb[:, 2].astype(np.uint32)
		points_rgb_uint32 = points_rgb_uint32[:, np.newaxis]  # Shape will be (number_of_points, 1)

		# Concatenate points_xyz, points_trav, and points_rgb along the last axis
		points = np.hstack((points_xyz, points_trav, points_rgb_uint32))

		# Convert data to bytes
		data = points.astype(dtype).tobytes()

		# Define the fields for the PointCloud2 message
		fields = [
			PointField(name='x', offset=0 * itemsize, datatype=ros_dtype, count=1),
			PointField(name='y', offset=1 * itemsize, datatype=ros_dtype, count=1),
			PointField(name='z', offset=2 * itemsize, datatype=ros_dtype, count=1),
			PointField(name='traversability', offset=3 * itemsize, datatype=ros_dtype, count=1),
			PointField(name='rgb', offset=4 * itemsize, datatype=PointField.UINT32, count=1)
		]

		# The PointCloud2 message also has a header which specifies which 
		# coordinate frame it is represented in.
		header = Header(frame_id=FRAME)

		# Create and return the PointCloud2 message
		return PointCloud2(
			header=header,
			height=1,
			width=points.shape[0],
			is_dense=False,
			is_bigendian=False,
			fields=fields,
			point_step=(itemsize * 5),  # Every point consists of four float32s and one uint32.
			row_step=(itemsize * 5 * points.shape[0]),
			data=data
		)
		
if __name__ == '__main__':
	rclpy.init()
	node = CTraversability()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()
