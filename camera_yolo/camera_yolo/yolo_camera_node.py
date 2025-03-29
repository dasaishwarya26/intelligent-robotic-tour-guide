import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Change if your topic is different
            self.image_callback,
            10)
        self.model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, etc.
        self.get_logger().info("YOLOv8 camera node started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, verbose=False)[0]
        annotated_frame = results.plot()
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

