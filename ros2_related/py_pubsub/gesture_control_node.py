import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import json
from collections import deque
import mediapipe as mp

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')

        self.declare_parameter('model_path', 'models/gesture_model.tflite')
        self.declare_parameter('label_map_path', 'models/label_map.json')
        self.declare_parameter('confidence_threshold', 0.90)
        self.declare_parameter('window_size', 5)
        self.declare_parameter('linear_speed', 2.0)
        self.declare_parameter('angular_speed', 2.0)

        model_path     = self.get_parameter('model_path').value
        label_map_path = self.get_parameter('label_map_path').value
        self.conf_thresh   = self.get_parameter('confidence_threshold').value
        self.window_size   = self.get_parameter('window_size').value
        self.linear_speed  = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.get_logger().info(f'Model loaded from {model_path}')

        with open(label_map_path, 'r') as f:
            raw = json.load(f)
        self.label_map = {int(k): v for k, v in raw.items()}
        self.get_logger().info(f'Labels: {self.label_map}')

        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.vote_window = deque(maxlen=self.window_size)

        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Webcam
        self.cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(2)

        if not self.cap.isOpened():
            self.get_logger().error('Cannot open webcam!')
            raise RuntimeError('Webcam not available')

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('Gesture control node started. Press Q in window to quit.')

    def draw_landmarks_on_black(self, hand_landmarks, w, h):
        """Reproduce training data style: white lines + red dots on black 640x480."""
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw connections in white
        for connection in self.mp_hands.HAND_CONNECTIONS:
            x1 = int(hand_landmarks.landmark[connection[0]].x * w)
            y1 = int(hand_landmarks.landmark[connection[0]].y * h)
            x2 = int(hand_landmarks.landmark[connection[1]].x * w)
            y2 = int(hand_landmarks.landmark[connection[1]].y * h)
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw landmarks as red dots
        for lm in hand_landmarks.landmark:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), -1)

        return canvas

    def preprocess(self, canvas):
        """Convert landmark canvas to model input: 64x64 grayscale normalized."""
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, 64, 64, 1)

    def predict(self, img):
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        idx = int(np.argmax(output))
        return self.label_map[idx], float(output[idx])

    def gesture_to_twist(self, gesture):
        msg = Twist()
        if gesture == 'up':
            msg.linear.x = self.linear_speed
        elif gesture == 'down':
            msg.linear.x = -self.linear_speed
        elif gesture == 'left':
            msg.angular.z = self.angular_speed
        elif gesture == 'right':
            msg.angular.z = -self.angular_speed
        return msg

    def majority_vote(self):
        if len(self.vote_window) < self.window_size:
            return None
        return max(set(self.vote_window), key=self.vote_window.count)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to grab frame')
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw skeleton on black canvas (matches training data)
            canvas = self.draw_landmarks_on_black(hand_landmarks, w, h)

            # Preprocess and predict
            img = self.preprocess(canvas)
            gesture, confidence = self.predict(img)

            # Show debug: model input
            debug = (img.reshape(64, 64) * 255).astype(np.uint8)
            cv2.imshow('Model Input (64x64)', cv2.resize(debug, (256, 256)))

            if confidence >= self.conf_thresh:
                self.vote_window.append(gesture)
            else:
                self.vote_window.clear()

            final_gesture = self.majority_vote()

            # Overlay on live frame
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            label = f'{gesture} ({confidence:.2f})'
            color = (0, 255, 0) if confidence >= self.conf_thresh else (0, 0, 255)
            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            if final_gesture:
                cv2.putText(frame, f'CMD: {final_gesture}', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                twist = self.gesture_to_twist(final_gesture)
                self.publisher.publish(twist)
                self.get_logger().info(
                    f'Published: {final_gesture} | lin={twist.linear.x:.1f} ang={twist.angular.z:.1f}',
                    throttle_duration_sec=0.5
                )
        else:
            cv2.putText(frame, 'No hand detected', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            self.vote_window.clear()

        cv2.imshow('Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
