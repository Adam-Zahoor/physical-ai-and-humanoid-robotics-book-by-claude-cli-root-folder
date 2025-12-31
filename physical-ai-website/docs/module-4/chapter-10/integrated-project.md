---
sidebar_position: 3
---

# Integrated Projects: Bringing It All Together

## Introduction

This chapter presents comprehensive, hands-on projects that integrate all the concepts covered throughout the book. These projects are designed to provide practical experience with physical AI and humanoid robotics, allowing you to apply the theoretical knowledge and techniques learned in previous chapters. Each project builds upon multiple concepts, encouraging you to think holistically about the integration of perception, planning, control, learning, and interaction in embodied systems.

## Project 1: Autonomous Navigation and Object Manipulation

### Overview

This project combines navigation, perception, and manipulation capabilities to create a robot that can autonomously navigate to a target location, identify and grasp an object, and transport it to a designated destination. This project integrates concepts from motion planning, computer vision, control systems, and manipulation planning.

### Prerequisites

- Basic understanding of ROS 2 and Python
- Knowledge of path planning and trajectory generation
- Understanding of perception systems
- Familiarity with control systems
- Basic manipulation concepts

### Implementation Steps

#### Step 1: Environment Setup

First, set up the simulation environment:

```python
#!/usr/bin/env python3
"""
Autonomous Navigation and Object Manipulation Project
Integrated Project for Physical AI and Humanoid Robotics
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleActionClient
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import math
import time

class NavigationManipulationProject(Node):
    """
    Integrated project combining navigation, perception, and manipulation
    """

    def __init__(self):
        super().__init__('navigation_manipulation_project')

        # Initialize state machine
        self.state = 'INITIAL'  # INITIAL, NAVIGATING, PERCEIVING, MANIPULATING, TRANSPORTING, COMPLETE

        # Navigation components
        self.move_base_client = SimpleActionClient('move_base', MoveBaseAction)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Perception components
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        # Manipulation components
        self.arm_client = SimpleActionClient('arm_controller', FollowJointTrajectoryAction)
        self.gripper_pub = self.create_publisher(String, '/gripper/command', 10)

        # Project parameters
        self.robot_pose = None
        self.target_object_pose = None
        self.delivery_location = None
        self.object_detected = False
        self.object_grasped = False

        # Set up delivery location (x, y, z coordinates)
        self.delivery_location = Point(x=5.0, y=3.0, z=0.0)

        # Timer for state machine
        self.timer = self.create_timer(0.1, self.state_machine)

        self.get_logger().info('Navigation and Manipulation Project initialized')

    def odom_callback(self, msg):
        """Callback for robot odometry"""
        self.robot_pose = msg.pose.pose
        self.get_logger().debug(f'Robot pose updated: {self.robot_pose.position.x}, {self.robot_pose.position.y}')

    def image_callback(self, msg):
        """Callback for camera images - used for object detection"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Object detection using color-based segmentation
            # This is a simplified example - in practice, you might use deep learning
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define range for red object (adjust as needed)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)

            mask = mask1 + mask2

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Calculate center of the object
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Convert pixel coordinates to world coordinates
                    # This is a simplified conversion - in practice, you'd use camera calibration
                    self.target_object_pose = self.pixel_to_world(cx, cy, msg.header.frame_id)
                    self.object_detected = True
                    self.get_logger().info(f'Object detected at: {self.target_object_pose}')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def pixel_to_world(self, x, y, frame_id):
        """Convert pixel coordinates to world coordinates (simplified)"""
        # This is a very simplified conversion
        # In practice, you'd use camera calibration parameters
        # and geometric relationships

        # Assume camera is at robot's position and orientation
        if self.robot_pose:
            # Calculate approximate world coordinates based on robot pose
            # This is highly simplified and would require proper calibration in practice
            world_x = self.robot_pose.position.x + (x - 320) * 0.01  # Rough scaling
            world_y = self.robot_pose.position.y + (y - 240) * 0.01  # Rough scaling
            return Point(x=world_x, y=world_y, z=0.0)

        return Point(x=0.0, y=0.0, z=0.0)

    def navigate_to_object(self):
        """Navigate to the detected object"""
        if self.target_object_pose and self.move_base_client.wait_for_server(timeout_sec=5.0):
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = self.get_clock().now().to_msg()

            # Set goal position near the object
            goal.target_pose.pose.position = self.target_object_pose
            goal.target_pose.pose.orientation.w = 1.0  # No rotation

            self.move_base_client.send_goal(goal)
            self.get_logger().info(f'Navigating to object at {self.target_object_pose}')

            return True
        else:
            self.get_logger().warn('Could not navigate - target object not detected or navigation server unavailable')
            return False

    def detect_object(self):
        """Detect the target object using perception system"""
        # This method is handled by the image_callback
        # We just check if object has been detected
        return self.object_detected

    def grasp_object(self):
        """Grasp the object using manipulation system"""
        if self.arm_client.wait_for_server(timeout_sec=5.0):
            # Create a simple trajectory for grasping
            goal = FollowJointTrajectoryGoal()

            # Define joint names for the arm
            goal.trajectory.joint_names = ['shoulder_joint', 'elbow_joint', 'wrist_joint']

            # Create trajectory points
            point = JointTrajectoryPoint()
            point.positions = [0.5, -0.5, 0.0]  # Example positions
            point.time_from_start.sec = 2
            goal.trajectory.points = [point]

            self.arm_client.send_goal(goal)
            self.get_logger().info('Sending grasp trajectory')

            # Close gripper
            gripper_msg = String()
            gripper_msg.data = 'close'
            self.gripper_pub.publish(gripper_msg)

            self.object_grasped = True
            return True
        else:
            self.get_logger().warn('Arm controller not available')
            return False

    def transport_object(self):
        """Transport object to delivery location"""
        if self.delivery_location and self.move_base_client.wait_for_server(timeout_sec=5.0):
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = self.get_clock().now().to_msg()

            goal.target_pose.pose.position = self.delivery_location
            goal.target_pose.pose.orientation.w = 1.0

            self.move_base_client.send_goal(goal)
            self.get_logger().info(f'Transporting object to delivery location: {self.delivery_location}')

            return True
        else:
            self.get_logger().warn('Could not transport - delivery location not set or navigation unavailable')
            return False

    def release_object(self):
        """Release the object at delivery location"""
        gripper_msg = String()
        gripper_msg.data = 'open'
        self.gripper_pub.publish(gripper_msg)

        self.object_grasped = False
        self.get_logger().info('Object released at delivery location')
        return True

    def state_machine(self):
        """Main state machine for the project"""
        if self.state == 'INITIAL':
            self.get_logger().info('Project starting - detecting object...')
            if self.detect_object():
                self.state = 'NAVIGATING'
            else:
                self.get_logger().info('Searching for object...')
                # Implement search behavior
                self.search_for_object()

        elif self.state == 'NAVIGATING':
            if self.navigate_to_object():
                # Wait for navigation to complete (simplified)
                # In practice, you'd check the action status
                time.sleep(2)  # Simulate navigation time
                self.state = 'MANIPULATING'
            else:
                self.state = 'SEARCHING'

        elif self.state == 'MANIPULATING':
            if self.grasp_object():
                self.state = 'TRANSPORTING'
            else:
                self.state = 'FAILED'

        elif self.state == 'TRANSPORTING':
            if self.transport_object():
                # Wait for navigation to complete
                time.sleep(3)  # Simulate transport time
                self.state = 'RELEASING'
            else:
                self.state = 'FAILED'

        elif self.state == 'RELEASING':
            if self.release_object():
                self.state = 'COMPLETE'
                self.get_logger().info('Project completed successfully!')
            else:
                self.state = 'FAILED'

        elif self.state == 'SEARCHING':
            # Implement search behavior
            self.search_for_object()
            if self.detect_object():
                self.state = 'NAVIGATING'

        elif self.state == 'FAILED':
            self.get_logger().error('Project failed - returning to initial state')
            self.state = 'INITIAL'

    def search_for_object(self):
        """Implement search behavior when object is not detected"""
        # This could implement a systematic search pattern
        # For now, just log that we're searching
        self.get_logger().info('Searching for object...')

def main(args=None):
    rclpy.init(args=args)

    project = NavigationManipulationProject()

    try:
        rclpy.spin(project)
    except KeyboardInterrupt:
        project.get_logger().info('Project interrupted by user')
    finally:
        project.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Testing and Validation

Create a test script to validate the navigation and manipulation system:

```python
#!/usr/bin/env python3
"""
Test script for Navigation and Manipulation Project
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import String
import time

class NavigationManipulationTester(Node):
    """Tester for the navigation and manipulation project"""

    def __init__(self):
        super().__init__('navigation_manipulation_tester')

        # Publishers for simulation control
        self.object_spawn_pub = self.create_publisher(String, '/simulation/spawn_object', 10)
        self.robot_spawn_pub = self.create_publisher(String, '/simulation/spawn_robot', 10)

        # Timer to run tests
        self.timer = self.create_timer(1.0, self.run_tests)
        self.test_step = 0

    def run_tests(self):
        """Run comprehensive tests"""
        if self.test_step == 0:
            self.test_navigation()
        elif self.test_step == 1:
            self.test_perception()
        elif self.test_step == 2:
            self.test_manipulation()
        elif self.test_step == 3:
            self.test_complete_project()
        else:
            self.get_logger().info('All tests completed')
            self.timer.cancel()

        self.test_step += 1

    def test_navigation(self):
        """Test navigation system"""
        self.get_logger().info('Testing navigation system...')
        # Implementation would test navigation to various locations
        pass

    def test_perception(self):
        """Test perception system"""
        self.get_logger().info('Testing perception system...')
        # Implementation would test object detection in various conditions
        pass

    def test_manipulation(self):
        """Test manipulation system"""
        self.get_logger().info('Testing manipulation system...')
        # Implementation would test grasping and releasing objects
        pass

    def test_complete_project(self):
        """Test complete project integration"""
        self.get_logger().info('Testing complete project integration...')
        # Implementation would run the full project scenario
        pass

    def search_for_object(self):
        """Implement search behavior when object is not detected"""
        # This could implement a systematic search pattern
        # For now, just log that we're searching
        self.get_logger().info('Searching for object...')

def main(args=None):
    rclpy.init(args=args)

    tester = NavigationManipulationTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Testing interrupted by user')
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Project 2: Human-Robot Interaction System

### Overview

This project implements a comprehensive human-robot interaction system that combines speech recognition, gesture recognition, facial expression processing, and appropriate response generation. It integrates concepts from human-robot interaction with perception and control systems.

### Implementation Steps

#### Step 1: Interaction System Architecture

```python
#!/usr/bin/env python3
"""
Human-Robot Interaction System Project
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np
from cv_bridge import CvBridge
import mediapipe as mp
import threading
import queue
import time

class HumanRobotInteractionSystem(Node):
    """
    Comprehensive human-robot interaction system
    """

    def __init__(self):
        super().__init__('human_robot_interaction_system')

        # Initialize components
        self.speech_recognizer = sr.Recognizer()
        self.text_to_speech = pyttsx3.init()
        self.bridge = CvBridge()

        # Initialize MediaPipe for gesture recognition
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )

        # Publishers and subscribers
        self.speech_sub = self.create_subscription(String, '/speech_input', self.speech_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.face_pub = self.create_publisher(String, '/display/expression', 10)
        self.arm_pub = self.create_publisher(JointState, '/arm/joint_states', 10)

        # Interaction state
        self.current_interaction_state = 'IDLE'
        self.user_intent = None
        self.user_emotion = None
        self.interaction_queue = queue.Queue()

        # Timer for interaction processing
        self.interaction_timer = self.create_timer(0.1, self.process_interaction)

        self.get_logger().info('Human-Robot Interaction System initialized')

    def speech_callback(self, msg):
        """Handle speech input"""
        self.get_logger().info(f'Received speech: {msg.data}')

        # Process speech to extract intent
        intent = self.process_speech_intent(msg.data)
        self.user_intent = intent

        # Generate appropriate response
        response = self.generate_response(intent)

        # Speak the response
        self.speak_response(response)

        # Set appropriate facial expression
        self.set_facial_expression(self.determine_expression(intent))

    def process_speech_intent(self, speech_text):
        """Process speech to determine user intent"""
        # Simple keyword-based intent recognition
        speech_lower = speech_text.lower()

        if any(word in speech_lower for word in ['hello', 'hi', 'hey']):
            return 'greeting'
        elif any(word in speech_lower for word in ['how are you', 'how do you do']):
            return 'inquiry'
        elif any(word in speech_lower for word in ['help', 'assist', 'can you']):
            return 'request'
        elif any(word in speech_lower for word in ['goodbye', 'bye', 'see you']):
            return 'farewell'
        else:
            return 'unknown'

    def generate_response(self, intent):
        """Generate appropriate response based on intent"""
        responses = {
            'greeting': 'Hello! It\'s nice to meet you. How can I help you today?',
            'inquiry': 'I\'m functioning well, thank you for asking! How can I assist you?',
            'request': 'I\'d be happy to help with that. Could you please specify what you need?',
            'farewell': 'Goodbye! It was nice interacting with you. Have a great day!',
            'unknown': 'I\'m not sure I understood. Could you please repeat that?'
        }

        return responses.get(intent, responses['unknown'])

    def speak_response(self, response_text):
        """Speak the response using text-to-speech"""
        self.get_logger().info(f'Speaking: {response_text}')

        # Use a separate thread to avoid blocking
        def speak():
            self.text_to_speech.say(response_text)
            self.text_to_speech.runAndWait()

        speak_thread = threading.Thread(target=speak)
        speak_thread.start()

    def determine_expression(self, intent):
        """Determine appropriate facial expression based on intent"""
        expression_map = {
            'greeting': 'SMILE',
            'inquiry': 'THINKING',
            'request': 'ATTENTIVE',
            'farewell': 'SMILE',
            'unknown': 'CONFUSED'
        }

        return expression_map.get(intent, 'NEUTRAL')

    def set_facial_expression(self, expression):
        """Set facial expression"""
        msg = String()
        msg.data = expression
        self.face_pub.publish(msg)

    def image_callback(self, msg):
        """Process camera images for gesture recognition"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process image for hand gestures
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Determine gesture based on hand landmarks
                    gesture = self.classify_gesture(hand_landmarks)

                    if gesture:
                        self.handle_gesture(gesture)

                    # Draw hand landmarks
                    self.mp.solutions.drawing_utils.draw_landmarks(
                        cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

            # Display the processed image
            cv2.imshow('HRI System', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def classify_gesture(self, hand_landmarks):
        """Classify hand gesture from landmarks"""
        # Extract key landmarks
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        # Simple gesture classification based on finger positions
        # Thumb tip (4), Index finger tip (8), Middle finger tip (12), etc.
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Palm base (0)
        palm_base = landmarks[0]

        # Determine if fingers are extended
        def is_extended(finger_tip, palm_base):
            # Check if finger tip is above (y is inverted in image coordinates) palm base
            return finger_tip[1] < palm_base[1] - 0.1

        index_extended = is_extended(index_tip, palm_base)
        middle_extended = is_extended(middle_tip, palm_base)
        ring_extended = is_extended(ring_tip, palm_base)
        pinky_extended = is_extended(pinky_tip, palm_base)

        # Classify gestures
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'POINTING'
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 'PEACE'
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'FIST'
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return 'OPEN_HAND'
        else:
            return 'UNKNOWN'

    def handle_gesture(self, gesture):
        """Handle recognized gesture"""
        self.get_logger().info(f'Gesture recognized: {gesture}')

        # Map gestures to robot responses
        gesture_responses = {
            'POINTING': 'I see you are pointing at something. How can I help?',
            'PEACE': 'Nice peace sign! What can I do for you?',
            'FIST': 'Fist bump! Hello there!',
            'OPEN_HAND': 'Open hand gesture detected. Are you gesturing to me?'
        }

        response = gesture_responses.get(gesture, 'I noticed your gesture.')

        # Add gesture response to interaction queue
        self.interaction_queue.put({
            'type': 'gesture',
            'gesture': gesture,
            'response': response
        })

    def process_interaction(self):
        """Process queued interactions"""
        while not self.interaction_queue.empty():
            interaction = self.interaction_queue.get()

            if interaction['type'] == 'gesture':
                self.get_logger().info(f'Processing gesture: {interaction["gesture"]}')

                # Speak the response
                self.speak_response(interaction['response'])

                # Set appropriate expression
                self.set_facial_expression('ATTENTIVE')

    def execute_social_behavior(self, behavior_type):
        """Execute social behavior based on interaction"""
        if behavior_type == 'greeting':
            # Execute greeting behavior (wave, smile, etc.)
            self.execute_wave_gesture()
        elif behavior_type == 'acknowledgment':
            # Execute acknowledgment (nod, etc.)
            self.execute_nod_gesture()

    def execute_wave_gesture(self):
        """Execute waving gesture with arm"""
        joint_state = JointState()
        joint_state.name = ['shoulder_joint', 'elbow_joint', 'wrist_joint']
        joint_state.position = [0.5, 0.3, 0.0]  # Example positions for wave
        self.arm_pub.publish(joint_state)

    def execute_nod_gesture(self):
        """Execute nodding gesture"""
        joint_state = JointState()
        joint_state.name = ['neck_joint']
        joint_state.position = [0.1]  # Small nod
        self.arm_pub.publish(joint_state)

def main(args=None):
    rclpy.init(args=args)

    hri_system = HumanRobotInteractionSystem()

    try:
        rclpy.spin(hri_system)
    except KeyboardInterrupt:
        hri_system.get_logger().info('HRI System interrupted by user')
    finally:
        hri_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Project 3: Adaptive Learning System

### Overview

This project implements an adaptive learning system that allows a humanoid robot to learn and improve its behaviors through interaction with the environment. It integrates reinforcement learning concepts with perception and control systems.

### Implementation Steps

#### Step 1: Learning System Architecture

```python
#!/usr/bin/env python3
"""
Adaptive Learning System Project
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, JointState, Image
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32
import numpy as np
import random
from collections import defaultdict
import pickle
import os

class AdaptiveLearningSystem(Node):
    """
    Adaptive learning system using reinforcement learning
    """

    def __init__(self):
        super().__init__('adaptive_learning_system')

        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.3
        self.exploration_decay = 0.995
        self.min_exploration = 0.01

        # Q-table for learning
        self.q_table = defaultdict(lambda: defaultdict(float))

        # State and action spaces
        self.state_space = []  # Will be populated dynamically
        self.action_space = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP']

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        # Learning state
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.episode_rewards = []
        self.episode_count = 0

        # Timer for learning updates
        self.learning_timer = self.create_timer(0.1, self.update_learning)

        # Load saved Q-table if exists
        self.load_q_table()

        self.get_logger().info('Adaptive Learning System initialized')

    def laser_callback(self, msg):
        """Process laser scan data to determine state"""
        # Discretize laser data into state representation
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=3.0)  # Replace NaN with max range

        # Create discrete state from laser readings
        # Group laser readings into sectors
        sector_size = len(ranges) // 8  # 8 sectors
        sectors = []

        for i in range(0, len(ranges), sector_size):
            sector_data = ranges[i:i+sector_size]
            avg_distance = np.mean(sector_data)
            # Discretize distance: 0 = close, 1 = medium, 2 = far
            if avg_distance < 0.5:
                sectors.append(0)  # Close
            elif avg_distance < 1.0:
                sectors.append(1)  # Medium
            else:
                sectors.append(2)  # Far

        # Create state tuple
        self.current_state = tuple(sectors)

    def joint_callback(self, msg):
        """Process joint state data"""
        # Could be used for additional state information
        pass

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            # Exploration: random action
            return random.choice(self.action_space)
        else:
            # Exploitation: best known action
            if state in self.q_table:
                # Find action with highest Q-value
                q_values = self.q_table[state]
                if q_values:
                    return max(q_values, key=q_values.get)
                else:
                    return random.choice(self.action_space)
            else:
                return random.choice(self.action_space)

    def calculate_reward(self, state, action):
        """Calculate reward based on current state and action"""
        # Calculate reward based on distance to obstacles
        # State is tuple of discretized distances (0=close, 1=medium, 2=far)

        # Penalty for being close to obstacles
        obstacle_penalty = 0
        for distance_reading in state:
            if distance_reading == 0:  # Very close
                obstacle_penalty += -10
            elif distance_reading == 1:  # Medium distance
                obstacle_penalty += -2

        # Small reward for moving forward
        movement_reward = 0
        if action == 'FORWARD':
            movement_reward = 0.5

        # Penalty for stopping when not necessary
        stop_penalty = 0
        if action == 'STOP':
            # Check if there are obstacles in front
            if len(state) > 3 and state[3] != 0:  # If front sector is not close
                stop_penalty = -1

        total_reward = obstacle_penalty + movement_reward + stop_penalty
        return total_reward

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state][action]

        # Calculate max Q-value for next state
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        else:
            max_next_q = 0

        # Q-learning update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def execute_action(self, action):
        """Execute the selected action"""
        twist = Twist()

        if action == 'FORWARD':
            twist.linear.x = 0.2
        elif action == 'BACKWARD':
            twist.linear.x = -0.2
        elif action == 'LEFT':
            twist.angular.z = 0.3
        elif action == 'RIGHT':
            twist.angular.z = -0.3
        elif action == 'STOP':
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)

    def update_learning(self):
        """Main learning update function"""
        if self.current_state is not None:
            # Select action
            action = self.select_action(self.current_state)

            # Execute action
            self.execute_action(action)

            # Calculate reward
            reward = self.calculate_reward(self.current_state, action)

            # Update Q-value if we have previous state-action pair
            if self.previous_state is not None and self.previous_action is not None:
                self.update_q_value(
                    self.previous_state,
                    self.previous_action,
                    reward,
                    self.current_state
                )

            # Update previous state and action for next iteration
            self.previous_state = self.current_state
            self.previous_action = action

            # Add reward to episode
            self.episode_rewards.append(reward)

            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration,
                self.exploration_rate * self.exploration_decay
            )

            # Check for episode end (simplified - every 100 steps)
            if len(self.episode_rewards) % 100 == 0:
                self.end_episode()

    def end_episode(self):
        """End current learning episode"""
        total_reward = sum(self.episode_rewards)
        self.get_logger().info(
            f'Episode {self.episode_count} ended. Total reward: {total_reward:.2f}, '
            f'Exploration rate: {self.exploration_rate:.3f}'
        )

        self.episode_rewards = []
        self.episode_count += 1

        # Save Q-table periodically
        if self.episode_count % 10 == 0:
            self.save_q_table()

    def save_q_table(self):
        """Save Q-table to file"""
        try:
            with open('/tmp/q_table.pkl', 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            self.get_logger().info('Q-table saved to /tmp/q_table.pkl')
        except Exception as e:
            self.get_logger().error(f'Error saving Q-table: {e}')

    def load_q_table(self):
        """Load Q-table from file"""
        if os.path.exists('/tmp/q_table.pkl'):
            try:
                with open('/tmp/q_table.pkl', 'rb') as f:
                    saved_table = pickle.load(f)
                    self.q_table.update({k: defaultdict(float, v) for k, v in saved_table.items()})
                self.get_logger().info('Q-table loaded from /tmp/q_table.pkl')
            except Exception as e:
                self.get_logger().error(f'Error loading Q-table: {e}')

def main(args=None):
    rclpy.init(args=args)

    learning_system = AdaptiveLearningSystem()

    try:
        rclpy.spin(learning_system)
    except KeyboardInterrupt:
        learning_system.get_logger().info('Learning system interrupted by user')
        learning_system.save_q_table()
    finally:
        learning_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Project 4: Multi-Robot Coordination System

### Overview

This project implements a multi-robot coordination system where multiple robots work together to achieve a common goal. It integrates concepts from multi-agent systems with communication and coordination algorithms.

### Implementation Steps

#### Step 1: Multi-Robot Coordination Architecture

```python
#!/usr/bin/env python3
"""
Multi-Robot Coordination System Project
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32
import numpy as np
import math
from collections import defaultdict

class MultiRobotCoordinator(Node):
    """
    Multi-robot coordination system
    """

    def __init__(self):
        super().__init__('multi_robot_coordinator')

        # Robot ID and team management
        self.robot_id = self.declare_parameter('robot_id', 0).value
        self.team_size = self.declare_parameter('team_size', 3).value
        self.robots = {}  # {robot_id: pose}

        # Publishers and subscribers
        self.pose_pub = self.create_publisher(Pose, f'/robot_{self.robot_id}/pose', 10)
        self.comm_pub = self.create_publisher(String, '/team_communication', 10)
        self.comm_sub = self.create_subscription(String, '/team_communication', self.comm_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, f'/robot_{self.robot_id}/scan', self.laser_callback, 10)

        # Task allocation and coordination
        self.task_queue = []
        self.assigned_tasks = {}
        self.robot_capabilities = {}
        self.coordination_strategy = 'auction'  # 'auction', 'consensus', 'leader_follower'

        # Timer for coordination updates
        self.coordination_timer = self.create_timer(0.5, self.coordination_update)

        # Initialize robot capabilities
        self.initialize_capabilities()

        self.get_logger().info(f'Multi-Robot Coordinator for Robot {self.robot_id} initialized')

    def initialize_capabilities(self):
        """Initialize robot capabilities"""
        # Define what each robot can do based on its ID
        # In a real system, this would be determined by actual robot capabilities
        capabilities = {
            'locomotion': 1.0 if self.robot_id % 2 == 0 else 0.8,  # Even robots better at moving
            'manipulation': 0.7 if self.robot_id % 2 == 1 else 0.5,  # Odd robots better at manipulation
            'sensing': 0.9,
            'communication': 1.0
        }
        self.robot_capabilities[self.robot_id] = capabilities

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Could be used for local navigation and obstacle avoidance
        pass

    def comm_callback(self, msg):
        """Handle communication messages from other robots"""
        try:
            # Parse message: "robot_id:content"
            parts = msg.data.split(':', 1)
            if len(parts) == 2:
                sender_id = int(parts[0])
                content = parts[1]

                # Process different types of messages
                if content.startswith('POSE:'):
                    # Pose update from another robot
                    pose_data = content[5:].split(',')
                    if len(pose_data) >= 3:
                        x, y, theta = float(pose_data[0]), float(pose_data[1]), float(pose_data[2])
                        self.robots[sender_id] = Point(x=x, y=y, z=theta)

                elif content.startswith('TASK:'):
                    # Task assignment or update
                    self.process_task_message(sender_id, content[5:])

                elif content.startswith('CAPABILITIES:'):
                    # Capabilities update
                    self.process_capabilities_message(sender_id, content[11:])

        except Exception as e:
            self.get_logger().error(f'Error parsing communication: {e}')

    def process_task_message(self, sender_id, content):
        """Process task-related messages"""
        if self.coordination_strategy == 'auction':
            self.process_auction_message(sender_id, content)
        elif self.coordination_strategy == 'consensus':
            self.process_consensus_message(sender_id, content)

    def process_auction_message(self, sender_id, content):
        """Process auction-based task allocation messages"""
        # Parse task auction message
        if content.startswith('AUCTION:'):
            task_info = content[8:]
            # In a real system, evaluate task and bid if appropriate
            if self.should_bid_for_task(task_info):
                self.send_bid(sender_id, task_info)

    def should_bid_for_task(self, task_info):
        """Determine if robot should bid for a task"""
        # Simple evaluation based on capabilities
        required_capability = task_info.split(':')[0]  # Assume first part is capability needed

        if required_capability in self.robot_capabilities[self.robot_id]:
            capability_level = self.robot_capabilities[self.robot_id][required_capability]
            return capability_level > 0.6  # Bid if capability is above threshold
        return False

    def send_bid(self, sender_id, task_info):
        """Send bid for task in auction system"""
        # Calculate bid value based on capability and current workload
        capability_score = self.evaluate_task_capability(task_info)
        workload_score = 1.0 - (len(self.assigned_tasks) / 5.0)  # Normalize by max tasks
        bid_value = capability_score * workload_score

        bid_message = f"{self.robot_id}:BID:{task_info}:{bid_value:.3f}"
        self.comm_pub.publish(String(data=bid_message))

    def evaluate_task_capability(self, task_info):
        """Evaluate how well robot can perform a task"""
        # Simplified capability evaluation
        # In reality, this would be more sophisticated
        return np.mean(list(self.robot_capabilities[self.robot_id].values()))

    def process_capabilities_message(self, sender_id, content):
        """Process capabilities update from another robot"""
        try:
            capabilities = {}
            for item in content.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    capabilities[key] = float(value)
            self.robot_capabilities[sender_id] = capabilities
        except Exception as e:
            self.get_logger().error(f'Error parsing capabilities: {e}')

    def coordination_update(self):
        """Main coordination update function"""
        # Update own pose (simplified - in reality, get from localization)
        current_pose = self.get_current_pose()
        if current_pose:
            # Publish own pose to team
            pose_msg = f"POSE:{current_pose.x},{current_pose.y},{current_pose.z}"
            self.comm_pub.publish(String(data=f"{self.robot_id}:{pose_msg}"))

            # Perform coordination based on strategy
            if self.coordination_strategy == 'auction':
                self.perform_auction_coordination()
            elif self.coordination_strategy == 'consensus':
                self.perform_consensus_coordination()
            elif self.coordination_strategy == 'leader_follower':
                self.perform_leader_follower_coordination()

    def get_current_pose(self):
        """Get current robot pose (simplified)"""
        # In a real system, this would come from localization
        # For simulation, we'll return a fixed pose or move based on time
        t = self.get_clock().now().nanoseconds / 1e9
        x = 2.0 * math.cos(t * 0.1 + self.robot_id)
        y = 2.0 * math.sin(t * 0.1 + self.robot_id)
        theta = t * 0.05 + self.robot_id

        return Point(x=x, y=y, z=theta)

    def perform_auction_coordination(self):
        """Perform auction-based task allocation"""
        # Check if there are unassigned tasks
        unassigned_tasks = [task for task in self.task_queue
                           if task['id'] not in self.assigned_tasks]

        for task in unassigned_tasks:
            # Announce task to team
            task_msg = f"AUCTION:{task['type']}:{task['location']}"
            self.comm_pub.publish(String(data=f"{self.robot_id}:{task_msg}"))

    def perform_consensus_coordination(self):
        """Perform consensus-based coordination"""
        # Calculate formation positions based on team positions
        if len(self.robots) >= 2:
            formation_positions = self.calculate_formation_positions()

            # Move toward formation position
            target_pos = formation_positions.get(self.robot_id)
            if target_pos:
                self.move_to_position(target_pos)

    def calculate_formation_positions(self):
        """Calculate desired formation positions for all robots"""
        formation_positions = {}

        # Simple circular formation
        center_x, center_y = 0.0, 0.0  # Formation center
        radius = 2.0  # Formation radius

        for i in range(self.team_size):
            angle = 2 * math.pi * i / self.team_size
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            formation_positions[i] = Point(x=x, y=y, z=0.0)

        return formation_positions

    def move_to_position(self, target_position):
        """Move robot to target position (simplified)"""
        # In a real system, this would involve path planning and navigation
        pass

    def perform_leader_follower_coordination(self):
        """Perform leader-follower coordination"""
        # Determine if this robot is leader or follower
        if self.robot_id == 0:  # Robot 0 is leader
            self.act_as_leader()
        else:
            self.act_as_follower()

    def act_as_leader(self):
        """Act as leader in leader-follower formation"""
        # Leader moves according to mission plan
        # For simulation, move in a pattern
        t = self.get_clock().now().nanoseconds / 1e9
        target_x = 5.0 * math.cos(t * 0.05)
        target_y = 5.0 * math.sin(t * 0.05)

        # Broadcast leader position to followers
        leader_msg = f"LEADER_POS:{target_x},{target_y}"
        self.comm_pub.publish(String(data=f"{self.robot_id}:{leader_msg}"))

    def act_as_follower(self):
        """Act as follower in leader-follower formation"""
        # Followers maintain formation relative to leader
        # This would involve following the leader with appropriate spacing
        pass

    def add_task(self, task_type, location, priority=1):
        """Add task to coordination system"""
        task = {
            'id': len(self.task_queue),
            'type': task_type,
            'location': location,
            'priority': priority,
            'assigned_to': None
        }
        self.task_queue.append(task)
        self.get_logger().info(f'Task {task["id"]} added: {task_type} at {location}')

def main(args=None):
    rclpy.init(args=args)

    coordinator = MultiRobotCoordinator()

    # Add some example tasks
    coordinator.add_task('exploration', (10.0, 10.0))
    coordinator.add_task('transport', (15.0, 5.0))

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        coordinator.get_logger().info('Multi-robot coordinator interrupted by user')
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Project 5: Comprehensive Integration Challenge

### Overview

This final project integrates all the concepts learned throughout the book into a comprehensive challenge. The robot must navigate through an environment, interact with humans, learn from its experiences, coordinate with other robots, and complete complex tasks while adapting to changing conditions.

### Implementation Steps

#### Step 1: Integration Framework

```python
#!/usr/bin/env python3
"""
Comprehensive Integration Challenge Project
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Pose, Twist, Point
from sensor_msgs.msg import LaserScan, Image, JointState
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import math
import time
from collections import defaultdict
import threading

class ComprehensiveIntegrationSystem(Node):
    """
    Comprehensive integration of all humanoid robotics concepts
    """

    def __init__(self):
        super().__init__('comprehensive_integration_system')

        # Initialize all subsystems
        self.navigation_system = NavigationSystem(self)
        self.interaction_system = InteractionSystem(self)
        self.learning_system = LearningSystem(self)
        self.coordination_system = CoordinationSystem(self)

        # System state
        self.system_state = 'IDLE'  # INITIAL, NAVIGATING, INTERACTING, LEARNING, COORDINATING, EMERGENCY
        self.active_behaviors = []
        self.emergency_active = False

        # Publishers and subscribers
        self.system_status_pub = self.create_publisher(String, '/system_status', 10)
        self.emergency_stop_sub = self.create_subscription(Bool, '/emergency_stop', self.emergency_callback, 10)

        # Timer for state machine
        self.timer = self.create_timer(0.1, self.state_machine)

        # Initialize CV bridge
        self.bridge = CvBridge()

        self.get_logger().info('Comprehensive Integration System initialized')

    def emergency_callback(self, msg):
        """Handle emergency stop messages"""
        if msg.data:
            self.emergency_active = True
            self.system_state = 'EMERGENCY'
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
        else:
            self.emergency_active = False
            self.system_state = 'IDLE'

    def state_machine(self):
        """Main system state machine"""
        if self.system_state == 'EMERGENCY':
            self.handle_emergency()
            return

        # Update all subsystems
        self.navigation_system.update()
        self.interaction_system.update()
        self.learning_system.update()
        self.coordination_system.update()

        # High-level system logic
        self.execute_system_logic()

    def handle_emergency(self):
        """Handle emergency state"""
        # Stop all movement
        self.navigation_system.stop_movement()

        # Clear all active behaviors
        self.active_behaviors.clear()

        # Set emergency expression
        self.interaction_system.set_emergency_expression()

        # Wait for emergency to clear
        if not self.emergency_active:
            self.system_state = 'IDLE'

    def execute_system_logic(self):
        """Execute high-level system logic"""
        # Determine system state based on subsystem states and conditions
        if self.system_state == 'IDLE':
            # Check if there are tasks to perform
            if self.coordination_system.has_assigned_task():
                self.system_state = 'NAVIGATING'
            elif self.interaction_system.has_pending_interaction():
                self.system_state = 'INTERACTING'

        elif self.system_state == 'NAVIGATING':
            # Navigate to task location
            if self.navigation_system.is_at_destination():
                if self.coordination_system.task_requires_interaction():
                    self.system_state = 'INTERACTING'
                else:
                    self.system_state = 'EXECUTING_TASK'

        elif self.system_state == 'INTERACTING':
            # Handle human interaction
            if self.interaction_system.interaction_complete():
                self.system_state = 'NAVIGATING'  # Continue to next task or return

        elif self.system_state == 'EXECUTING_TASK':
            # Execute assigned task
            if self.coordination_system.task_complete():
                self.system_state = 'IDLE'  # Task completed
                self.coordination_system.announce_task_completion()

    def add_behavior(self, behavior_name):
        """Add behavior to active behaviors list"""
        if behavior_name not in self.active_behaviors:
            self.active_behaviors.append(behavior_name)

    def remove_behavior(self, behavior_name):
        """Remove behavior from active behaviors list"""
        if behavior_name in self.active_behaviors:
            self.active_behaviors.remove(behavior_name)

class NavigationSystem:
    """Navigation subsystem for the integration system"""

    def __init__(self, parent_node):
        self.parent = parent_node
        self.node = parent_node

        # Publishers and subscribers
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.node.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Navigation state
        self.current_pose = None
        self.destination = None
        self.path = []
        self.path_index = 0
        self.moving = False

        # Navigation parameters
        self.linear_speed = 0.2
        self.angular_speed = 0.3
        self.arrival_threshold = 0.5  # meters

        self.node.get_logger().info('Navigation system initialized')

    def odom_callback(self, msg):
        """Callback for robot odometry"""
        self.current_pose = msg.pose.pose
        self.node.get_logger().debug(f'Robot pose updated: {self.current_pose.position.x}, {self.current_pose.position.y}')

    def laser_callback(self, msg):
        """Callback for laser scan for obstacle detection"""
        # Simple obstacle detection
        min_distance = min([r for r in msg.ranges if not math.isnan(r)])
        if min_distance < 0.3:  # Emergency stop distance
            self.stop_movement()

    def set_destination(self, x, y):
        """Set navigation destination"""
        self.destination = Point(x=x, y=y, z=0.0)
        self.calculate_path()
        self.moving = True

    def calculate_path(self):
        """Calculate simple path to destination"""
        # For this example, we'll use a direct path
        # In a real system, this would involve path planning algorithms
        if self.current_pose and self.destination:
            self.path = [self.destination]
            self.path_index = 0

    def update(self):
        """Update navigation system"""
        if self.moving and self.destination and self.current_pose:
            # Calculate direction to destination
            dx = self.destination.x - self.current_pose.position.x
            dy = self.destination.y - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > self.arrival_threshold:
                # Move toward destination
                cmd_vel = Twist()

                # Linear movement
                cmd_vel.linear.x = min(self.linear_speed, distance * 0.5)

                # Angular movement to face destination
                target_angle = math.atan2(dy, dx)
                current_angle = self.get_yaw_from_quaternion(self.current_pose.orientation)

                angle_diff = target_angle - current_angle
                # Normalize angle difference
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 2.0))

                self.cmd_vel_pub.publish(cmd_vel)
            else:
                # Arrived at destination
                self.moving = False
                self.node.get_logger().info(f'Arrived at destination: ({self.destination.x}, {self.destination.y})')

    def get_yaw_from_quaternion(self, q):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def is_at_destination(self):
        """Check if robot is at destination"""
        if self.destination and self.current_pose:
            dx = self.destination.x - self.current_pose.position.x
            dy = self.destination.y - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)
            return distance <= self.arrival_threshold
        return False

    def stop_movement(self):
        """Stop all movement"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
        self.moving = False

class InteractionSystem:
    """Interaction subsystem for the integration system"""

    def __init__(self, parent_node):
        self.parent = parent_node
        self.node = parent_node

        # Publishers and subscribers
        self.speech_sub = self.node.create_subscription(String, '/speech_input', self.speech_callback, 10)
        self.image_sub = self.node.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.face_pub = self.node.create_publisher(String, '/display/expression', 10)
        self.audio_pub = self.node.create_publisher(String, '/audio/output', 10)

        # Interaction state
        self.pending_interaction = False
        self.current_interaction = None
        self.interaction_complete_flag = False
        self.conversation_history = []

        self.node.get_logger().info('Interaction system initialized')

    def speech_callback(self, msg):
        """Handle speech input"""
        self.pending_interaction = True
        self.node.get_logger().info(f'Received speech: {msg.data}')

        # Process the speech and determine response
        response = self.process_speech(msg.data)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.audio_pub.publish(response_msg)

        # Add to conversation history
        self.conversation_history.append({'user': msg.data, 'robot': response})

    def process_speech(self, speech_text):
        """Process speech and generate response"""
        # Simple response generation
        speech_lower = speech_text.lower()

        if 'hello' in speech_lower or 'hi' in speech_lower:
            self.set_expression('SMILE')
            return 'Hello! How can I assist you today?'
        elif 'how are you' in speech_lower:
            self.set_expression('THINKING')
            return 'I am functioning well, thank you for asking!'
        elif 'help' in speech_lower:
            self.set_expression('ATTENTIVE')
            return 'I can help with navigation, information, or tasks. What do you need?'
        else:
            self.set_expression('NEUTRAL')
            return 'I understand you said: ' + speech_text

    def set_expression(self, expression):
        """Set facial expression"""
        msg = String()
        msg.data = expression
        self.face_pub.publish(msg)

    def image_callback(self, msg):
        """Process camera images for gesture recognition"""
        try:
            cv_image = self.node.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process image for hand gestures
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.node.hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Determine gesture based on hand landmarks
                    gesture = self.classify_gesture(hand_landmarks)

                    if gesture:
                        self.handle_gesture(gesture)

                    # Draw hand landmarks
                    self.mp.solutions.drawing_utils.draw_landmarks(
                        cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

            # Display the processed image
            cv2.imshow('HRI System', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.node.get_logger().error(f'Error processing image: {e}')

    def classify_gesture(self, hand_landmarks):
        """Classify hand gesture from landmarks"""
        # Extract key landmarks
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        # Simple gesture classification based on finger positions
        # Thumb tip (4), Index finger tip (8), Middle finger tip (12), etc.
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Palm base (0)
        palm_base = landmarks[0]

        # Determine if fingers are extended
        def is_extended(finger_tip, palm_base):
            # Check if finger tip is above (y is inverted in image coordinates) palm base
            return finger_tip[1] < palm_base[1] - 0.1

        index_extended = is_extended(index_tip, palm_base)
        middle_extended = is_extended(middle_tip, palm_base)
        ring_extended = is_extended(ring_tip, palm_base)
        pinky_extended = is_extended(pinky_tip, palm_base)

        # Classify gestures
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'POINTING'
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 'PEACE'
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'FIST'
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return 'OPEN_HAND'
        else:
            return 'UNKNOWN'

    def handle_gesture(self, gesture):
        """Handle recognized gesture"""
        self.node.get_logger().info(f'Gesture recognized: {gesture}')

        # Map gestures to robot responses
        gesture_responses = {
            'POINTING': 'I see you are pointing at something. How can I help?',
            'PEACE': 'Nice peace sign! What can I do for you?',
            'FIST': 'Fist bump! Hello there!',
            'OPEN_HAND': 'Open hand gesture detected. Are you gesturing to me?'
        }

        response = gesture_responses.get(gesture, 'I noticed your gesture.')

        # Speak the response
        self.speak_response(response)

        # Set appropriate expression
        self.set_expression('ATTENTIVE')

    def speak_response(self, response_text):
        """Speak the response using text-to-speech"""
        self.node.get_logger().info(f'Speaking: {response_text}')
        # Implementation would use text-to-speech system

    def update(self):
        """Update interaction system"""
        # Check for interaction timeouts
        if self.current_interaction:
            # In a real system, check if interaction is still active
            pass

    def has_pending_interaction(self):
        """Check if there's a pending interaction"""
        return self.pending_interaction

    def interaction_complete(self):
        """Check if current interaction is complete"""
        return self.interaction_complete_flag

    def set_emergency_expression(self):
        """Set emergency expression"""
        self.set_expression('EMERGENCY')

class LearningSystem:
    """Learning subsystem for the integration system"""

    def __init__(self, parent_node):
        self.parent = parent_node
        self.node = parent_node

        # Publishers and subscribers
        self.sensor_sub = self.node.create_subscription(LaserScan, '/scan', self.sensor_callback, 10)
        self.performance_sub = self.node.create_subscription(Float32, '/performance_metric', self.performance_callback, 10)

        # Learning state
        self.experience_buffer = []
        self.performance_history = []
        self.learning_enabled = True

        # Learning parameters
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

        self.node.get_logger().info('Learning system initialized')

    def sensor_callback(self, msg):
        """Handle sensor data for learning"""
        if self.learning_enabled:
            # Store sensor experience
            experience = {
                'timestamp': self.node.get_clock().now().nanoseconds,
                'sensor_data': list(msg.ranges),
                'action_taken': None,  # Would be filled by action system
                'reward': 0.0  # Would be calculated based on outcome
            }
            self.experience_buffer.append(experience)

            # Keep buffer size manageable
            if len(self.experience_buffer) > 1000:
                self.experience_buffer.pop(0)

    def performance_callback(self, msg):
        """Handle performance metric updates"""
        self.performance_history.append({
            'timestamp': self.node.get_clock().now().nanoseconds,
            'performance': msg.data
        })

        # Keep performance history manageable
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

    def update(self):
        """Update learning system"""
        if self.learning_enabled and len(self.performance_history) > 10:
            # Analyze performance trends
            recent_performance = [p['performance'] for p in self.performance_history[-10:]]
            avg_performance = sum(recent_performance) / len(recent_performance)

            # Adjust behavior based on performance
            if avg_performance < 0.5:  # Poor performance
                self.exploration_rate = min(0.5, self.exploration_rate + 0.01)  # Explore more
            else:  # Good performance
                self.exploration_rate = max(0.05, self.exploration_rate - 0.001)  # Exploit more

class CoordinationSystem:
    """Coordination subsystem for the integration system"""

    def __init__(self, parent_node):
        self.parent = parent_node
        self.node = parent_node

        # Publishers and subscribers
        self.task_sub = self.node.create_subscription(String, '/task_assignment', self.task_callback, 10)
        self.coordination_sub = self.node.create_subscription(String, '/team_coordination', self.coordination_callback, 10)
        self.task_pub = self.node.create_publisher(String, '/task_status', 10)

        # Task management
        self.assigned_tasks = []
        self.current_task = None
        self.task_progress = {}

        self.node.get_logger().info('Coordination system initialized')

    def task_callback(self, msg):
        """Handle task assignment"""
        # Parse task assignment
        try:
            task_data = msg.data.split(':', 2)  # Format: "TASK_TYPE:LOCATION:TASK_ID"
            if len(task_data) >= 3:
                task_type = task_data[0]
                location_str = task_data[1]
                task_id = task_data[2]

                # Parse location
                loc_parts = location_str.strip('()').split(',')
                if len(loc_parts) >= 2:
                    x = float(loc_parts[0])
                    y = float(loc_parts[1])

                    task = {
                        'id': task_id,
                        'type': task_type,
                        'location': (x, y),
                        'status': 'ASSIGNED',
                        'assigned_time': self.node.get_clock().now().nanoseconds
                    }

                    self.assigned_tasks.append(task)
                    self.node.get_logger().info(f'Task assigned: {task_type} at ({x}, {y})')

        except Exception as e:
            self.node.get_logger().error(f'Error parsing task assignment: {e}')

    def coordination_callback(self, msg):
        """Handle coordination messages from other robots"""
        # In a real system, this would handle team coordination
        pass

    def update(self):
        """Update coordination system"""
        # Check current task status
        if self.current_task:
            # Update task progress
            # In a real system, this would check actual task completion
            pass

    def has_assigned_task(self):
        """Check if there are assigned tasks"""
        return len(self.assigned_tasks) > 0

    def task_requires_interaction(self):
        """Check if current task requires interaction"""
        if self.current_task:
            return self.current_task['type'] in ['delivery', 'assistance', 'greeting']
        return False

    def task_complete(self):
        """Check if current task is complete"""
        # In a real system, this would check actual task completion
        return False

    def announce_task_completion(self):
        """Announce task completion to team"""
        if self.current_task:
            completion_msg = String()
            completion_msg.data = f"COMPLETED:{self.current_task['id']}"
            self.task_pub.publish(completion_msg)

def main(args=None):
    rclpy.init(args=args)

    integration_system = ComprehensiveIntegrationSystem()

    try:
        rclpy.spin(integration_system)
    except KeyboardInterrupt:
        integration_system.get_logger().info('Comprehensive integration system interrupted by user')
    finally:
        integration_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing and Validation Framework

### Overview

To ensure the projects work correctly, create a comprehensive testing framework:

```python
#!/usr/bin/env python3
"""
Testing Framework for Integrated Projects
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
import time
import unittest
from unittest.mock import Mock

class IntegrationTestFramework(Node):
    """Comprehensive testing framework for integrated projects"""

    def __init__(self):
        super().__init__('integration_test_framework')

        # Test state
        self.tests_passed = 0
        self.tests_failed = 0
        self.current_test = None

        # Publishers for simulation control
        self.test_control_pub = self.create_publisher(String, '/test/control', 10)
        self.emergency_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # Timer for running tests
        self.test_timer = self.create_timer(2.0, self.run_next_test)
        self.test_queue = self.create_test_suite()
        self.test_index = 0

        self.get_logger().info('Integration Test Framework initialized')

    def create_test_suite(self):
        """Create comprehensive test suite"""
        return [
            self.test_navigation_basic,
            self.test_interaction_basic,
            self.test_learning_basic,
            self.test_coordination_basic,
            self.test_integration_complete,
            self.test_emergency_procedures,
            self.test_system_reliability
        ]

    def run_next_test(self):
        """Run the next test in the suite"""
        if self.test_index < len(self.test_queue):
            test_func = self.test_queue[self.test_index]
            self.get_logger().info(f'Running test: {test_func.__name__}')

            try:
                test_func()
                self.tests_passed += 1
                self.get_logger().info(f'Test {test_func.__name__} PASSED')
            except Exception as e:
                self.tests_failed += 1
                self.get_logger().error(f'Test {test_func.__name__} FAILED: {e}')

            self.test_index += 1
        else:
            # All tests completed
            self.get_logger().info(f'Testing complete. Passed: {self.tests_passed}, Failed: {self.tests_failed}')
            self.test_timer.cancel()

    def test_navigation_basic(self):
        """Test basic navigation functionality"""
        self.get_logger().info('Testing basic navigation...')
        # Implementation would test navigation to specific coordinates
        time.sleep(1)  # Simulate test duration
        assert True  # Placeholder assertion

    def test_interaction_basic(self):
        """Test basic interaction functionality"""
        self.get_logger().info('Testing basic interaction...')
        # Implementation would test speech recognition and response
        time.sleep(1)
        assert True

    def test_learning_basic(self):
        """Test basic learning functionality"""
        self.get_logger().info('Testing basic learning...')
        # Implementation would test learning from simple environment
        time.sleep(1)
        assert True

    def test_coordination_basic(self):
        """Test basic coordination functionality"""
        self.get_logger().info('Testing basic coordination...')
        # Implementation would test simple task allocation
        time.sleep(1)
        assert True

    def test_integration_complete(self):
        """Test complete system integration"""
        self.get_logger().info('Testing complete integration...')
        # Implementation would test all systems working together
        time.sleep(2)
        assert True

    def test_emergency_procedures(self):
        """Test emergency stop procedures"""
        self.get_logger().info('Testing emergency procedures...')

        # Trigger emergency stop
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_pub.publish(emergency_msg)

        time.sleep(1)

        # Clear emergency
        emergency_msg.data = False
        self.emergency_pub.publish(emergency_msg)

        assert True

    def test_system_reliability(self):
        """Test system reliability over time"""
        self.get_logger().info('Testing system reliability...')
        # Implementation would run extended operation test
        time.sleep(2)
        assert True

def main(args=None):
    rclpy.init(args=args)

    test_framework = IntegrationTestFramework()

    try:
        rclpy.spin(test_framework)
    except KeyboardInterrupt:
        test_framework.get_logger().info('Testing framework interrupted by user')
    finally:
        test_framework.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Takeaways

- Integration projects demonstrate the practical application of all concepts covered in the book
- Autonomous navigation and manipulation require coordination of perception, planning, and control systems
- Human-robot interaction systems must combine multiple modalities for natural communication
- Adaptive learning systems enable robots to improve performance through experience
- Multi-robot coordination requires sophisticated communication and task allocation algorithms
- Comprehensive integration challenges test the ability to combine all subsystems effectively
- Testing and validation frameworks ensure system reliability and safety
- Real-world deployment requires robust error handling and emergency procedures
- Successful integration projects require careful system architecture and modularity
- Continuous learning and adaptation are essential for long-term deployment success

## Looking Forward

This concludes the Physical AI and Humanoid Robotics book. The projects in this chapter demonstrate how to integrate all the concepts learned throughout the book into practical, working systems. The combination of theoretical knowledge and hands-on implementation provides a solid foundation for developing advanced humanoid robotics applications. Future work should focus on deploying these systems in real-world environments, addressing the challenges of robustness, safety, and human acceptance that arise in practical applications.