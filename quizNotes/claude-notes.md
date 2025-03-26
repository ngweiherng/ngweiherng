# Edge Computing Lab Quiz Notes

## Table of Contents
1. [Setup and General Commands](#setup-and-general-commands)
2. [Sound Analytics (Lab 2)](#sound-analytics-lab-2)
3. [Image Analytics (Lab 3)](#image-analytics-lab-3)
4. [Video Analytics (Lab 4)](#video-analytics-lab-4)
5. [Edge ML and Quantization (Lab 5)](#edge-ml-and-quantization-lab-5)
6. [MQTT for IoT Communication (Lab 6)](#mqtt-for-iot-communication-lab-6)
7. [Key Concepts and Definitions](#key-concepts-and-definitions)

---

## Setup and General Commands

### Virtual Environment Setup (Used in all labs)
```bash
# Create virtual environment
sudo apt install python3-venv
python3 -m venv [env_name]  # Replace [env_name] with appropriate name

# Activate virtual environment
source [env_name]/bin/activate

# Update system packages
sudo apt update
sudo apt upgrade
```

### Common Dependencies
```bash
# OpenCV for image/video
pip install opencv-python

# Audio libraries
pip install pyaudio sounddevice scipy matplotlib librosa

# ML/Edge libraries
pip install torch torchvision torchaudio mediapipe

# IoT communication
pip install paho-mqtt
```

---

## Sound Analytics (Lab 2)

### Key Commands
```bash
# Record audio
arecord --duration=10 test.wav

# Play audio
aplay test.wav
```

### Audio Processing Libraries
- **PyAudio/SoundDevice**: Capture audio stream
- **SciPy**: Signal processing
- **Librosa**: Audio feature extraction

### Important Audio Features
1. **Spectrogram**: Visual representation of frequency spectrum over time
2. **Chromogram**: Representation of the twelve different pitch classes
3. **Mel-Spectrogram**: Uses Mel Scale (perceptual scale of pitches) instead of Frequency
4. **MFCC (Mel Frequency Cepstral Coefficients)**: Representation of short-term power spectrum

### Speech Recognition Code Snippet
```python
import speech_recognition as sr

r = speech_recognition.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    
# Two recognition methods:
# 1. Offline (on device) - less accurate
sphinx_text = r.recognize_sphinx(audio)

# 2. Online (cloud API) - more accurate
google_text = r.recognize_google(audio)
```

### Audio Filtering
```python
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
```

---

## Image Analytics (Lab 3)

### Image Capture and Processing
```python
import cv2
import numpy as np

# Capture video
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Image segmentation based on color
red = frame.copy()
red[:, :, 1] = 0  # Remove green channel
red[:, :, 0] = 0  # Remove blue channel
```

### Feature Extraction - HOG (Histogram of Oriented Gradients)
```python
from skimage import feature

# Extract HOG features
gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
h, hog_image = feature.hog(
    gray, 
    orientations=8, 
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1), 
    visualize=True
)
```

### Face Detection with MediaPipe
```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process frame
results = face_mesh.process(rgb_frame)

# Draw landmarks if face detected
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            face_landmarks, 
            mp_face_mesh.FACEMESH_CONTOURS
        )
```

---

## Video Analytics (Lab 4)

### Optical Flow Methods
```python
# 1. Lucas-Kanade method
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

# 2. Farneback dense optical flow
flow = cv2.calcOpticalFlowFarneback(
    old_gray, frame_gray, 
    None, 0.5, 3, 15, 3, 5, 1.2, 0
)
```

### Hand Landmark Detection with MediaPipe
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize model
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Process frame
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
detection_result = detector.detect(image)

# Finger landmarks are indexed 0-20
# Thumb tip is index 4
# Index finger tip is index 8
```

### Object Detection with MediaPipe
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize model
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5
)
detector = vision.ObjectDetector.create_from_options(options)

# Process image
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
detection_result = detector.detect(mp_image)

# Draw bounding boxes
for detection in detection_result.detections:
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    bbox = detection.bounding_box
    
    # Draw rectangle and label
    cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), 
                 (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 255, 0), 2)
    cv2.putText(frame, f"{category_name} {probability}", 
               (bbox.origin_x, bbox.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

---

## Edge ML and Quantization (Lab 5)

### Loading MobileNetV2 (Non-Quantized)
```python
import torch
import torchvision.models as models

# Load pre-trained model
model = models.mobilenet_v2(pretrained=True)
model.eval()
```

### Loading MobileNetV2 (Quantized)
```python
import torch
import torchvision.models as models

# Load pre-trained quantized model
model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model.eval()
```

### Quantization Types
1. **Post-Training Quantization**: 
   - Applied after training is complete
   - Converts 32-bit floats to 8-bit integers
   - Easiest to implement but may have accuracy loss

2. **Quantization-Aware Training**:
   - Inserts fake quantization during training
   - Better accuracy than post-training quantization
   - Requires retraining the model

### Performance Comparison
- Non-quantized MobileNetV2: ~5-6 FPS on Raspberry Pi 4B
- Quantized MobileNetV2: ~30 FPS on Raspberry Pi 4B

### Image Preprocessing for ML Models
```python
import torchvision.transforms as transforms

# Standard image transformations for MobileNetV2
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Apply transformation to frame
input_tensor = transform(frame)
input_batch = input_tensor.unsqueeze(0)
```

---

## MQTT for IoT Communication (Lab 6)

### MQTT Components
1. **MQTT Broker**: Accepts and delivers messages between clients
2. **Topic**: Namespace for messages on the broker
3. **Publisher**: Client that sends messages to topics
4. **Subscriber**: Client that receives messages from topics

### Setting Up MQTT Broker (Mosquitto)
```bash
# Install Mosquitto broker
sudo apt install mosquitto

# Edit configuration
sudo nano /etc/mosquitto/mosquitto.conf

# Add these lines to config:
listener 1883
allow_anonymous true

# Start broker manually
sudo mosquitto -c /etc/mosquitto/mosquitto.conf

# OR set to run on boot
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# Check status
systemctl status mosquitto
```

### MQTT Publisher Code
```python
import paho.mqtt.client as mqtt
import time

client = mqtt.Client("Publisher")
client.connect("broker_ip_address", 1883)  # Replace with actual broker IP

while True:
    client.publish("test/topic", "Hello, MQTT!")
    time.sleep(5)
```

### MQTT Subscriber Code
```python
import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    print(f"Received message '{message.payload.decode()}' on topic '{message.topic}'")

client = mqtt.Client("Subscriber")
client.on_message = on_message
client.connect("broker_ip_address", 1883)  # Replace with actual broker IP
client.subscribe("test/topic")
client.loop_forever()
```

### Image Transfer Using MQTT
```python
# Publisher (image capture and send)
import paho.mqtt.client as mqtt
import cv2
import base64

def capture_and_send_image():
    client = mqtt.Client("ImagePublisher")
    client.connect("broker_ip_address", 1883)
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    # Encode image to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    
    # Publish encoded image
    client.publish("camera/image", jpg_as_text)
    client.disconnect()

# Subscriber (receive and display image)
import paho.mqtt.client as mqtt
import cv2
import base64
import numpy as np

def on_message(client, userdata, msg):
    # Decode image from base64 string
    img_data = base64.b64decode(msg.payload)
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Display or process the image
    cv2.imshow("Received Image", img)
    cv2.waitKey(1)

client = mqtt.Client("ImageSubscriber")
client.on_message = on_message
client.connect("broker_ip_address", 1883)
client.subscribe("camera/image")
client.loop_forever()
```

---

## Key Concepts and Definitions

### Edge Computing
- Processing data near the source instead of sending to cloud/central server
- Benefits: reduced latency, bandwidth savings, enhanced privacy, offline capabilities

### Edge Computer Vision (ECV)
- Recognized by Gartner as top emerging technology in 2023
- Benefits: real-time processing, enhanced privacy/security, reduced network dependency

### MQTT (Message Queue Telemetry Transport)
- Lightweight publish/subscribe messaging protocol for IoT
- Runs over TCP/IP
- Designed for constrained environments with limited bandwidth

### Quantization
- Process of reducing precision of model weights and activations
- Typically from 32-bit floating point to 8-bit integers
- Results in smaller model size and faster inference

### MediaPipe
- On-device ML framework for cross-platform multimodal ML pipelines
- Uses TensorFlow Lite for efficient edge deployment
- Supports vision, audio, and other input modalities

### Common Raspberry Pi Hardware Limitations
- Limited processing power compared to desktops/servers
- Memory constraints (typically 1-8GB RAM)
- Storage limitations on SD cards
- Power consumption considerations for battery-powered applications

### Audio Feature Extraction
- **Spectrogram**: Frequency vs. Time visualization
- **Mel Scale**: Perceptual scale of pitches judged by listeners to be equal in distance
- **MFCC**: Compact representation of audio signal based on human auditory system

### Computer Vision Techniques
- **HOG (Histogram of Oriented Gradients)**: Feature descriptor for object detection
- **Optical Flow**: Pattern of apparent motion between frames caused by object movement
- **Face/Hand Landmarks**: Key points that define the shape of a face or hand
