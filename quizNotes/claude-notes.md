# Edge Computing Lab Quiz - Quick Reference

## Virtual Environment Setup (All Labs)
```bash
sudo apt install python3-venv
python3 -m venv [env_name]
source [env_name]/bin/activate
```

## Lab 2: Sound Analytics

### Key Audio Features
- **Spectrogram**: Visual frequency spectrum over time
- **Chromogram**: 12 pitch classes representation
- **Mel-Spectrogram**: Uses perceptual Mel Scale
- **MFCC**: Compact audio representation based on human hearing

### Speech Recognition
```python
# Offline (on device) - less accurate
sphinx_text = r.recognize_sphinx(audio)

# Online (cloud API) - more accurate
google_text = r.recognize_google(audio)
```

### Audio Filtering
```python
# Bandpass filter for specific frequency range
b, a = butter(order, [low_freq/nyquist, high_freq/nyquist], btype='band')
filtered_audio = lfilter(b, a, audio_data)
```

## Lab 3: Image Analytics

### Image Processing
```python
# Color conversion
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Color segmentation
red_only = frame.copy()
red_only[:, :, 1:3] = 0  # Zero out G and B channels
```

### HOG Feature Extraction
```python
h, hog_image = feature.hog(
    gray, orientations=8, pixels_per_cell=(16, 16),
    cells_per_block=(1, 1), visualize=True
)
```

### MediaPipe Face Detection
```python
# Initialize
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Process
results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

## Lab 4: Video Analytics

### Optical Flow Methods
1. **Lucas-Kanade**: Sparse, tracks specific points
2. **Farneback**: Dense, calculates flow for all pixels

### MediaPipe Hand Detection
- Hand landmarks are indexed 0-20
- Thumb tip is index 4
- Index finger tip is index 8

### Object Detection Workflow
1. Load model with detection threshold
2. Convert frame to MediaPipe image format
3. Run detection on image
4. Process detection results (bounding boxes, class labels)

## Lab 5: Edge ML and Quantization

### Quantization Types
1. **Post-Training Quantization**: 32-bit floats → 8-bit integers
2. **Quantization-Aware Training**: Inserts fake quantization during training

### Performance Impact
- Non-quantized MobileNetV2: ~5-6 FPS
- Quantized MobileNetV2: ~30 FPS

### Image Preprocessing
```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

## Lab 6: MQTT for IoT Communication

### MQTT Components
- **Broker**: Message server (Mosquitto)
- **Publisher**: Sends messages to topics
- **Subscriber**: Receives topic messages
- **Topic**: Message namespace/channel

### Broker Configuration
```
# Add to mosquitto.conf
listener 1883
allow_anonymous true
```

### Publisher/Subscriber Pattern
```python
# Publisher
client = mqtt.Client("Publisher")
client.connect("broker_ip", 1883)
client.publish("topic/name", "message")

# Subscriber
client.on_message = on_message_callback
client.connect("broker_ip", 1883)
client.subscribe("topic/name")
client.loop_forever()
```

## Lab-Specific Command Examples

### MQTT Camera System (Lab 6)
- **MQTT Topics Structure**:
  ```python
  MQTT_BROKER = "localhost"
  MQTT_PORT = 1883
  TRIGGER_TOPIC = "camera/trigger"
  IMAGE_TOPIC = "camera/image"
  SAVE_DIR = "received_images"
  ```

- **MQTT Camera Client Code Pattern**:
  ```python
  # Key imports
  import paho.mqtt.publish as publish
  import time
  import base64
  import os
  import cv2
  
  # Image capture and publish
  cam = cv2.VideoCapture(0)
  ret, frame = cam.read()
  timestamp = int(time.time())
  filename = f"capture_{timestamp}.jpg"
  cv2.imwrite(filename, frame)
  print(f"Image saved as {filename}")
  print("Image published to MQTT topic")
  ```

### Optical Flow Parameters (Lab 4)
- **Lucas-Kanade Optical Flow**:
  ```python
  # Original parameters
  lk_params = dict(
      winSize=(15, 15),
      maxLevel=2,
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
  )
  
  # Improved parameters
  lk_params = dict(
      winSize=(21, 21),  # Larger window for smoother tracking
      maxLevel=3,        # Increase pyramid levels for robustness
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01)
  )
  ```

- **Farneback Dense Optical Flow**:
  ```python
  # Original parameters
  flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 
                                      0.5, 3, 15, 3, 5, 1.2, 0)
  
  # Modified parameters for more stable flow
  flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None,
                                      0.5, 5, 20, 3, 7, 1.5, 0)
  ```

### Hand Landmark Detection (Lab 4)
- **MediaPipe Hand Landmark Indices**:
  ```python
  # Finger tips indices
  finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
  finger_base = [6, 10, 14, 18]  # Corresponding PIP joints
  thumb_tip = 4                  # Thumb tip
  thumb_base = 3                 # Thumb IP joint
  ```

- **Finger Counting Logic**:
  ```python
  def count_fingers(hand_landmarks):
      """Counts the number of raised fingers excluding the thumb."""
      fingers = 0
      
      # Check four fingers
      for tip, base in zip(finger_tips, finger_base):
          if hand_landmarks[tip].y < hand_landmarks[base].y:
              fingers += 1
              
      # Check thumb separately (horizontal position)
      if hand_landmarks[4].x < hand_landmarks[3].x:  # For right hand
          fingers += 1
          
      return fingers
  ```

### Color Segmentation (Lab 3)
- **Color Boundaries for Segmentation**:
  ```python
  boundaries = [
      ([17, 15, 100], [50, 56, 200]),   # Red
      ([86, 31, 4], [220, 88, 50]),     # Blue
      ([25, 90, 4], [62, 200, 50]),     # Green
      ([20, 100, 100], [30, 255, 255])  # Yellow (in HSV)
  ]
  ```

- **Image Resizing for Performance**:
  ```python
  # Resize for faster processing
  frame = cv2.resize(frame, (256, 256))  # Significantly improves FPS
  ```

### Deep Learning Optimization (Lab 5)
- **Performance Comparison**:
  - Standard MobileNetV2: ~2.3-2.9 FPS
  - Quantized MobileNetV2: ~15-16 FPS (5-6× improvement)

- **Model Size Reduction**:
  - Original CNN: 0.179 MB
  - Quantized CNN: 0.050 MB (72% reduction)

- **Accuracy Impact**:
  - Original CNN: 97.95% (FP32)
  - Quantized CNN: 98.03% (INT8) - Accuracy sometimes improves with quantization!
  - QAT CNN: 97.87% (INT8) - Slight decrease but still very good

### Audio Feature Extraction (Lab 2)
- **Bandpass Filter Parameters**:
  ```python
  # Bandpass filter design (100-2500 Hz)
  sos = design_filter(100, 2500, RATE, 3)
  ```

- **Wake Word Detection**:
  ```python
  def detect_wake_word(text):
      if "hi computer" in text.lower():
          print("WAKE WORD DETECTED: 'Hi Computer'")
          print("I'm ready for your command!")
          return True
      return False
  ```

- **Speech Recognition Performance**:
  - Google: ~1 second, high accuracy
  - Sphinx: ~7 seconds, lower accuracy
  - Wit.ai: ~3 seconds, high accuracy

## Critical Command Syntax

### Raspberry Pi Specific
- **System updates**: `sudo apt update` and `sudo apt upgrade`
- **Service management**: `sudo systemctl start [service]`, `sudo systemctl enable [service]`
- **Check service status**: `systemctl status [service]`
- **Edit system files**: `sudo nano /path/to/file`
- **VNC connection**: Enable via `sudo raspi-config` → Interface Options → VNC → Enable
- **Check IP address**: `hostname -I`
- **Install packages**: `sudo apt install [package-name]`
- **File permissions**: 
  - `chmod +x filename.py` (make executable)
  - `chmod u+r filename` (user read)
  - `chmod u+w filename` (user write)
  - `chmod u+rw filename` (user read and write)
  - `chmod a+r filename` (read for all)
  - Number system: 4=read, 2=write, 1=execute
  - `chmod 755 filename` (rwx for owner, rx for group/others)

### OpenCV
- **Reading frames**: `cv2.VideoCapture(0)` then `ret, frame = cap.read()`
- **Color conversion**: `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`
- **Displaying**: `cv2.imshow("Window Title", frame)`
- **Drawing rectangle**: `cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)`
- **Adding text**: `cv2.putText(frame, "Text", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)`
- **Optical flow**: `cv2.calcOpticalFlowPyrLK()` or `cv2.calcOpticalFlowFarneback()`
- **Waiting for key**: `cv2.waitKey(1)` (1ms delay) or `cv2.waitKey(0)` (wait indefinitely)
- **Releasing**: `cap.release()` and `cv2.destroyAllWindows()`

### Audio
- **Recording**: `arecord --duration=10 test.wav`
- **Playing**: `aplay test.wav`
- **PyAudio stream**: `p = pyaudio.PyAudio()` then `stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)`
- **SoundDevice record**: `sounddevice.rec(frames, samplerate=RATE, channels=CHANNELS)`

### MediaPipe
- **Face mesh**: `mp.solutions.face_mesh.FaceMesh()`
- **Drawing utilities**: `mp.solutions.drawing_utils.draw_landmarks()`
- **Image conversion**: `mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)`
- **Detection method**: `detector.detect(image)` (not `detector.process()`)

### MQTT
- **Client initialization**: `mqtt.Client("ClientID")`
- **Connection**: `client.connect("broker_ip", 1883)`
- **Publishing**: `client.publish("topic/name", "message")`
- **Subscribing**: `client.subscribe("topic/name")`
- **Message callback**: `client.on_message = on_message_function`
- **Starting loop**: `client.loop_forever()` or `client.loop_start()`

### PyTorch
- **Model loading**: `models.mobilenet_v2(pretrained=True)`
- **Quantized model**: `models.quantization.mobilenet_v2(pretrained=True, quantize=True)`
- **Model mode**: `model.eval()` (not `model.evaluate()`)
- **Inference**: `with torch.no_grad(): outputs = model(input_batch)`
- **Get predictions**: `_, predicted = torch.max(outputs, 1)`

### Additional Libraries
- **MediaPipe model download**: `wget -q https://storage.googleapis.com/mediapipe-models/[model-path]`
- **Librosa load audio**: `y, sr = librosa.load('filename.wav')`
- **Speech recognition**: `r = sr.Recognizer()` then `r.recognize_google(audio)`
- **Bandpass filter**: `from scipy.signal import butter, lfilter`
- **FFT for audio**: `scipy.fft.fft(audio_data)`
- **Base64 encoding**: `base64.b64encode(image_data)` (for MQTT image transfer)
- **Image encoding**: `_, buffer = cv2.imencode('.jpg', frame)`
