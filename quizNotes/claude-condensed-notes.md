# Edge Computing Lab Quiz - Quick Reference (40-Minute Format)

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

## Key Comparison Points

| Technology | Edge Benefits | Limitations |
|------------|---------------|------------|
| MediaPipe  | Lightweight, pre-trained models | Limited model customization |
| Quantization | 4-6× speedup, smaller size | Slight accuracy decrease |
| MQTT | Low bandwidth, pub/sub model | Limited QoS on constrained devices |
| Edge CV | Privacy, low latency | Limited processing power |
| Edge Audio | Real-time processing | Limited recognition accuracy |

## Frequently Tested Concepts

1. **Hardware limits**: RAM (1-8GB), processing power, storage, power consumption
2. **FPS improvement techniques**: Downsampling, quantization, algorithm selection
3. **MediaPipe landmarks**: Face (468 points), Hand (21 points), Pose (33 points)
4. **Audio features**: Time domain vs. frequency domain analysis
5. **MQTT QoS levels**: 0 (at most once), 1 (at least once), 2 (exactly once)
6. **Edge ML workflow**: Capture → Preprocess → Inference → Post-process → Act
