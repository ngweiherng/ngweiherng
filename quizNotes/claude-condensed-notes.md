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
