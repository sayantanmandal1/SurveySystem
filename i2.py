import cv2
import numpy as np
import joblib
from tqdm import tqdm
from ultralytics import YOLO

# ---------------- Load YOLO and anomaly model ----------------
yolo_model = YOLO("C:/Users/msaya/Downloads/survey/yolov8n.pt")
model_data = joblib.load("C:/Users/msaya/Downloads/survey/models/balanced_anomaly_model.pkl")
best_model = model_data['model']
scaler = model_data['scaler']

# ---------------- Feature extraction (50 features) ----------------
def extract_features(frame):
    frame = cv2.resize(frame, (224, 224))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    features = []

    # 1. Statistical (8)
    features.extend([
        float(np.mean(gray)), float(np.std(gray)), float(np.var(gray)),
        float(np.min(gray)), float(np.max(gray)), float(np.median(gray)),
        float(np.percentile(gray, 25)), float(np.percentile(gray, 75))
    ])

    # 2. Histogram (16)
    hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256])
    features.extend(hist_gray.flatten().astype(float))

    # 3. Texture edges (3)
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    edges = cv2.filter2D(gray, -1, kernel)
    features.extend([float(np.mean(edges)), float(np.std(edges)), float(np.var(edges))])

    # 4. Gradient (6)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([float(np.mean(sobelx)), float(np.std(sobelx)),
                     float(np.mean(sobely)), float(np.std(sobely)),
                     float(np.mean(grad_mag)), float(np.std(grad_mag))])

    # 5. HSV color stats (6)
    for i in range(3):
        c = hsv[:,:,i]
        features.extend([float(np.mean(c)), float(np.std(c))])

    # 6. Contours (5)
    contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        perims = [cv2.arcLength(c, True) for c in contours]
        features.extend([
            len(contours),
            float(np.mean(areas)), float(np.std(areas)) if len(areas)>1 else 0.0,
            float(np.mean(perims)), float(np.std(perims)) if len(perims)>1 else 0.0
        ])
    else:
        features.extend([0.0,0.0,0.0,0.0,0.0])

    # 7. Frequency domain (4)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    mag_spectrum = np.log(np.abs(fshift)+1)
    features.extend([float(np.mean(mag_spectrum)), float(np.std(mag_spectrum)),
                     float(np.max(mag_spectrum)), float(np.min(mag_spectrum))])

    # 8. LBP-like (2)
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            c = gray[i,j]; code=0
            code |= (gray[i-1,j-1]>c)<<7
            code |= (gray[i-1,j]>c)<<6
            code |= (gray[i-1,j+1]>c)<<5
            code |= (gray[i,j+1]>c)<<4
            code |= (gray[i+1,j+1]>c)<<3
            code |= (gray[i+1,j]>c)<<2
            code |= (gray[i+1,j-1]>c)<<1
            code |= (gray[i,j-1]>c)<<0
            lbp[i,j]=code
    features.extend([float(np.mean(lbp)), float(np.std(lbp))])

    return np.array(features, dtype=np.float32).reshape(1, -1)  # shape (1,50)

# ---------------- Video inference ----------------
input_video = "C:/Users/msaya/Downloads/survey/test.avi"
cap = cv2.VideoCapture(input_video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("result_anomaly2.mp4", fourcc, fps, (width, height))

# Use tqdm for frame progress
for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1,y1,x2,y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        feat = extract_features(crop)
        feat_scaled = scaler.transform(feat)
        score = float(best_model.predict_proba(feat_scaled)[:,1][0])

        color = (0,0,255) if score>0.5 else (0,255,0)
        label = f"Anomaly {score:.2f}" if score>0.5 else f"Normal {score:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    out.write(frame)

cap.release()
out.release()
print("✅ Saved anomaly detection video to result_anomaly.mp4")
