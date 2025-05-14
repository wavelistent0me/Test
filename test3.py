import torch
import cv2
import numpy as np
from retinaface import RetinaFace
from PIL import Image

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Number of GPUs: " + str(torch.cuda.device_count()))

print("Device: " + device)

# exit()

# Load model
model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
model.eval()
model.to(device)

def process_frame(frame, model, transform):
    """Process a single frame through the gaze detection pipeline"""
    # Convert frame to PIL Image for transform
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    height, width = frame.shape[:2]
    
    # Detect faces
    resp = RetinaFace.detect_faces(frame)
    if not isinstance(resp, dict):
        return frame  # Return original frame if no faces detected
    
    bboxes = [resp[key]['facial_area'] for key in resp.keys()]
    
    # Prepare input for the model
    img_tensor = transform(pil_frame).unsqueeze(0).to(device)
    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]
    
    input_data = {
        "images": img_tensor,
        "bboxes": norm_bboxes
    }
    
    # Get model predictions
    with torch.no_grad():
        output = model(input_data)
    
    # 画面可视化
    # Draw visualizations
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # BGR format for OpenCV
    
    for p, (bbox, heatmap) in enumerate(zip(bboxes, output['heatmap'][0])):
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[p % 3], 2)

        print("bbox")
        print(bbox)
        
        # Get gaze point from heatmap
        if output['inout'] is not None and output['inout'][0][p].item() > 0.5:
            heatmap_np = heatmap.detach().cpu().numpy()

            # Normalize heatmap to 0–255 and convert to uint8
            heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_uint8 = heatmap_norm.astype(np.uint8)

            # Resize heatmap to match original frame size
            heatmap_resized = cv2.resize(heatmap_uint8, (width, height))

            # Apply a colormap (e.g., JET) to make it colorful
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

            # Optional: blend the heatmap with the original frame
            overlayed_frame = cv2.addWeighted(frame, 0.6, heatmap_color, 0.05, 0)

            # Replace frame with overlayed_frame (or display separately)
            frame = overlayed_frame

            # 找到热力图中最大值的坐标
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            min_index = np.unravel_index(np.argmin(heatmap_np), heatmap_np.shape)

            print("max_index")
            print(max_index)
            print(min_index)
            print(int(max_index[0]/ heatmap_np.shape[1] * width))

            print("heatmap_np")
            print(heatmap_np.shape)

            print("x1 x2")
            print(x1, x2)

            # Define the region of interest (ROI) around the maximum value
            margin = 64  # Define the size of the margin around the max point
            xx1 = int(max_index[1] / heatmap_np.shape[1] * width)  
            yy1 = int(max_index[0] / heatmap_np.shape[0] * height) 
            xx2 = int(min_index[1] / heatmap_np.shape[1] * width)  
            yy2 = int(min_index[0] / heatmap_np.shape[0] * height)  

            print("xx1 yy1")
            print(xx1, yy1,xx2, yy2)

            
            # Calculate gaze target coordinates
            gaze_x = int(max_index[1] / heatmap_np.shape[1] * width)
            gaze_y = int(max_index[0] / heatmap_np.shape[0] * height)

            print("gaze_x gaze_y")
            print(gaze_x,gaze_y)

            
            # Calculate face center
            face_center_x = (int(x1) + int(x2)) // 2
            face_center_y = (int(y1) + int(y2)) // 2
            
            # Draw gaze point and line
            # cv2.circle(frame, (gaze_x, gaze_y), 10, colors[p % 3], -1)
            cv2.line(frame, (gaze_x, gaze_y), (face_center_x, face_center_y), colors[p % 3], 2)

            # cv2.rectangle(frame, (int(max_index[0]/ heatmap_np.shape[1] * width), 
            #                       int(max_index[1]/ heatmap_np.shape[0] * height)), 
            #                       (int(min_index[0]/ heatmap_np.shape[1] * width),
            #                         int(min_index[1]/ heatmap_np.shape[0] * height)), (0, 255, 0), 2)

            # cv2.rectangle(frame, (xx2, yy1), (xx1,yy2), (0, 255, 0), 2)
            # cv2.line(frame, (xx2, yy1), (xx1, yy2), colors[p % 3], 2)

            
            # Add confidence score
            conf_text = f"conf: {output['inout'][0][p].item():.2f}"
            # cv2.putText(frame, conf_text, (int(x1), int(y1) - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[p % 3], 2)
            
            
    
    return frame

def process_video(input_path, output_path):
    """Process entire video and save visualization"""
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(frame)
        
        # Process frame
        processed_frame = process_frame(frame, model, transform)
        
        # Write frame
        out.write(processed_frame)
        
        # Display progress
        frame_count += 1
        if frame_count % 10 == 0:  # Update progress every 10 frames
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing: {progress:.1f}% complete", end="")
    
    print("\nProcessing complete!")
    
    # Release resources
    cap.release()
    out.release()

def main():
    input_video = "v4.mp4"
    output_video = "output_v4.mp4"
    
    print(f"Processing video: {input_video}")
    print(f"Output will be saved to: {output_video}")
    
    process_video(input_video, output_video)

if __name__ == "__main__":
    main()