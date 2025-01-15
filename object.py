import cv2
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO('yolo11n.pt')  
# Load an image for detection
image_path = 'buddy.jpg'  # Update with your test image path
img = cv2.imread(image_path)

# Run inference on the image
results = model.predict(source=img, save=False, conf=0.5)


# Visualize results
# Process results
for result in results:
    if result.boxes:  # Check if there are any detections
        for box in result.boxes:
            # Extract box details
            xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box, 'xyxy') else None
            conf = box.conf.cpu().numpy()[0] if hasattr(box, 'conf') else None
            cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else None
            
            if xyxy is not None and conf is not None and cls_id is not None:
                x1, y1, x2, y2 = xyxy
                class_name = model.names[cls_id]
                
                # Draw bounding boxes and labels
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print("Incomplete box attributes.")
    else:
        print("No detections found.")


# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Optional: Save the output image
cv2.imwrite('output.png', img)
