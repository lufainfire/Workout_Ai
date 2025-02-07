import cv2
from ultralytics import YOLO, solutions
import winsound

winsound.PlaySound('sound.wav', winsound.SND_FILENAME)
model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

gym_object = solutions.AIGym(
    line_thickness=2,
    view_img=True,
    pose_type="pushup",
    kpts_to_check=[6, 8, 10],
)

angles = []

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = model.track(im0, verbose=False)
    im0, angle_data = gym_object.start_counting(im0, results)
    angles.append(angle_data)

cap.release()
cv2.destroyAllWindows()
print(angles)