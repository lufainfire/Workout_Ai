import cv2
from ultralytics import YOLO, solutions

# Load models
classification_model = YOLO("exercise_classification_model.pt")
pose_models = {
    "pushup": YOLO("yolov8n-pose-pushup.pt"),
    "pullup": YOLO("yolov8n-pose-pullup.pt"),
    "squat": YOLO("yolov8n-pose-squat.pt")
}

cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

gym_objects = {
    "pushup": solutions.AIGym(line_thickness=2, view_img=True, pose_type="pushup", kpts_to_check=[6, 8, 10]),
    "pullup": solutions.AIGym(line_thickness=2, view_img=True, pose_type="pullup", kpts_to_check=[6, 8, 10]),
    "squat": solutions.AIGym(line_thickness=2, view_img=True, pose_type="squat", kpts_to_check=[6, 8, 10])
}

current_exercise = None
frame_count = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_count += 1

    # Classify exercise type every few frames
    if frame_count % 30 == 0:  # Adjust frequency as needed
        classification_results = classification_model.predict(im0)
        current_exercise = classification_results[0].probs.top1  # Assuming top1 gives the exercise type

    if current_exercise:
        pose_model = pose_models[current_exercise]
        gym_object = gym_objects[current_exercise]
        results = pose_model.track(im0, verbose=False)
        im0 = gym_object.start_counting(im0, results, frame_count)

    cv2.imshow("Workout Monitoring", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()