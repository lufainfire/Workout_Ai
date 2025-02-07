import cv2

from ultralytics import YOLO, solutions

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(r"C:\Users\lufai\Pictures\Camera Roll/file5.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

model= YOLO("yolo11n-pose.pt")

# Video writer
video_writer = cv2.VideoWriter("workouts.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init AIGym
gym = solutions.AIGym(
    show=True,  # Display the frame
    kpts=[5, 7, 9],  #
    up_angle =145.0,
    down_angle = 90.0,
    model="yolo11n-pose.pt",  # Path to the YOLO11 pose estimation model file
    line_width=2,  # Adjust the line width for bounding boxes and text display

)
gym_objects = {
    "pushup": solutions.AIGym(line_thickness=2, view_img=True, pose_type="pushup", kpts_to_check=[6, 8, 10]),
    "pullup": solutions.AIGym(line_thickness=2, view_img=True, pose_type="pullup", kpts_to_check=[6, 8, 10]),
    # Add other exercises as needed
}
score = []

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = gym.monitor(im0)
    video_writer.write(im0)
    results = model.track(im0, verbose=False)
    #im0, angle_data = gym_object.start_counting(im0, results)
    #score.append(angle_data)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_writer.release()