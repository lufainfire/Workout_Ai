import cv2
from ultralytics import YOLO, solutions
import easygui

list = []
# Load video
#cap = cv2.VideoCapture(r"C:/Users/lufai/Pictures/Camera Roll/file5.mp4")

def pushup(push, path):
    cap = cv2.VideoCapture(r"C:/Users/lufai/Pictures/Camera Roll/file5.mp4")
    #cap = cv2.VideoCapture(path)

    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Load YOLO pose model
    model = YOLO("yolo11n-pose.pt")

    # Video writer
    video_writer = cv2.VideoWriter("workouts.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


    pushup_tracker = solutions.AIGym(
        line_thickness=2,
        view_img=True,
        pose_type="pushup",
        kpts_to_check=[6, 8, 10]
    )


    pushup_count = 0

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(f"Total push-ups performed: {pushup_count}")
            break




        im0 = pushup_tracker.monitor(im0)


        if hasattr(pushup_tracker, "count"):  # Check if `count` attribute exists
            pushup_count = pushup_tracker.count


        cv2.putText(im0, f"Push-Ups: {pushup_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        video_writer.write(im0)


        cv2.imshow("Workout Tracker", im0)

        if pushup_count[0] >= push:
            print(f"Push-up goal reached! Exiting... Total push-ups: {pushup_count}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
    video_writer.release()
    cap.release()


    print(f"Final push-up count: {pushup_count}")
    list.append(pushup_count)


def squat(squat):
    cap = cv2.VideoCapture(r"C:/Users/lufai/Pictures/Camera Roll/file5.mp4")
    #cap = cv2.VideoCapture("r"+path)

    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Load YOLO pose model
    model = YOLO("yolo11n-pose.pt")

    # Video writer
    video_writer = cv2.VideoWriter("workouts.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pushup_tracker = solutions.AIGym(
        line_thickness=2,
        view_img=True,
        pose_type="squat",
        kpts_to_check=[15, 13, 11]
    )

    squat_count = 0

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(f"Total push-ups performed: {squat_count}")
            break

        im0 = pushup_tracker.monitor(im0)

        if hasattr(pushup_tracker, "count"):  # Check if `count` attribute exists
            squat_count = pushup_tracker.count

        cv2.putText(im0, f"Push-Ups: {squat_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video_writer.write(im0)

        cv2.imshow("Workout Tracker", im0)

        if squat_count[0] >= squat:
            print(f"Push-up goal reached! Exiting... Total push-ups: {squat_count}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.destroyAllWindows()
        video_writer.release()
        cap.release()

        print(f"Final push-up count: {squat_count}")
        list.append(squat_count)

path = push = easygui.enterbox("what is the path to the video?")
push = easygui.enterbox("How many pushups?")
squats = easygui.enterbox("How many squats?")
curls = easygui.enterbox("How many curls on each arm (left to right)?")
pushup(int(push), path)

try:
    squat(int(squats), path)
except:
    pass
try:
    pushup(int(curls), path)
except:
    pass
print(list)