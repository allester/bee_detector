import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Create the main window
window = tk.Tk()
window.title("YOLOv8 Object Detection")
window.geometry("800x600")

# Create the canvas for displaying the video frames
canvas = tk.Canvas(window, width=480, height=480)
canvas.pack(side=tk.TOP)


# Create the slider widget for frame navigation
slider = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, length=600)
slider.pack(side=tk.BOTTOM)

# Create the listbox widget for displaying detection frames
listbox = tk.Listbox(window, width=40)
listbox.pack(side=tk.RIGHT)

# Set the path to the video file
video_path = ""

# Load the YOLOv8 model
model = YOLO('runs/detect/yolov8n_bee6/weights/best.pt')

frames = []
detection_frames = []

def process_video():
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Check if there are any detections
        if results[0].boxes:
            detection_frames.append(frame_count)

        frame_count += 1

    cap.release()  # Release the video capture object

    # Configure the slider widget based on the number of frames
    slider.config(to=len(frames) - 1, command=display_frame)
    display_frame(0)

def display_frame(frame_index):
    frame_index = int(frame_index)
    frame = frames[frame_index]

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Convert the annotated frame to Tkinter-compatible format
    image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    img = ImageTk.PhotoImage(image_pil)

    # Update the canvas with the annotated frame
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img

    # Update the listbox with the detection frames
    listbox.delete(0, tk.END)
    for frame_index in detection_frames:
        listbox.insert(tk.END, f"Frame {frame_index}")

def on_slider_changed(event):
    frame_index = slider.get()
    display_frame(frame_index)

def open_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    process_video()

def export_frames():
    output_dir = filedialog.askdirectory()

    for frame_index in detection_frames:
        frame = frames[frame_index]
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_pil.save(f"{output_dir}/frame_{frame_index}.png")

# Create the button widget for opening the file
button_open = tk.Button(window, text="Open File", command=open_file)
button_open.pack(side=tk.TOP)

# Create the button widget for exporting frames
button_export = tk.Button(window, text="Export Frames", command=export_frames)
button_export.pack(side=tk.TOP)

# Bind the slider event to the frame display function
slider.bind("<ButtonRelease-1>", on_slider_changed)

# Start the Tkinter event loop
window.mainloop()
