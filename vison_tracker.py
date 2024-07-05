import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Menu, messagebox
import tkinter as tk
from tkinter import ttk
from typing import Tuple
import threading
import numpy as np
import os
from fer import FER

class FaceDetectionApp:
    def __init__(self, window: Tk):
        self.window = window
        self.window.title("Face Detection App")
        self.window.geometry("400x300")

        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=10)
        self.style.configure("TLabel", font=("Helvetica", 12), padding=10)

        self.main_frame = ttk.Frame(window, padding="10 10 10 10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        
        self.upload_button = ttk.Button(self.main_frame, text="Upload Image for Detection", command=self.upload_image)
        self.upload_button.grid(row=0, column=0, pady=10, padx=10)

        self.webcam_button = ttk.Button(self.main_frame, text="Webcam Detection", command=self.start_vid_Cam)
        self.webcam_button.grid(row=1, column=0, pady=10, padx=10)

        self.exit_button = ttk.Button(self.main_frame, text="Exit", command=self.on_closing)
        self.exit_button.grid(row=2, column=0, pady=10, padx=10)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.stop_event = threading.Event()
        self.thread = None
        self.img_with_faces = None

        self.create_menu()

        # Initialize OpenCV's face detector and facial landmark predictor
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.landmark_model = cv2.face.createFacemarkLBF()

        # Define the absolute path to the lbfmodel.yaml file
        model_path = r"C:\Users\SURESH\Desktop\Programs\Python projects\Vision Tracker\lbfmodel.yaml"
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.landmark_model.loadModel(model_path)

        # Initialize FER emotion detector
        self.emotion_detector = FER(mtcnn=True)

    def create_menu(self):
        menubar = Menu(self.window)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save Detected Image", command=self.save_detected_image)
        filemenu.add_command(label="Save Detected Faces", command=self.save_detected_faces)
        filemenu.add_command(label="Record Video with Detection", command=self.record_video)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=filemenu)
        self.window.config(menu=menubar)

    def detect_faces(self, image: cv2.Mat) -> Tuple[list, cv2.Mat]:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return faces, gray_image

    def draw_faces(self, image: cv2.Mat, faces: list) -> cv2.Mat:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return image

    def detect_and_draw_landmarks(self, image: cv2.Mat, gray_image: cv2.Mat, faces: list):
        if len(faces) > 0:
            faces_rects = np.array(faces)
            _, landmarks = self.landmark_model.fit(gray_image, faces_rects)
            for landmark in landmarks:
                for (x, y) in landmark[0]:
                    cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        return image

    def detect_emotions(self, image: cv2.Mat):
        results = self.emotion_detector.detect_emotions(image)
        for result in results:
            emotions = result["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            x, y, w, h = result["box"]
            cv2.putText(image, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    def static_img(self, imagepath: str):
        try:
            img = cv2.imread(imagepath)
            if img is None:
                print("Image not found. Check the path.")
                return
            print(f"Original image shape: {img.shape}")

            faces, gray_image = self.detect_faces(img)
            print(f"Gray image shape: {gray_image.shape}")
            print(f"Number of faces detected: {len(faces)}")

            self.img_with_faces = self.draw_faces(img, faces)
            self.img_with_faces = self.detect_and_draw_landmarks(self.img_with_faces, gray_image, faces)
            self.img_with_faces = self.detect_emotions(self.img_with_faces)
            img_rgb = cv2.cvtColor(self.img_with_faces, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(20, 10))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")

    def save_detected_faces(self):
        if self.img_with_faces is not None:
            faces, gray_image = self.detect_faces(self.img_with_faces)
            for i, (x, y, w, h) in enumerate(faces):
                face = self.img_with_faces[y:y+h, x:x+w]
                file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                         filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
                                                         initialfile=f"face_{i}.jpg")
                if file_path:
                    cv2.imwrite(file_path, face)
                    messagebox.showinfo("Face Saved", f"Face saved as '{file_path}'.")
        else:
            messagebox.showwarning("No Image", "No detected image to save.")

    def save_detected_image(self):
        if self.img_with_faces is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.img_with_faces)
                messagebox.showinfo("Image Saved", f"Image saved as '{file_path}'.")
        else:
            messagebox.showwarning("No Image", "No detected image to save.")

    def vid_Cam(self):
        video_capture = cv2.VideoCapture(0)

        def detect_bounding_box(vid: cv2.Mat) -> cv2.Mat:
            gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
            return self.draw_faces(vid, faces), gray_image, faces

        try:
            while not self.stop_event.is_set():
                result, video_frame = video_capture.read()
                if not result:
                    print("Failed to read frame from webcam. Exiting...")
                    break

                video_frame, gray_image, faces = detect_bounding_box(video_frame)
                video_frame = self.detect_and_draw_landmarks(video_frame, gray_image, faces)
                video_frame = self.detect_emotions(video_frame)
                cv2.imshow("My Face Detection Project", video_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

    def record_video(self):
        video_capture = cv2.VideoCapture(0)
        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

        def detect_bounding_box(vid: cv2.Mat) -> cv2.Mat:
            gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
            return self.draw_faces(vid, faces), gray_image, faces

        try:
            while True:
                result, video_frame = video_capture.read()
                if not result:
                    print("Failed to read frame from webcam. Exiting...")
                    break

                video_frame, gray_image, faces = detect_bounding_box(video_frame)
                video_frame = self.detect_and_draw_landmarks(video_frame, gray_image, faces)
                video_frame = self.detect_emotions(video_frame)
                out.write(video_frame)
                cv2.imshow("Face Detection Recording", video_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            video_capture.release()
            out.release()
            cv2.destroyAllWindows()

    def start_vid_Cam(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.vid_Cam)
        self.thread.start()

    def upload_image(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.static_img(filepath)

    def on_closing(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.window.quit()
        self.window.destroy()
        cv2.destroyAllWindows()

def main():
    window = Tk()
    app = FaceDetectionApp(window)
    window.mainloop()

if __name__ == "__main__":
    main()
