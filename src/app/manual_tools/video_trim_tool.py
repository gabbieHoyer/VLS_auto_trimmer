import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class VideoTrimTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Trim Tool")
        self.video_path = None
        self.video_basename = None
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.frame = None
        self.video_queue = []
        self.current_queue_index = -1
        self.save_dir = None  # Store the default save directory
        self.start_frame = None  # Start frame for trimming
        self.end_frame = None  # End frame for trimming

        # GUI Elements
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Sidebar for video queue
        sidebar_frame = tk.Frame(main_frame, width=200)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        tk.Label(sidebar_frame, text="Video Queue").pack(anchor=tk.W)
        self.queue_listbox = tk.Listbox(sidebar_frame, width=30, height=20)
        self.queue_listbox.pack(fill=tk.Y, expand=True)
        self.queue_listbox.bind('<<ListboxSelect>>', self.load_from_queue)

        # Canvas and controls
        content_frame = tk.Frame(main_frame)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(content_frame, width=640, height=480)
        self.canvas.pack(pady=10)

        self.frame_label = tk.Label(content_frame, text="Frame: 0 / 0")
        self.frame_label.pack()

        self.slider = tk.Scale(content_frame, from_=0, to=0, orient=tk.HORIZONTAL, length=600, command=self.slider_update)
        self.slider.pack(pady=5)

        # Trimming controls
        trim_frame = tk.Frame(content_frame)
        trim_frame.pack(pady=5)
        tk.Button(trim_frame, text="Set Start Frame", command=self.set_start_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(trim_frame, text="Set End Frame", command=self.set_end_frame).pack(side=tk.LEFT, padx=5)
        self.start_label = tk.Label(trim_frame, text="Start Frame: None")
        self.start_label.pack(side=tk.LEFT, padx=5)
        self.end_label = tk.Label(trim_frame, text="End Frame: None")
        self.end_label.pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(content_frame)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Select Save Folder", command=self.select_save_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Previous Frame", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        self.next_video_btn = tk.Button(btn_frame, text="Next Video", command=self.load_next_video, state=tk.DISABLED)
        self.next_video_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Trimmed Video", command=self.save_video).pack(side=tk.LEFT, padx=5)

    def select_save_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.save_dir = folder_path

    def set_start_frame(self):
        if not self.cap:
            messagebox.showwarning("Warning", "No video loaded")
            return
        self.start_frame = self.current_frame
        self.start_label.configure(text=f"Start Frame: {self.start_frame}")
        self.validate_frames()

    def set_end_frame(self):
        if not self.cap:
            messagebox.showwarning("Warning", "No video loaded")
            return
        self.end_frame = self.current_frame
        self.end_label.configure(text=f"End Frame: {self.end_frame}")
        self.validate_frames()

    def validate_frames(self):
        # Ensure start_frame is less than end_frame
        if self.start_frame is not None and self.end_frame is not None:
            if self.start_frame > self.end_frame:
                self.start_frame, self.end_frame = self.end_frame, self.start_frame
                self.start_label.configure(text=f"Start Frame: {self.start_frame}")
                self.end_label.configure(text=f"End Frame: {self.end_frame}")

    def load_video(self, video_path=None, from_queue=False):
        if not video_path:
            video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mp4")])
        if not video_path:
            return
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.frame = None
        self.start_frame = None
        self.end_frame = None
        self.start_label.configure(text="Start Frame: None")
        self.end_label.configure(text="End Frame: None")
        self.slider.set(0)
        self.slider.configure(to=0)
        self.frame_label.configure(text="Frame: 0 / 0")
        self.canvas.delete("all")
        if not from_queue:
            self.video_queue = []
            self.current_queue_index = -1
            self.queue_listbox.delete(0, tk.END)
            self.save_dir = None  # Reset save directory when loading a new single video
        self.video_path = video_path
        self.video_basename = os.path.splitext(os.path.basename(video_path))[0]
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            self.cap = None
            self.video_path = None
            self.video_basename = None
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.configure(to=self.total_frames - 1)
        self.frame_label.configure(text=f"Frame: 0 / {self.total_frames}")
        if from_queue:
            self.update_queue_selection()
        self.update_frame()

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        self.video_queue = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.mp4', '.avi'))
        ]
        if not self.video_queue:
            messagebox.showwarning("Warning", "No video files found in the selected folder")
            return
        self.queue_listbox.delete(0, tk.END)
        for video_path in self.video_queue:
            self.queue_listbox.insert(tk.END, os.path.basename(video_path))
        self.current_queue_index = 0
        self.save_dir = None  # Reset save directory when loading a new folder
        self.load_video(self.video_queue[0], from_queue=True)
        self.next_video_btn.configure(state=tk.NORMAL)

    def load_from_queue(self, event):
        selection = self.queue_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if index != self.current_queue_index:
            self.current_queue_index = index
            self.load_video(self.video_queue[index], from_queue=True)

    def load_next_video(self):
        if not self.video_queue or self.current_queue_index < 0:
            return
        self.current_queue_index = (self.current_queue_index + 1) % len(self.video_queue)
        self.load_video(self.video_queue[self.current_queue_index], from_queue=True)

    def update_queue_selection(self):
        self.queue_listbox.selection_clear(0, tk.END)
        if self.current_queue_index >= 0:
            self.queue_listbox.selection_set(self.current_queue_index)
            self.queue_listbox.see(self.current_queue_index)

    def update_frame(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame = frame
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(frame_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.frame_label.configure(text=f"Frame: {self.current_frame} / {self.total_frames}")
        self.slider.set(self.current_frame)

    def slider_update(self, val):
        self.current_frame = int(val)
        self.update_frame()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_frame()

    def save_video(self):
        if not self.cap:
            messagebox.showwarning("Warning", "No video loaded")
            return
        if self.start_frame is None or self.end_frame is None:
            messagebox.showwarning("Warning", "Please set both start and end frames for trimming")
            return
        if self.start_frame == self.end_frame:
            messagebox.showwarning("Warning", "Start and end frames cannot be the same")
            return

        default_output = f"{self.video_basename}_trimmed.mp4" if self.video_basename else "output_trimmed.mp4"
        initial_dir = self.save_dir if self.save_dir else (os.path.dirname(self.video_path) if self.video_path else None)
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            initialfile=default_output,
            initialdir=initial_dir
        )
        if not output_path:
            return
        # Update save_dir to the directory of the selected output path
        self.save_dir = os.path.dirname(output_path)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        for frame_idx in range(self.start_frame, self.end_frame + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        messagebox.showinfo("Success", "Trimmed video saved successfully")
        # Reset trimming points
        self.start_frame = None
        self.end_frame = None
        self.start_label.configure(text="Start Frame: None")
        self.end_label.configure(text="End Frame: None")
        self.update_frame()

    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrimTool(root)
    root.mainloop()

# python -m src.app.manual_tools.video_trim_tool