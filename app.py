import gradio as gr
import cv2
import numpy as np
import queue
import threading
from pathlib import Path
from algorithms import load_algorithms
from utils.variations import create_variation_functions
from algorithm_reference import create_reference_interface

ALGORITHMS = load_algorithms()

VARIATIONS = create_variation_functions()

frame_buffer = queue.Queue(maxsize=30)
processing_lock = threading.Lock()

class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.skip_frames = 2
        self.last_frame = None

    def apply_custom_variations(self, frame, pose_angle=0, brightness=1, contrast=1,
                              temperature=4500, noise_level=0, snow_intensity=0):
        """Apply custom variations based on slider values"""
        result = frame.copy()

        if pose_angle != 0:
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, pose_angle, 1.0)
            result = cv2.warpAffine(result, rotation_matrix, (width, height))

        result = cv2.convertScaleAbs(result, alpha=contrast, beta=(brightness-1)*100)

        if temperature != 4500:
            temp = temperature / 100
            if temp <= 66:
                r = 255
                g = 99.4708025861 * np.log(temp) - 161.1195681661
                b = 138.5177312231 * np.log(temp - 10) - 305.0447927307
            else:
                r = 329.698727446 * np.power(temp - 60, -0.1332047592)
                g = 288.1221695283 * np.power(temp - 60, -0.0755148492)
                b = 255

            r = np.clip(r / 255, 0, 2)
            g = np.clip(g / 255, 0, 2)
            b = np.clip(b / 255, 0, 2)

            result = cv2.multiply(result, np.array([b, g, r]))

        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 50, result.shape).astype(np.uint8)
            result = cv2.add(result, noise)

        if snow_intensity > 0:
            height, width = result.shape[:2]
            num_snow = int((height * width) * snow_intensity * 0.01)
            snow_points = np.random.randint(0, width, (num_snow, 2))
            snow_points[:, 1] = np.random.randint(0, height, num_snow)

            for point in snow_points:
                x, y = point
                cv2.circle(result, (x, y), np.random.randint(1, 3), (255, 255, 255), -1)

        return cv2.convertScaleAbs(result)

    def process_frame(self, frame: np.ndarray, algorithm_name: str, variation_name: str,
                     pose_angle=0, brightness=1, contrast=1, temperature=4500,
                     noise_level=0, snow_intensity=0) -> tuple[np.ndarray, np.ndarray]:
        if frame is None:
            return None if variation_name == "None" else (None, None)

        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            if variation_name == "None":
                return self.last_frame
            return self.last_frame if isinstance(self.last_frame, tuple) else (self.last_frame, self.last_frame)

        with processing_lock:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                algorithm = ALGORITHMS.get(algorithm_name)
                if algorithm is None:
                    result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.last_frame = result
                    self.current_frame = result
                    return result if variation_name == "None" else (result, result)

                if not algorithm.is_initialized:
                    algorithm.initialize()

                detections = algorithm.detect(frame)
                result_frame = algorithm.draw_detections(frame, detections)
                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

                if variation_name == "None":
                    result_rgb = self.apply_custom_variations(
                        result_rgb, pose_angle, brightness, contrast, temperature,
                        noise_level, snow_intensity
                    )
                    self.last_frame = result_rgb
                    self.current_frame = result_rgb
                    return result_rgb

                varied_frame = VARIATIONS[variation_name](frame)
                varied_frame = self.apply_custom_variations(
                    varied_frame, pose_angle, brightness, contrast, temperature,
                    noise_level, snow_intensity
                )

                varied_detections = algorithm.detect(varied_frame)
                varied_result = algorithm.draw_detections(varied_frame, varied_detections)
                varied_rgb = cv2.cvtColor(varied_result, cv2.COLOR_BGR2RGB)

                result = (result_rgb, varied_rgb)
                self.last_frame = result
                self.current_frame = result
                return result

            except Exception as e:
                print(f"Error processing frame: {e}")
                backup_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return backup_frame if variation_name == "None" else (backup_frame, backup_frame)

def create_detection_interface():
    processor = VideoProcessor()

    with gr.Blocks() as detection_interface:
        with gr.Row():
            with gr.Column(scale=1):
                input_method = gr.Radio(
                    choices=["Webcam", "Video Upload"],
                    label="Input Method",
                    value="Webcam"
                )

                video_input = gr.Video(label="Upload Video", visible=False)
                process_btn = gr.Button("Process Video", visible=False)
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Processing Progress",
                    interactive=False,
                    visible=False
                )

                algorithm_dropdown = gr.Dropdown(
                    choices=list(ALGORITHMS.keys()),
                    label="Select Algorithm",
                    value=list(ALGORITHMS.keys())[0] if ALGORITHMS else None
                )

                variation_dropdown = gr.Dropdown(
                    choices=list(VARIATIONS.keys()),
                    label="Select Variation",
                    value="None"
                )

                capture_btn = gr.Button("Capture Frame", visible=True)

                with gr.Tab("Performance Settings"):
                    skip_frames = gr.Slider(1, 5, value=2, step=1, 
                                         label="Frame Skip (higher = better performance)")

                with gr.Tabs():
                    with gr.Tab("Pose Settings"):
                        pose_angle = gr.Slider(-90, 90, value=0, label="Custom Rotation Angle")
                    with gr.Tab("Lighting Settings"):
                        brightness = gr.Slider(0, 2, value=1, label="Brightness Factor")
                        contrast = gr.Slider(0, 2, value=1, label="Contrast Factor")
                        temperature = gr.Slider(2000, 7000, value=4500, label="Color Temperature (K)")
                    with gr.Tab("Perturbation"):
                        noise_level = gr.Slider(0, 1, value=0, label="Noise Level",
                                              info="Add Gaussian noise to the image")
                        snow_intensity = gr.Slider(0, 1, value=0, label="Snow Intensity",
                                                 info="Add snow effect to the image")

            with gr.Column(scale=2):
                with gr.Row(visible=True) as single_view:
                    output_display = gr.Image(label="Detection Output")

                with gr.Row(visible=False) as comparison_view:
                    original_display = gr.Image(label="Original Detection")
                    varied_display = gr.Image(label="Variation Detection")

                with gr.Row() as captured_view:
                    captured_original = gr.Image(label="Captured Original Frame (Viola-Jones)", visible=True)
                    captured_detection = gr.Image(label="Captured Detection Frame (Viola-Jones)", visible=True)

                with gr.Row():
                    video_output = gr.Video(label="Processed Video", visible=False)
        webcam = gr.Image(source="webcam", streaming=True, visible=True)

        def update_original_frame_label(algorithm):
            return gr.update(label=f"Captured Original Frame ({algorithm})")

        algorithm_dropdown.change(
            fn=update_original_frame_label,
            inputs=[algorithm_dropdown],
            outputs=[captured_original]
        )

        def update_detection_frame_label(algorithm):
            return gr.update(label=f"Captured  Detection Frame ({algorithm})")

        algorithm_dropdown.change(
            fn=update_detection_frame_label,
            inputs=[algorithm_dropdown],
            outputs=[captured_detection]
        )

        def update_input_method(choice):
            return {
                webcam: gr.update(visible=choice == "Webcam"),
                video_input: gr.update(visible=choice == "Video Upload"),
                process_btn: gr.update(visible=choice == "Video Upload"),
                progress_bar: gr.update(visible=choice == "Video Upload"),
                video_output: gr.update(visible=choice == "Video Upload"),
                capture_btn: gr.update(visible=choice == "Webcam"),
                output_display: gr.update(visible=choice == "Webcam"),
                captured_view: gr.update(visible=choice == "Webcam"),
                single_view: gr.update(visible=choice == "Webcam"),
                comparison_view: gr.update(visible=choice == "Webcam" and variation_dropdown.value != "None")
            }

        input_method.change(
            fn=update_input_method,
            inputs=[input_method],
            outputs=[webcam, video_input, process_btn, progress_bar, video_output,
                    capture_btn, output_display, captured_view, single_view, comparison_view]
        )

        def update_skip_frames(value):
            processor.skip_frames = int(value)

        skip_frames.change(fn=update_skip_frames, inputs=[skip_frames])

        def update_view_visibility(variation):
            return {
                single_view: gr.update(visible=variation == "None"),
                comparison_view: gr.update(visible=variation != "None")
            }

        variation_dropdown.change(
            fn=update_view_visibility,
            inputs=[variation_dropdown],
            outputs=[single_view, comparison_view]
        )

        def process_video(video_path, algorithm, variation, pose_angle, brightness,
                         contrast, temperature, noise_level, snow_intensity):
            try:
                if not video_path:
                    return None, gr.update(value=0)

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_path = "processed_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed = processor.process_frame(
                        frame_rgb, algorithm, variation,
                        pose_angle, brightness, contrast, temperature,
                        noise_level, snow_intensity
                    )

                    if isinstance(processed, tuple):
                        processed = processed[0]

                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    out.write(processed_bgr)

                    frame_count += 1
                    progress = int((frame_count / total_frames) * 100)
                    yield output_path, gr.update(value=progress)

                cap.release()
                out.release()
                yield output_path, gr.update(value=100)

            except Exception as e:
                print(f"Error processing video: {e}")
                return None, gr.update(value=0)

        def handle_frame(frame, algorithm, variation, pose_angle, brightness,
                        contrast, temperature, noise_level, snow_intensity):
            try:
                result = processor.process_frame(
                    frame, algorithm, variation,
                    pose_angle, brightness, contrast, temperature,
                    noise_level, snow_intensity
                )
                if variation == "None":
                    return result, None, None
                else:
                    return None, result[0], result[1]
            except Exception as e:
                print(f"Error in handle_frame: {e}")
                return None, None, None

        def capture_frame(frame, algorithm, pose_angle, brightness, contrast,
                         temperature, noise_level, snow_intensity):
            try:
                if frame is not None:
                    original = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    original = processor.apply_custom_variations(
                        original, pose_angle, brightness, contrast, temperature,
                        noise_level, snow_intensity
                    )
                    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

                    algorithm_obj = ALGORITHMS.get(algorithm)
                    if algorithm_obj is not None:
                        if not algorithm_obj.is_initialized:
                            algorithm_obj.initialize()
                        detections = algorithm_obj.detect(original)
                        detection_frame = algorithm_obj.draw_detections(original, detections)
                        detection_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                        return original_rgb, detection_rgb

                    return original_rgb, original_rgb
            except Exception as e:
                print(f"Error capturing frame: {e}")
                return None, None

        webcam.change(
            fn=handle_frame,
            inputs=[
                webcam, algorithm_dropdown, variation_dropdown,
                pose_angle, brightness, contrast, temperature,
                noise_level, snow_intensity
            ],
            outputs=[output_display, original_display, varied_display]
        )

        capture_btn.click(
            fn=capture_frame,
            inputs=[
                webcam, algorithm_dropdown,
                pose_angle, brightness, contrast, temperature,
                noise_level, snow_intensity
            ],
            outputs=[captured_original, captured_detection]
        )

        process_btn.click(
            fn=process_video,
            inputs=[
                video_input, algorithm_dropdown, variation_dropdown,
                pose_angle, brightness, contrast, temperature,
                noise_level, snow_intensity
            ],
            outputs=[video_output, progress_bar]
        )

    return detection_interface

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Face Detection Testing Suite")

        with gr.Tabs():
            with gr.Tab("Detection Testing"):
                gr.Markdown("## Algorithm Testing Interface")
                create_detection_interface()

            with gr.Tab("Algorithm Reference"):
                gr.Markdown("## Algorithm Performance Reference")
                create_reference_interface()

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.queue().launch(server_port=7860, share=True)