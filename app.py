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
        self.skip_frames = 2  # Process every nth frame
        self.last_frame = None
        
    def process_frame(self, frame: np.ndarray, 
                     algorithm_name: str,
                     variation_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Process a single frame with selected algorithm and variation"""
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
                    return result if variation_name == "None" else (result, result)
                
                if not algorithm.is_initialized:
                    algorithm.initialize()
                
                detections = algorithm.detect(frame)
                result_frame = algorithm.draw_detections(frame, detections)
                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                if variation_name == "None":
                    self.last_frame = result_rgb
                    return result_rgb
                
                # Apply variation and process varied frame
                varied_frame = VARIATIONS[variation_name](frame)
                varied_detections = algorithm.detect(varied_frame)
                varied_result = algorithm.draw_detections(varied_frame, varied_detections)
                varied_rgb = cv2.cvtColor(varied_result, cv2.COLOR_BGR2RGB)
                
                result = (result_rgb, varied_rgb)
                self.last_frame = result
                return result
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Return original frame in case of error
                backup_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return backup_frame if variation_name == "None" else (backup_frame, backup_frame)

def create_detection_interface():
    """Create the face detection testing interface"""
    processor = VideoProcessor()
    
    with gr.Blocks() as detection_interface:
        with gr.Row():
            with gr.Column(scale=1):
                # Input methods
                input_method = gr.Radio(
                    choices=["Webcam", "Video Upload"],
                    label="Input Method",
                    value="Webcam"
                )
                
                # Algorithm selection
                algorithm_dropdown = gr.Dropdown(
                    choices=list(ALGORITHMS.keys()),
                    label="Select Algorithm",
                    value=list(ALGORITHMS.keys())[0] if ALGORITHMS else None
                )
                
                # Variation selection
                variation_dropdown = gr.Dropdown(
                    choices=list(VARIATIONS.keys()),
                    label="Select Variation",
                    value="None"
                )

                # Performance settings
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
            
            with gr.Column(scale=2):
                with gr.Row(visible=True) as single_view:
                    output_display = gr.Image(label="Detection Output")
                
                with gr.Row(visible=False) as comparison_view:
                    original_display = gr.Image(label="Original Detection")
                    varied_display = gr.Image(label="Variation Detection")
        
        webcam = gr.Image(source="webcam", streaming=True)
        
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
        
        # Process frames
        def handle_frame(frame, algorithm, variation):
            try:
                result = processor.process_frame(frame, algorithm, variation)
                if variation == "None":
                    return result, None, None
                else:
                    return None, result[0], result[1]
            except Exception as e:
                print(f"Error in handle_frame: {e}")
                return None, None, None
        
        webcam.change(
            fn=handle_frame,
            inputs=[webcam, algorithm_dropdown, variation_dropdown],
            outputs=[output_display, original_display, varied_display]
        )

    return detection_interface

def create_interface():
    """Create the main interface with both detection and reference tabs"""
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
    # Create and launch interface
    interface = create_interface()
    interface.launch(server_port=7860, share=True)