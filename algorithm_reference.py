import gradio as gr

# Detailed algorithm capabilities and limitations
ALGORITHM_FEATURES = {
    "Pose Variations": {
        "Profile View": {
            "Viola-Jones": "❌ No support",
            "HOG + SVM": "❌ No support",
            "MediaPipe": "⚠️ Limited support",
            "MTCNN": "✅ Good support",
            "RetinaFace": "✅ Excellent support",
            "YOLO": "✅ Good support"
        },
        "Extreme Angles": {
            "Viola-Jones": "❌ Fails beyond ±30°",
            "HOG + SVM": "⚠️ Works up to ±45°",
            "MediaPipe": "✅ Works up to ±60°",
            "MTCNN": "✅ Works up to ±75°",
            "RetinaFace": "✅ Works up to ±90°",
            "YOLO": "✅ Works up to ±70°"
        },
        "Head Tilt": {
            "Viola-Jones": "❌ Very sensitive",
            "HOG + SVM": "⚠️ Moderate tolerance",
            "MediaPipe": "✅ Good tolerance",
            "MTCNN": "✅ High tolerance",
            "RetinaFace": "✅ Excellent tolerance",
            "YOLO": "✅ High tolerance"
        }
    },
    "Lighting Conditions": {
        "Low Light": {
            "Viola-Jones": "❌ Poor performance",
            "HOG + SVM": "⚠️ Moderate performance",
            "MediaPipe": "✅ Good performance",
            "MTCNN": "✅ Very good performance",
            "RetinaFace": "✅ Excellent performance",
            "YOLO": "✅ Very good performance"
        },
        "Harsh Shadows": {
            "Viola-Jones": "❌ Fails frequently",
            "HOG + SVM": "⚠️ Some tolerance",
            "MediaPipe": "✅ Good tolerance",
            "MTCNN": "✅ High tolerance",
            "RetinaFace": "✅ Excellent tolerance",
            "YOLO": "✅ Good tolerance"
        },
        "Backlighting": {
            "Viola-Jones": "❌ No support",
            "HOG + SVM": "⚠️ Limited support",
            "MediaPipe": "⚠️ Moderate support",
            "MTCNN": "✅ Good support",
            "RetinaFace": "✅ Very good support",
            "YOLO": "✅ Good support"
        }
    },
    "Occlusions": {
        "Face Masks": {
            "Viola-Jones": "❌ No support",
            "HOG + SVM": "⚠️ Poor support",
            "MediaPipe": "✅ Good support",
            "MTCNN": "✅ Very good support",
            "RetinaFace": "✅ Excellent support",
            "YOLO": "✅ Good support"
        },
        "Glasses": {
            "Viola-Jones": "⚠️ Basic support",
            "HOG + SVM": "✅ Good support",
            "MediaPipe": "✅ Very good support",
            "MTCNN": "✅ Excellent support",
            "RetinaFace": "✅ Excellent support",
            "YOLO": "✅ Very good support"
        },
        "Partial Face Coverage": {
            "Viola-Jones": "❌ No support",
            "HOG + SVM": "⚠️ Limited support",
            "MediaPipe": "✅ Good support",
            "MTCNN": "✅ Very good support",
            "RetinaFace": "✅ Excellent support",
            "YOLO": "✅ Good support"
        }
    },
    "Performance Considerations": {
        "CPU Speed": {
            "Viola-Jones": "✅ Very fast (30+ FPS)",
            "HOG + SVM": "✅ Fast (20+ FPS)",
            "MediaPipe": "⚠️ Moderate (15+ FPS)",
            "MTCNN": "⚠️ Slow (10+ FPS)",
            "RetinaFace": "❌ Very slow (5+ FPS)",
            "YOLO": "✅ Fast (20+ FPS)"
        },
        "Memory Usage": {
            "Viola-Jones": "✅ Very light",
            "HOG + SVM": "✅ Light",
            "MediaPipe": "⚠️ Moderate",
            "MTCNN": "⚠️ Heavy",
            "RetinaFace": "❌ Very heavy",
            "YOLO": "⚠️ Moderate"
        },
        "Initialization Time": {
            "Viola-Jones": "✅ Instant",
            "HOG + SVM": "✅ Quick",
            "MediaPipe": "⚠️ Moderate",
            "MTCNN": "⚠️ Slow",
            "RetinaFace": "❌ Very slow",
            "YOLO": "⚠️ Moderate"
        }
    },
    "Special Features": {
        "Multi-Face Detection": {
            "Viola-Jones": "✅ Supported",
            "HOG + SVM": "✅ Supported",
            "MediaPipe": "✅ Supported",
            "MTCNN": "✅ Supported",
            "RetinaFace": "✅ Supported",
            "YOLO": "✅ Excellent support"
        },
        "Small Face Detection": {
            "Viola-Jones": "❌ Poor with small faces",
            "HOG + SVM": "⚠️ Moderate support",
            "MediaPipe": "✅ Good support",
            "MTCNN": "✅ Very good support",
            "RetinaFace": "✅ Excellent support",
            "YOLO": "✅ Very good support"
        },
        "GPU Acceleration": {
            "Viola-Jones": "❌ Not available",
            "HOG + SVM": "❌ Limited support",
            "MediaPipe": "✅ Good support",
            "MTCNN": "✅ Good support",
            "RetinaFace": "✅ Excellent support",
            "YOLO": "✅ Excellent support"
        }
    }
}

def create_feature_comparison(feature_category):
    """Create a comparison table for a specific feature category"""
    features = ALGORITHM_FEATURES[feature_category]
    algorithms = ["Viola-Jones", "HOG + SVM", "MediaPipe", "MTCNN", "RetinaFace", "YOLO"]
    
    with gr.Blocks() as comparison:
        for feature, performance in features.items():
            gr.Markdown(f"### {feature}")
            with gr.Row():
                for algo in algorithms:
                    with gr.Column():
                        gr.Markdown(f"**{algo}**")
                        gr.Markdown(performance[algo])
    return comparison

def create_reference_interface():
    """Create the algorithm reference and testing dimensions interface"""
    with gr.Blocks() as reference_interface:
        gr.Markdown("# Algorithm Capabilities Comparison")
        
        with gr.Tabs() as tabs:
            # Pose Variations Tab
            with gr.Tab("Pose Variations"):
                gr.Markdown("""
                ### Pose Handling Capabilities
                Comparison of how each algorithm handles different face poses and orientations.
                """)
                create_feature_comparison("Pose Variations")
                
                gr.Markdown("""
                #### Key Notes:
                - Profile view support is crucial for wide-angle applications
                - Extreme angles often occur in real-world scenarios
                - Head tilt tolerance affects usability in casual settings
                """)
            
            # Lighting Conditions Tab
            with gr.Tab("Lighting Conditions"):
                gr.Markdown("""
                ### Lighting Handling Capabilities
                Comparison of performance under various lighting conditions.
                """)
                create_feature_comparison("Lighting Conditions")
                
                gr.Markdown("""
                #### Key Notes:
                - Low light performance is critical for indoor/evening use
                - Shadow handling affects outdoor performance
                - Backlight tolerance is important for window-facing cameras
                """)
            
            # Occlusions Tab
            with gr.Tab("Occlusions"):
                gr.Markdown("""
                ### Occlusion Handling
                Comparison of how algorithms handle various types of face occlusions.
                """)
                create_feature_comparison("Occlusions")
                
                gr.Markdown("""
                #### Key Notes:
                - Mask detection has become crucial in recent years
                - Glasses handling is important for everyday use
                - Partial occlusion tolerance affects real-world reliability
                """)
            
            # Performance Tab
            with gr.Tab("Performance"):
                gr.Markdown("""
                ### Performance Characteristics
                Comparison of computational requirements and performance metrics.
                """)
                create_feature_comparison("Performance Considerations")
                
                gr.Markdown("""
                #### Key Notes:
                - CPU speed affects real-time performance on standard hardware
                - Memory usage is important for mobile/embedded applications
                - Initialization time impacts application start-up
                """)

            # Special Features Tab
            with gr.Tab("Special Features"):
                gr.Markdown("""
                ### Special Capabilities
                Comparison of additional features and specialized capabilities.
                """)
                create_feature_comparison("Special Features")
                
                gr.Markdown("""
                #### Key Notes:
                - Multi-face detection is important for group scenarios
                - Small face detection matters for surveillance/crowd monitoring
                - GPU acceleration enables faster processing on supported hardware
                """)

    return reference_interface