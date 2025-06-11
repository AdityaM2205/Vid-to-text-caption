import cv2
import decord
import numpy as np
from vllm import LLM
from pathlib import Path
import base64

def extract_frames(video_path, frame_interval=30):
    """Extract video frames at specified intervals using decord"""
    vr = decord.VideoReader(video_path)
    return [vr[i].asnumpy() for i in range(0, len(vr), frame_interval)]

def process_video(video_path, output_file="captions.parquet"):
    # Frame extraction
    frames = extract_frames(video_path)
    
    # Initialize VLM (Qwen2-VL-2B-Instruct)
    vlm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})
    
    captions = []
    for i in range(0, len(frames), 4):
        batch_frames = frames[i:i+4]
        
        # Build multi-image prompt
        message = {
            "role": "user",
            "content": [
                {"type": "text": "Describe these video frames in detail:"},
                *[{"type": "image": "data:image/jpeg;base64," + 
                    base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()} 
                 for frame in batch_frames]
            ]
        }
        
        # Generate captions
        outputs = vlm.chat([message])
        captions.extend([o.outputs[0].text for o in outputs])
    
    # LLM summarization (Llama-3-8B-Instruct)
    llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
    summary_prompt = f"""
    Combine these frame captions into a coherent video description:
    {chr(10).join(captions)}
    
    Provide a concise summary focusing on main subjects, actions, and scene changes.
    """
    
    summary = llm.generate([summary_prompt])[0].outputs[0].text
    
    # Save results
    Path(output_file).write_text(summary)
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", default="captions.txt", 
                       help="Output file path")
    args = parser.parse_args()
    
    result = process_video(args.video_path, args.output)
    print(f"Video description saved to {args.output}:")
    print(result)
  
