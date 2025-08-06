# watermark_remover
Multi-GPU enabled fast batch removal of watermarks from large image datasets

Key Architectural Features:
- Multi GPU (AI) and CPU (I/O) workloads for true parallelism and high throughput.
- Resumable processing via a `.processing_log.txt` checkpoint file.
- Per-GPU status tracking in a pretty Rich console display

--------------------------------------------------------------------
USAGE INSTRUCTIONS
--------------------------------------------------------------------
1. Install required Python packages:
   pip install rich ultralytics simple-lama-inpainting opencv-python torch --upgrade

2. Download the required YOLOv11 model checkpoint from Hugging Face:
   (See https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection)
   - On Linux/macOS, run this in your terminal:
     wget https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt

3. Run the script:
   python3 watermark_remover_final.py -i /path/to/inputs -o /path/to/outputs -R
