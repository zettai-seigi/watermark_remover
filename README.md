# watermark_remover
Multi-GPU enabled fast batch removal of watermarks from large image datasets

Key Architectural Features:
- Multi GPU (AI) and CPU (I/O) workloads for high throughput.
- Pause/resume processing at any time
- Per-GPU performance/status tracking in a pretty console display

On a dual RTX 4090 machine, you can de-watermark over 1000 images per minute. ⚡😎⚡

--------------------------------------------------------------------
INSTALL INSTRUCTIONS
--------------------------------------------------------------------
1. Install required Python packages:
   pip install rich ultralytics simple-lama-inpainting opencv-python torch --upgrade

2. Download [fancyfeast](https://huggingface.co/fancyfeast)'s custom [YOLOv11 watermark detection model](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection) checkpoint from Hugging Face:
   - On Linux/macOS, run this in your terminal:
     wget https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt

--------------------------------------------------------------------
USAGE INSTRUCTIONS
--------------------------------------------------------------------

Run the script in terminal:
   python3 watermark_remover_final.py -i /path/to/inputs -o /path/to/outputs -R

Enjoy!

<img width="1122" height="462" alt="image" src="https://github.com/user-attachments/assets/6c0b6b34-e33d-498a-a9f9-4e6f0604ffb1" />
