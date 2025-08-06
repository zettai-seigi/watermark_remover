# watermark_remover
Multi-GPU enabled fast batch removal of watermarks from large image datasets

Key Architectural Features:
- Multi GPU (AI) and CPU (I/O) workloads for high throughput.
- Pause/resume processing at any time
- Per-GPU performance/status tracking in a pretty console display

On a dual RTX 4090 machine, you can de-watermark over 1000 images per minute. ⚡😎⚡

<img width="1083" height="485" alt="image" src="https://github.com/user-attachments/assets/093ce1ac-e364-4dbb-bf33-0d47dc4cad5f" />

--------------------------------------------------------------------
INSTALL INSTRUCTIONS
--------------------------------------------------------------------
1. Clone the repository:
   git clone https://github.com/jferments/watermark_remover.git
2. Enter the project directory
   cd watermark_remover   
3. Install required Python packages
    python -m venv venv
    source venv/bin/activate
    pip install rich ultralytics simple-lama-inpainting opencv-python torch --upgrade
4. Download [fancyfeast](https://huggingface.co/fancyfeast)'s custom [YOLOv11 watermark detection model](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection) checkpoint from Hugging Face:
   - On Linux/macOS, run this in your terminal:
     wget https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt

--------------------------------------------------------------------
USAGE INSTRUCTIONS
--------------------------------------------------------------------

**Basic usage:**

    python3 watermark_remover_final.py -i /path/to/inputs -o /path/to/outputs -R


To run the script, you must provide an input and an output directory.

* **To process a single folder:**
    ```bash
    python3 watermark_remover_final.py -i /path/to/your/images -o /path/to/save/clean_images
    ```
* **To process a folder and all its subdirectories:**
    ```bash
    python3 watermark_remover_final.py -i /path/to/your/images -o /path/to/save/clean_images -R
    ```

---

**Pausing and Resuming:**

For very large datasets, you can safely stop the script at any time by pressing **`Ctrl+C`**. The script will perform a graceful shutdown, save its progress, and print a summary for the session.

When you run the script again with the **exact same output directory**, it will automatically detect the `.processing_log.txt` checkpoint file and resume where it left off, skipping any images that were already successfully processed.

---

**Command-Line Arguments**

Here is a detailed explanation of all available arguments:

* **`-i, --input`** `<path>` **(Required)** Specifies the path to the folder containing the images you want to process.

* **`-o, --output`** `<path>` **(Required)** Specifies the path to the folder where the clean, processed images will be saved. The original directory structure from the input will be replicated here. This directory also stores the `.processing_log.txt` file for resuming sessions.

* **`-w, --weights`** `<path>`  
    Specifies the path to the YOLOv11 model weights file.  
    (Default: `yolo11x-train28-best.pt`)

* **`--conf`** `<float>`  
    The confidence threshold for the YOLO object detection model (from `0.0` to `1.0`). Lower values will detect more potential watermarks but may also have more false positives.  
    (Default: `0.1`)

* **`--dilate`** `<integer>`  
    The number of pixels to expand (dilate) the detected watermark mask. This is useful for ensuring the inpainting model covers any faint "glow" or aliasing around the edges of a watermark. Set to `0` to disable.  
    (Default: `15`)

* **`-R, --recursive`** A flag that, if present, tells the script to search for images in all subdirectories of the input folder. If omitted, it will only process images in the top-level directory.

* **`--cpu-workers`** `<integer>`  
    The total number of CPU processes to spawn for the I/O-bound task of writing image files to disk. By default, it uses all available CPU cores to maximize I/O throughput.  
    (Default: Your system's CPU core count)

* **`--debug`** A flag that, if present, will save two intermediate images for each detected watermark into an `output/debug/` directory:
    1.  `_mask_raw.png`: The raw black and white mask.
    2.  `_mask_preview.png`: The mask overlaid in semi-transparent red on the original image.


Enjoy!
