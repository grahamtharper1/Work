from fastapi import FastAPI, UploadFile, File, Response, Query
import cv2
import numpy as np

app = FastAPI(title="Vector Icon Creator & Converter API")

# Target Icon Color System (Blue #5B9BD5 in BGR format for OpenCV)
TARGET_BLUE = (213, 155, 91)   # BGR: Blue=213, Green=155, Red=91


@app.get("/")
async def root():
    return {"status": "ok", "message": "Sketch Converter API is running"}


# ------------------------------------------------------------------
# ENDPOINT: Sketch to Vector Icon Converter
# ------------------------------------------------------------------
@app.post("/convert-sketch")
@app.post("/remove-bg")
async def convert_sketch(
    file: UploadFile = File(...),
    line_thickness: int = Query(1, description="Erosion iterations: 1=standard thin, 2=extra fine")
):
    """
    Takes an uploaded hand-drawn sketch (JPG/PNG), normalizes brightness,
    isolates ink using Otsu's thresholding, smooths jagged hand-drawn edges, 
    and converts it into a blue vector-style icon with a transparent background.
    """
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content=b"Invalid image file", status_code=400)

    # 1. Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Normalize intensity (stretches dark ink toward 0 and paper background toward 255)
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 3. Smooth minor paper texture noise
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

    # 4. Otsu's Automatic Thresholding (Finds exact split point between paper & ink)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. NEW: Clean up and smooth the drawing lines
    # Morphological closing with an elliptical kernel fills tiny holes and connects broken stroke fragments
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth_kernel, iterations=2)
    
    # Blur and re-threshold to anti-alias and round out wobbly, jagged edges
    binary = cv2.GaussianBlur(binary, (5, 5), 0)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    # 6. Thinning (Erosion) to keep vector lines crisp and delicate
    if line_thickness > 0:
        erode_kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, erode_kernel, iterations=line_thickness)

    # 7. Create transparent 4-channel RGBA canvas
    h, w = gray.shape
    output = np.zeros((h, w, 4), dtype=np.uint8)

    # 8. Colorize extracted drawing lines in target icon blue (#5B9BD5)
    output[binary > 0, 0] = TARGET_BLUE[0]  # Blue channel
    output[binary > 0, 1] = TARGET_BLUE[1]  # Green channel
    output[binary > 0, 2] = TARGET_BLUE[2]  # Red channel
    output[binary > 0, 3] = 255             # Alpha channel (Opaque)

    # 9. Encode and return output image as PNG
    _, img_png = cv2.imencode(".png", output)
    return Response(content=img_png.tobytes(), media_type="image/png")
