from fastapi import FastAPI, UploadFile, File, Response, Query
import cv2
import numpy as np

app = FastAPI(title="Vector Icon Creator & Converter API")

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
    Takes an uploaded hand-drawn sketch (JPG/PNG), isolates ink using Otsu's 
    thresholding, applies aggressive circular smoothing to eliminate wobbles, 
    and converts it into a blue vector-style icon with a transparent background.
    """
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content=b"Invalid image file", status_code=400)

    # 1. Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Normalize intensity 
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 3. Smooth minor paper texture noise
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

    # 4. Otsu's Automatic Thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. AGGRESSIVE SMOOTHING: Force perfectly round, vector-like edges
    # Use a larger circular kernel to aggressively fill gaps and round off jagged turns
    round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, round_kernel, iterations=3)
    
    # Heavy Gaussian blur to "melt" the wobbly edges into smooth gradients
    binary = cv2.GaussianBlur(binary, (11, 11), 0)
    
    # Hard re-threshold to snap the melted gradients back into crisp, perfectly smooth lines
    _, binary = cv2.threshold(binary, 140, 255, cv2.THRESH_BINARY)

    # 6. Thinning (Erosion) to restore delicate thickness using a circular kernel
    if line_thickness > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Add +1 iteration to compensate for the thickening caused by the heavy blur stage
        binary = cv2.erode(binary, erode_kernel, iterations=line_thickness + 1)

    # 7. Create transparent 4-channel RGBA canvas
    h, w = gray.shape
    output = np.zeros((h, w, 4), dtype=np.uint8)

    # 8. Colorize extracted drawing lines in target icon blue (#5B9BD5)
    output[binary > 0, 0] = TARGET_BLUE[0]  
    output[binary > 0, 1] = TARGET_BLUE[1]  
    output[binary > 0, 2] = TARGET_BLUE[2]  
    output[binary > 0, 3] = 255             

    # 9. Encode and return output image as PNG
    _, img_png = cv2.imencode(".png", output)
    return Response(content=img_png.tobytes(), media_type="image/png")
