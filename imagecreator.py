from fastapi import FastAPI, UploadFile, File, Response, Query
import cv2
import numpy as np

app = FastAPI(title="Vector Icon Creator & Converter API")

TARGET_BLUE = (213, 155, 91)   # BGR: Blue=213, Green=155, Red=91

@app.get("/")
async def root():
    return {"status": "ok", "message": "Dynamic Sketch Converter API is running"}


@app.post("/convert-sketch")
@app.post("/remove-bg")
async def convert_sketch(
    file: UploadFile = File(...),
    line_thickness: int = Query(1, description="Erosion iterations: 1=standard thin, 2=extra fine")
):
    """
    Dynamically processes the uploaded hand-drawn sketch (JPG/PNG), isolates ink 
    using Otsu's thresholding, applies a smoothing filter to the hand-drawn edges, 
    and converts the unique drawing into a blue icon with a transparent background.
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

    # 5. DYNAMIC SMOOTHING: Smooth out wobbly ink while keeping the drawing's original shape
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth_kernel, iterations=2)
    
    binary = cv2.GaussianBlur(binary, (7, 7), 0)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    # 6. Thinning (Erosion) to restore delicate line thickness
    if line_thickness > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.erode(binary, erode_kernel, iterations=line_thickness)

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
