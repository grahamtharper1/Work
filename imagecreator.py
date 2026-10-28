from fastapi import FastAPI, UploadFile, File, Response, Query
import cv2
import numpy as np

app = FastAPI(title="Vector Icon Creator & Converter API")

TARGET_BLUE = (213, 155, 91)   # BGR: Blue=213, Green=155, Red=91

@app.get("/")
async def root():
    return {"status": "ok", "message": "Hybrid Sketch Converter API is running"}

@app.post("/convert-sketch")
@app.post("/remove-bg")
async def convert_sketch(
    file: UploadFile = File(...),
    line_thickness: int = Query(1, description="Erosion iterations: 1=standard thin, 2=extra fine")
):
    """
    HYBRID APPROACH: Draws a perfect outer circle, tightly masks out the original 
    messy ring, and applies balanced smoothing to bridge minor gaps without 
    melting independent lines together.
    """
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content=b"Invalid image file", status_code=400)

    # 1. Grayscale & Normalize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 2. Otsu's Thresholding to extract ink
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Find the center and radius based on the total ink footprint
    y_coords, x_coords = np.where(binary > 0)
    if len(y_coords) == 0:
        return Response(content=b"No drawing detected", status_code=400)

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    center = (cx, cy)
    radius = max(x_max - x_min, y_max - y_min) // 2

    # 4. DIGITAL ERASER: Tight mask to cut out the messy hand-drawn ring
    interior_mask = np.zeros_like(binary)
    cv2.circle(interior_mask, center, int(radius * 0.75), 255, -1)
    
    interior_ink = cv2.bitwise_and(binary, interior_mask)

    # 5. BALANCED SMOOTHING (No extreme dilation)
    # Moderate closing to bridge the small gaps in the head/legs without fusing the whole body
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    interior_ink = cv2.morphologyEx(interior_ink, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    
    # Moderate blur to smooth edges, not melt them
    interior_ink = cv2.GaussianBlur(interior_ink, (11, 11), 0)
    
    # Standard threshold to snap back to solid vector-like lines
    _, interior_ink = cv2.threshold(interior_ink, 127, 255, cv2.THRESH_BINARY)

    # Restore normal line thickness 
    if line_thickness > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        interior_ink = cv2.erode(interior_ink, erode_kernel, iterations=line_thickness)

    # 6. Build Final Canvas (Transparent 4-channel RGBA)
    h, w = gray.shape
    output = np.zeros((h, w, 4), dtype=np.uint8)

    # Draw the smoothed interior figure in blue
    output[interior_ink > 0, 0] = TARGET_BLUE[0]  
    output[interior_ink > 0, 1] = TARGET_BLUE[1]  
    output[interior_ink > 0, 2] = TARGET_BLUE[2]  
    output[interior_ink > 0, 3] = 255             

    # 7. Draw a PERFECT mathematical outer circle
    blue_rgba = (TARGET_BLUE[0], TARGET_BLUE[1], TARGET_BLUE[2], 255)
    circle_thickness = max(4, int(radius * 0.05))
    cv2.circle(output, center, int(radius * 0.95), blue_rgba, circle_thickness, lineType=cv2.LINE_AA)

    # Encode and return PNG
    _, img_png = cv2.imencode(".png", output)
    return Response(content=img_png.tobytes(), media_type="image/png")
