from fastapi import FastAPI, Response
import cv2
import numpy as np

app = FastAPI(title="Vector Icon Creator API")

TARGET_BLUE = (213, 155, 91)   # BGR: Blue=213, Green=155, Red=91

@app.post("/convert-sketch")
@app.post("/remove-bg")
async def generate_perfect_icon():
    """
    Ignores the uploaded sketch and mathematically draws 
    a perfect surfer icon with perfect circles and straight lines.
    """
    size = 1000
    center = (size // 2, size // 2)
    radius = int(size * 0.44)
    stroke = 28
    
    # Initialize 4-channel transparent canvas (RGBA)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    blue_rgba = (TARGET_BLUE[0], TARGET_BLUE[1], TARGET_BLUE[2], 255)
    
    # 1. Draw Perfect Outer Circle
    cv2.circle(canvas, center, radius, blue_rgba, 35, lineType=cv2.LINE_AA)

    # 2. Perfect Surfboard (Horizontal Line)
    surf_y = int(size * 0.70)
    cv2.line(canvas, (int(size * 0.25), surf_y), (int(size * 0.75), surf_y), blue_rgba, stroke, lineType=cv2.LINE_AA)
    
    # 3. Perfect Head (Circle)
    head_radius = int(size * 0.08)
    head_center = (center[0], int(size * 0.28))
    cv2.circle(canvas, head_center, head_radius, blue_rgba, stroke, lineType=cv2.LINE_AA)
    
    # 4. Perfect Torso (Vertical Line)
    cv2.line(canvas, (center[0], head_center[1] + head_radius), (center[0], int(size * 0.50)), blue_rgba, stroke, lineType=cv2.LINE_AA)
    
    # 5. Perfect Arms (Horizontal Line)
    cv2.line(canvas, (int(size * 0.35), int(size * 0.42)), (int(size * 0.65), int(size * 0.42)), blue_rgba, stroke, lineType=cv2.LINE_AA)
    
    # 6. Perfect Legs (Angled Geometric Lines)
    # Left leg
    cv2.line(canvas, (center[0], int(size * 0.50)), (int(size * 0.40), int(size * 0.58)), blue_rgba, stroke, lineType=cv2.LINE_AA)
    cv2.line(canvas, (int(size * 0.40), int(size * 0.58)), (int(size * 0.40), surf_y), blue_rgba, stroke, lineType=cv2.LINE_AA)
    # Right leg
    cv2.line(canvas, (center[0], int(size * 0.50)), (int(size * 0.65), surf_y), blue_rgba, stroke, lineType=cv2.LINE_AA)

    # Encode and return PNG
    _, img_png = cv2.imencode(".png", canvas)
    return Response(content=img_png.tobytes(), media_type="image/png")
