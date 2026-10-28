from fastapi import FastAPI, UploadFile, File, Response, Query
from pydantic import BaseModel
import cv2
import numpy as np
from typing import Optional

app = FastAPI(title="Vector Icon Creator & Converter API")

# Target Icon Color System (Blue #5B9BD5 in BGR format)
TARGET_BLUE = (213, 155, 91)   # BGR: Blue=213, Green=155, Red=91
WHITE = (255, 255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)

# ------------------------------------------------------------------
# ENDPOINT 1: Sketch to Vector Icon Converter
# ------------------------------------------------------------------
@app.post("/convert-sketch")
async def convert_sketch(
    file: UploadFile = File(...),
    line_thickness: int = Query(1, description="Erosion iterations: 1=standard thin, 2=extra fine")
):
    """
    Takes an uploaded hand-drawn sketch (JPG/PNG) and converts it 
    into a clean, blue vector-style icon with a transparent background.
    """
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert threshold to separate drawing lines
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Thinning (Erosion) to reduce line thickness
    if line_thickness > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=line_thickness)

    # Create transparent RGBA canvas
    h, w = gray.shape
    output = np.zeros((h, w, 4), dtype=np.uint8)

    # Colorize extracted drawing in target icon blue
    output[binary > 0, 0] = TARGET_BLUE[0] # B
    output[binary > 0, 1] = TARGET_BLUE[1] # G
    output[binary > 0, 2] = TARGET_BLUE[2] # R
    output[binary > 0, 3] = 255            # Alpha (Opaque)

    # Return PNG image bytes
    _, img_png = cv2.imencode(".png", output)
    return Response(content=img_png.tobytes(), media_type="image/png")


# ------------------------------------------------------------------
# ENDPOINT 2: Direct Icon Creator (Programmatic / Presets)
# ------------------------------------------------------------------
@app.get("/create-icon")
async def create_icon(
    pose: str = Query("warrior", description="Options: warrior, lotus, deadlift, surf"),
    canvas_size: int = Query(1000, description="Canvas resolution in pixels"),
    ring_thickness: int = Query(35, description="Outer ring stroke width")
):
    """
    Programmatically builds a pristine vector icon from scratch 
    matching the exact blue icon design system.
    """
    size = canvas_size
    center = (size // 2, size // 2)
    radius = int(size * 0.44)
    
    # Initialize 4-channel transparent canvas (RGBA)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    
    # Blue color in BGRA
    blue_rgba = (TARGET_BLUE[0], TARGET_BLUE[1], TARGET_BLUE[2], 255)
    
    # 1. Draw Outer Circle Frame
    cv2.circle(canvas, center, radius, blue_rgba, ring_thickness, lineType=cv2.LINE_AA)

    # 2. Draw Ground / Mat Line
    ground_y = int(size * 0.70)
    mat_left = int(size * 0.25)
    mat_right = int(size * 0.75)
    cv2.line(canvas, (mat_left, ground_y), (mat_right, ground_y), blue_rgba, int(ring_thickness * 0.7), lineType=cv2.LINE_AA)

    # 3. Render Requested Pose / Icon Graphic
    head_radius = int(size * 0.065)
    stroke = int(ring_thickness * 0.8)

    if pose.lower() in ["warrior", "yoga"]:
        # Head (Ring style)
        head_center = (center[0], int(size * 0.32))
        cv2.circle(canvas, head_center, head_radius, blue_rgba, stroke, lineType=cv2.LINE_AA)
        
        # Torso & Arms
        cv2.line(canvas, (center[0], head_center[1] + head_radius), (center[0], int(size * 0.52)), blue_rgba, stroke, lineType=cv2.LINE_AA)
        cv2.line(canvas, (int(size * 0.38), int(size * 0.44)), (int(size * 0.62), int(size * 0.44)), blue_rgba, stroke, lineType=cv2.LINE_AA)
        
        # Warrior Stance Legs
        cv2.line(canvas, (center[0], int(size * 0.52)), (int(size * 0.41), int(size * 0.58)), blue_rgba, stroke, lineType=cv2.LINE_AA)
        cv2.line(canvas, (int(size * 0.41), int(size * 0.58)), (int(size * 0.41), ground_y), blue_rgba, stroke, lineType=cv2.LINE_AA)
        cv2.line(canvas, (center[0], int(size * 0.52)), (int(size * 0.64), ground_y), blue_rgba, stroke, lineType=cv2.LINE_AA)

    elif pose.lower() == "lotus":
        # Lotus Meditation Pose
        head_center = (center[0], int(size * 0.36))
        cv2.circle(canvas, head_center, head_radius, blue_rgba, stroke, lineType=cv2.LINE_AA)
        
        # Torso
        cv2.line(canvas, (center[0], head_center[1] + head_radius), (center[0], int(size * 0.56)), blue_rgba, stroke, lineType=cv2.LINE_AA)
        
        # Folded Legs
        pts_left = np.array([[center[0], int(size * 0.56)], [int(size * 0.32), int(size * 0.65)], [center[0], ground_y]], np.int32)
        pts_right = np.array([[center[0], int(size * 0.56)], [int(size * 0.68), int(size * 0.65)], [center[0], ground_y]], np.int32)
        cv2.polylines(canvas, [pts_left], False, blue_rgba, stroke, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts_right], False, blue_rgba, stroke, lineType=cv2.LINE_AA)

    elif pose.lower() == "deadlift":
        # Weightlifting Pose
        head_center = (int(size * 0.42), int(size * 0.35))
        cv2.circle(canvas, head_center, head_radius, blue_rgba, stroke, lineType=cv2.LINE_AA)
        
        # Barbell & Weights
        cv2.line(canvas, (int(size * 0.20), int(size * 0.54)), (int(size * 0.80), int(size * 0.54)), blue_rgba, stroke, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, (int(size * 0.22), int(size * 0.44)), (int(size * 0.26), int(size * 0.64)), blue_rgba, -1)
        cv2.rectangle(canvas, (int(size * 0.74), int(size * 0.44)), (int(size * 0.78), int(size * 0.64)), blue_rgba, -1)

    # Encode and return PNG
    _, img_png = cv2.imencode(".png", canvas)
    return Response(content=img_png.tobytes(), media_type="image/png")
