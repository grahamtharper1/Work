from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI()

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    h, w = gray.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = white_mask.copy()

    for x in range(w):
        if filled[0, x] == 255: cv2.floodFill(filled, ff_mask, (x, 0), 128)
        if filled[h - 1, x] == 255: cv2.floodFill(filled, ff_mask, (x, h - 1), 128)
    for y in range(h):
        if filled[y, 0] == 255: cv2.floodFill(filled, ff_mask, (0, y), 128)
        if filled[y, w - 1] == 255: cv2.floodFill(filled, ff_mask, (w - 1, y), 128)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[filled == 128, 3] = 0

    _, encoded_img = cv2.imencode('.png', rgba)
    return Response(content=encoded_img.tobytes(), media_type="image/png")
