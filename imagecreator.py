from fastapi import FastAPI, UploadFile, File, Response
import cv2
import numpy as np

app = FastAPI()


TARGET_BLUE = (213, 155, 91)  

@app.post("/remove-bg")
async def convert_sketch_to_icon(file: UploadFile = File(...)):
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

 
    thinning_kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, thinning_kernel, iterations=1)

    h, w = gray.shape
    output = np.zeros((h, w, 4), dtype=np.uint8)

    output[binary > 0, 0] = TARGET_BLUE[0] 
    output[binary > 0, 1] = TARGET_BLUE[1] 
    output[binary > 0, 2] = TARGET_BLUE[2] 
    output[binary > 0, 3] = 255            


    _, img_png = cv2.imencode(".png", output)

    return Response(content=img_png.tobytes(), media_type="image/png")
