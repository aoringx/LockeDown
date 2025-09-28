# OpenCV and helper libraries imports
import sys
import os
from pathlib import Path
from queue import Queue
import cv2 as cv
import numpy as np
from memryx import AsyncAccl
import asyncio
import torchvision.ops as ops
from typing import List
import argparse
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import threading
import pyttsx3

state = ""

fast = FastAPI()

fast.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fast.get("/data")
def get_data():
    return {"state": state}

@fast.post("/data")
async def receive_data(request: Request):
    try:
        data = await request.json()
        print(f"Received POST data: {data}")
        return {"status": "success", "received": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class App:
    def __init__(self, cam, model_input_shape, mirror=False, src_is_cam=True, **kwargs):

        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.model_input_shape = model_input_shape
        self.capture_queue = Queue(maxsize=5)  
        self.mirror = mirror  
        self.box_score = 0.25  
        self.ratio = None
        self.kpt_score = 0.5  
        self.nms_thr = 0.2  
        self.src_is_cam = src_is_cam

        self.COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                   [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                   [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
                   [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
                   [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]])

        self.KEYPOINT_PAIRS = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def generate_frame(self):
        while True:
            ok, frame = self.cam.read()
            if not ok:
                print('EOF')  
                return None
            if self.src_is_cam and self.capture_queue.full():
                continue
            else:
                if self.mirror:
                    frame = cv.flip(frame, 1) 
                self.capture_queue.put(frame)  
                out, self.ratio = self.preprocess_image(frame) 
                return out

    def preprocess_image(self, image):
        h, w = image.shape[:2]
        r = min(self.model_input_shape[0] / h, self.model_input_shape[1] / w)
        image_resized = cv.resize(image, (int(w * r), int(h * r)), interpolation=cv.INTER_LINEAR)
        
        padded_img = np.ones((self.model_input_shape[0], self.model_input_shape[1], 3), dtype=np.uint8) * 114
        padded_img[:int(h * r), :int(w * r)] = image_resized

        padded_img = padded_img / 255.0
        padded_img = padded_img.astype(np.float32)
        
        padded_img = np.transpose(padded_img, (2, 0, 1)) 
        padded_img = np.expand_dims(padded_img, axis=0)  
        
        return padded_img, r

    def xywh2xyxy(self, box: np.ndarray) -> np.ndarray:
        box_xyxy = box.copy()
        box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
        box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
        box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
        box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
        return box_xyxy

    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        '''
        box and boxes are in format [x1, y1, x2, y2]
        '''
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        return inter_area / union_area  

    def nms_process(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        sorted_idx = np.argsort(scores)[::-1]  
        keep_idx = []
        while sorted_idx.size > 0:
            idx = sorted_idx[0]  
            keep_idx.append(idx)
            ious = self.compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])  
            rest_idx = np.where(ious < iou_thr)[0]  
            sorted_idx = sorted_idx[rest_idx + 1]
        return keep_idx

    def process_model_output(self, *ofmaps):
        predict = ofmaps[0].squeeze(0).T  
        predict = predict[predict[:, 4] > self.box_score, :]  
        scores = predict[:, 4]
        boxes = predict[:, 0:4] / self.ratio

        boxes = self.xywh2xyxy(boxes)  

        kpts = predict[:, 5:]
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:  
                    kpts[i, 3*j: 3*(j+1)] = [-1, -1, -1]
                else:
                    kpts[i, 3*j] /= self.ratio
                    kpts[i, 3*j+1] /= self.ratio 

        if scores.size == 0:
            img = self.capture_queue.get()
            self.capture_queue.task_done()
            self.show(img)
            state = "Not Detected"
            return img, state
        top_idx = int(np.argmax(scores))
        idxes = [top_idx]
        result = {'boxes': boxes[idxes,: ].astype(int).tolist(),
                  'kpts': kpts[idxes,: ].astype(float).tolist(),
                  'scores': scores[idxes].tolist()}

        img = self.capture_queue.get()  
        self.capture_queue.task_done()

        color = (0,255,0)
        boxes, kpts, scores = result['boxes'], result['kpts'], result['scores']
        for  kpt, score in zip(kpts, scores):
            for pair in self.KEYPOINT_PAIRS:
                pt1 = kpt[3 * pair[0]: 3 * (pair[0] + 1)]
                pt2 = kpt[3 * pair[1]: 3 * (pair[1] + 1)]
                if pt1[2] > 0 and pt2[2] > 0:
                    cv.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 255), 3)

            for idx in range(len(kpt) // 3):
                x, y, score = kpt[3*idx: 3*(idx+1)]
                if score > 0:
                    cv.circle(img, (int(x), int(y)), 5, self.COLOR_LIST[idx % len(self.COLOR_LIST)], -1)

        nose_idx, ls_idx, rs_idx = 0, 5, 6
        nx, ny, ns = kpt[3*nose_idx: 3*nose_idx + 3]
        lx, ly, ls = kpt[3*ls_idx: 3*ls_idx + 3]
        rx, ry, rs = kpt[3*rs_idx: 3*rs_idx + 3]
        angle = None
        if ns > 0 and ls > 0 and rs > 0:
            v1 = np.array([lx - nx, ly - ny], dtype=np.float32)
            v2 = np.array([rx - nx, ry - ny], dtype=np.float32)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cosang = float(np.dot(v1, v2) / (n1 * n2))
                cosang = max(-1.0, min(1.0, cosang))  
                angle = float(np.degrees(np.arccos(cosang)))
        if angle is None:
            state = "Not Detected"
        elif ny > ly or ny > ry:
            state = "Bad"
        elif angle > 135:
            state = "Warning"
        else:
            state = "Good"

        cv.putText(img, state, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

        for kp_idx, name in [(0, 'nose'), (5, 'left shoulder'), (6, 'right shoulder')]:
            x, y, sc = kpt[3*kp_idx: 3*kp_idx+3]
            if sc > 0:
                label_color = tuple(self.COLOR_LIST[kp_idx % len(self.COLOR_LIST)])
                cv.putText(
                    img,
                    name,
                    (int(x) + 6, int(y) - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    label_color,
                    2,
                    cv.LINE_AA,
                )


        self.show(img)  
        return img, state

    def show(self, img):
        cv.imshow('Output', img)
        if cv.waitKey(1) == ord('q'):  
            self.cam.release()
            cv.destroyAllWindows()
            exit(1)

def run_mxa(dfp, post_model, app):
    accl = AsyncAccl(dfp)
    accl.set_postprocessing_model(post_model, model_idx=0)
    accl.connect_input(app.generate_frame)
    last_print_time = 0.0
    last_state = None
    def _output_wrapper(*ofmaps):
        nonlocal last_print_time, last_state
        result = app.process_model_output(*ofmaps)
        if isinstance(result, tuple) and len(result) == 2:
            _, s = result
            last_state = s
            global state
            state = s
            now = time.monotonic()
            if now - last_print_time >= 1.0:
                try:
                    if(last_state == "Bad"):
                        engine = pyttsx3.init() #technically not supposed to do this every time
                        engine.say("bad posture")
                        engine.runAndWait()
                    print(last_state, flush=True)
                except Exception:
                    print("error")
                last_print_time = now
        return result

    accl.connect_output(_output_wrapper)
    accl.wait()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MX3 real-time inference")
    parser.add_argument('-d', '--dfp', type=str, default="models/YOLO_v8_medium_pose_640_640_3_onnx.dfp", help="Specify the path to the compiled DFP file. Default is 'models/YOLO_v8_medium_pose_640_640_3_onnx.dfp'.")
    parser.add_argument('-post', '--post_model', type=str, default="models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx", help="Specify the path to the post model. Default is 'models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx.")
    args = parser.parse_args()

    

    cam = cv.VideoCapture(0)
    model_input_shape = (640, 640)
    app = App(cam, model_input_shape, mirror=False, src_is_cam=True)
    dfp = args.dfp
    post_model = args.post_model
    config = uvicorn.Config(fast, host="0.0.0.0", port=5000, log_level="info")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    run_mxa(dfp, post_model, app)

