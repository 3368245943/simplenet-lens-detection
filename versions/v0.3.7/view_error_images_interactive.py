#!/usr/bin/env python
"""Interactive one-by-one viewer for three separately configured sensor folders."""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from infer_sensor_model import preprocess
ROOT = Path(__file__).resolve().parent
# ===================== 用户配置区 =====================
MODEL_PATH = ROOT / '/home/bai/project/INDEMIND/深度数据/镜头脏污预警/simplenet-lens-detection/results/LensDirt_Results/stereo_aggregation/run/sensor_model_normal_q95_cleaned_reexport_opset11.onnx'
IMAGE_DIR_RGB = None
IMAGE_DIR_STEREO =  Path('/media/bai/D27A57537A573407/datatset/datasets/负样本/室内/光照良好/灰尘/大量/depth_data/img/left/')
IMAGE_DIR_TOF = Path('/media/bai/D27A57537A573407/datatset/datasets/正样本/室内/强光/depth_data/tof/img/')
THRESHOLDS = {
'rgb': 1.6450339555740356, 
'stereo': 147.0, 
'tof': 2.686849355697632
}
# ======================================================
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
SENSORS = ('rgb', 'stereo', 'tof')
def is_sensor_image(path, sensor):
    parts = path.as_posix().split('/')
    if sensor == 'rgb':
        return 'cam0' in parts and 'depth_data' not in parts
    if sensor == 'stereo':
        return 'depth_data' in parts and 'img' in parts and ('left' in parts or 'right' in parts)
    if sensor == 'tof':
        return 'depth_data' in parts and 'tof' in parts and 'img' in parts
    return False

def collect_images(directory, sensor):
    if directory is None: return []
    directory = Path(directory).expanduser()
    if not directory.is_dir(): raise FileNotFoundError(f'{sensor} 图片目录不存在: {directory}')
    return [(path, sensor) for path in sorted(directory.rglob('*'))
            if path.is_file() and path.suffix.lower() in EXTENSIONS and is_sensor_image(path, sensor)]
def main():
    if not MODEL_PATH.is_file(): raise FileNotFoundError(MODEL_PATH)
    configured = [('rgb', IMAGE_DIR_RGB), ('stereo', IMAGE_DIR_STEREO), ('tof', IMAGE_DIR_TOF)]
    directories = [Path(d).expanduser().resolve() for _, d in configured if d is not None]
    if len(directories) != len(set(directories)): raise ValueError('RGB、双目、ToF 不能填写同一个目录，请分别填写三路目录')
    images = sum((collect_images(d, s) for s, d in configured), [])
    counts = {sensor: len(collect_images(directory, sensor)) for sensor, directory in configured}
    print('三路图片数量:', counts, flush=True)
    if not images: raise RuntimeError('三个配置目录中都没有符合对应传感器结构的图片')
    session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    index = 0; window = '单图实时查看 | N下一张 P上一张 Q退出'; cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        path, sensor = images[index]
        score, _ = session.run(None, {'image': preprocess(str(path)), 'sensor_id': np.array([SENSORS.index(sensor)], dtype=np.int64)})
        score = float(score[0]); threshold = float(THRESHOLDS[sensor]); prediction = 'ANOMALY' if score >= threshold else 'NORMAL'; color = (0, 0, 220) if prediction == 'ANOMALY' else (0, 150, 0)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: raise RuntimeError(f'无法读取图片: {path}')
        header = np.full((100, image.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(header, f'{index + 1}/{len(images)}  {sensor.upper()}  {prediction}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
        cv2.putText(header, f'score={score:.4f}  threshold={threshold:.4f}', (12, 63), cv2.FONT_HERSHEY_SIMPLEX, .58, (35, 45, 55), 1, cv2.LINE_AA)
        cv2.putText(header, str(path), (12, 88), cv2.FONT_HERSHEY_SIMPLEX, .42, (35, 45, 55), 1, cv2.LINE_AA)
        cv2.imshow(window, np.vstack((header, image))); key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), ord('Q'), 27): break
        if key in (ord('n'), ord('N'), ord(' '), 13): index = (index + 1) % len(images)
        elif key in (ord('p'), ord('P')): index = (index - 1) % len(images)
    cv2.destroyAllWindows()
if __name__ == '__main__': main()
