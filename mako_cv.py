"""
mako_cv.py
==========
MakoVision — Mako 桌宠的"视觉皮层"模块 (v3.0 - HD Servo Zoom Edition)

v3.0 核心升级：高清伺服云台 + 动态 ROI 裁切
  - 摄像头分辨率提升至 1080P（硬件支持时）
  - 保留原始高清帧 self._raw_hd_frame 供裁切使用
  - 新增 mp.solutions.hands 手部关键点检测
  - 新增三个动态高清裁切 API：
    · get_eye_roi_image()   — 眼部特写（1.5x padding）
    · get_hands_roi_image() — 手部特写（2.0x padding）
    · get_face_roi_image()  — 人脸特写（1.3x padding）
  - 所有裁切坐标带有安全边界检查

Live2D 面捕正统四角眼神范围校准与归一化视线映射算法。

算法流程:
  1. MediaPipe 478 点 → 虹膜比率 (rh, rv)
  2. EMA 平滑 → (rh_smooth, rv_smooth)
  3. 四角校准 → 极值边界 (rh_min, rh_max, rv_min, rv_max)
  4. 双线性归一化 → Screen_X_Ratio, Screen_Y_Ratio (0.0 ~ 1.0)
  5. 乘以屏幕分辨率 → 绝对像素落点
  6. 检查是否在 Mako 窗口内 → 饱和积分状态机
  7. 高清帧持久化 + Hands 并行检测 → ROI 裁切就绪

测试入口:
    python mako_cv.py

依赖:
    opencv-python, mediapipe, numpy
"""

from __future__ import annotations

import time
import math
import cv2
import numpy as np
from PIL import Image

import mediapipe as mp
print("[Debug] MediaPipe actual path:", mp.__file__)


# ============================================================
# MediaPipe Face Mesh Keypoints (refine_landmarks=True, 478pts)
# ============================================================
LEFT_IRIS  = 468
RIGHT_IRIS = 473

LEFT_EYE_OUTER   = 33
LEFT_EYE_INNER   = 133
LEFT_EYE_TOP     = 159
LEFT_EYE_BOTTOM  = 145

RIGHT_EYE_OUTER   = 263
RIGHT_EYE_INNER   = 362
RIGHT_EYE_TOP     = 386
RIGHT_EYE_BOTTOM  = 374

NOSE_TIP     = 4
CHIN         = 152
LEFT_MOUTH   = 61
RIGHT_MOUTH  = 291


# ============================================================
# Standard 3D Head Model (mm, right-hand coord)
# ============================================================
HEAD_3D_MODEL = np.array([
    (0.0,   0.0,   0.0),     # nose tip    (4)
    (0.0,  -330.0, -65.0),   # chin        (152)
    (-225.0, 170.0, -135.0), # left eye    (33)
    (225.0,  170.0, -135.0), # right eye   (263)
    (-150.0, -150.0, -125.0),# left mouth  (61)
    (150.0, -150.0, -125.0), # right mouth (291)
], dtype=np.float64)

HEAD_MODEL_INDICES = [
    NOSE_TIP, CHIN,
    LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
    LEFT_MOUTH, RIGHT_MOUTH,
]


# ============================================================
# Default Config
# ============================================================
DEFAULT_EMA_ALPHA         = 0.25
DEFAULT_SCORE_THRESH      = 20    # STARING trigger
DEFAULT_SCORE_RESET       = 5     # STARING release
DEFAULT_SCORE_HIT         = 3     # hit increment
DEFAULT_SCORE_MISS        = -1    # miss increment
DEFAULT_HEAD_ANGLE_MAX    = 20.0  # kept for reference only

SCREEN_RES_X = 1920
SCREEN_RES_Y = 1080

# ---- 裁切 Padding 默认倍率 ----
EYE_ROI_PADDING   = 1.5   # 眼部：将眉毛和鼻梁包入
HANDS_ROI_PADDING = 2.0   # 手部：将键盘/手腕背景包入
FACE_ROI_PADDING  = 1.3   # 人脸：适度外扩


class MakoVision:
    """
    MakoVision v3.0 — HD Servo Zoom Edition.

    v3.0 新增能力：
      - 摄像头分辨率自动拉升到 1080P（硬件允许时）
      - 原始高清帧持久化（self._raw_hd_frame）
      - MediaPipe Hands 并行检测
      - 三个动态高清 ROI 裁切 API
    """

    def __init__(self, camera_id: int = 0):
        # ============================================================
        # 1. 摄像头初始化 + 分辨率升级（1080P 优先）
        # ============================================================
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera (camera_id={camera_id})."
            )

        # 尝试设置 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 如果实际分辨率未达到 Full HD，尝试降级到 720P
        if actual_w < 1280:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._cam_w = actual_w
        self._cam_h = actual_h
        print(f"[MakoVision v3.0] Camera resolution: {actual_w}x{actual_h}")

        # ============================================================
        # 2. HD 原始帧持久化（裁切底片）
        # ============================================================
        self._raw_hd_frame = None

        # ============================================================
        # 3. MediaPipe FaceMesh（面部 478 点）
        # ============================================================
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ============================================================
        # 4. MediaPipe Hands（手部 21 点）— v3.0 新增
        # ============================================================
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ---- 最新帧的检测结果缓存（供 ROI 裁切 API 使用） ----
        self._latest_face_landmarks = None      # face mesh landmarks (478 pts)
        self._latest_hand_results = None         # hands detection results

        # ---- EMA smooth buffers ----
        self._prev_rh    = 0.5
        self._prev_rv    = 0.5
        self._prev_yaw   = 0.0
        self._prev_pitch = 0.0

        # ---- Calibration data ----
        self.is_calibrated    = False
        self._calib_bounds    = None
        self._calib_samples   = []

        # ---- Mako window bounding box ----
        self._mako_win_x = 800
        self._mako_win_y = 400
        self._mako_win_w = 300
        self._mako_win_h = 200

        # ---- State machine ----
        self._gaze_score = 0
        self._is_staring = False

        print("[MakoVision v3.0] Initialized"
              " (HD Servo Zoom + FaceMesh + Hands)")

    # ---------------------------------------------------------------
    # EMA smoother
    # ---------------------------------------------------------------
    def _ema(self, new_val: float, prev_val: float, alpha: float = DEFAULT_EMA_ALPHA) -> float:
        return alpha * new_val + (1.0 - alpha) * prev_val

    # ---------------------------------------------------------------
    # Landmark to pixel
    # ---------------------------------------------------------------
    def _lm(self, landmarks, idx: int, w: int, h: int) -> tuple:
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    # ---------------------------------------------------------------
    # Gaze estimation (iris ratio)
    # ---------------------------------------------------------------
    def _estimate_gaze(self, landmarks, w: int, h: int):
        # Left eye
        l_iris_x, l_iris_y   = self._lm(landmarks, LEFT_IRIS, w, h)
        l_inner_x, _         = self._lm(landmarks, LEFT_EYE_INNER, w, h)
        l_outer_x, _         = self._lm(landmarks, LEFT_EYE_OUTER, w, h)
        _, l_top_y           = self._lm(landmarks, LEFT_EYE_TOP, w, h)
        _, l_bottom_y        = self._lm(landmarks, LEFT_EYE_BOTTOM, w, h)

        l_eye_width   = max(l_outer_x - l_inner_x, 1)
        l_eye_height  = max(l_bottom_y - l_top_y, 1)

        l_rh = (l_iris_x - l_inner_x) / l_eye_width
        l_rv = (l_iris_y - l_top_y)   / l_eye_height

        # Right eye
        r_iris_x, r_iris_y   = self._lm(landmarks, RIGHT_IRIS, w, h)
        r_inner_x, _         = self._lm(landmarks, RIGHT_EYE_INNER, w, h)
        r_outer_x, _         = self._lm(landmarks, RIGHT_EYE_OUTER, w, h)
        _, r_top_y           = self._lm(landmarks, RIGHT_EYE_TOP, w, h)
        _, r_bottom_y        = self._lm(landmarks, RIGHT_EYE_BOTTOM, w, h)

        r_eye_width   = max(r_outer_x - r_inner_x, 1)
        r_eye_height  = max(r_bottom_y - r_top_y, 1)

        r_rh = (r_iris_x - r_inner_x) / r_eye_width
        r_rv = (r_iris_y - r_top_y)   / r_eye_height

        rh = max(0.0, min(1.0, (l_rh + r_rh) / 2.0))
        rv = max(0.0, min(1.0, (l_rv + r_rv) / 2.0))

        return rh, rv

    # ---------------------------------------------------------------
    # Head pose (kept for visualization only, not used in logic)
    # ---------------------------------------------------------------
    def _estimate_head_pose(self, landmarks, w: int, h: int, camera_matrix=None):
        image_points = []
        for idx in HEAD_MODEL_INDICES:
            x, y = self._lm(landmarks, idx, w, h)
            image_points.append((x, y))
        image_points = np.array(image_points, dtype=np.float64)

        if camera_matrix is None:
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0,             center[0]],
                [0,             focal_length, center[1]],
                [0,             0,             1        ],
            ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            HEAD_3D_MODEL, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, None, None, camera_matrix

        rot_mat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rot_mat[2, 0] ** 2 + rot_mat[2, 1] ** 2)
        if sy < 1e-6:
            yaw   = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
            pitch = math.atan2(-rot_mat[2, 0], sy)
        else:
            yaw   = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
            pitch = math.atan2(-rot_mat[2, 0], rot_mat[2, 2])

        yaw   = math.degrees(yaw)
        pitch = math.degrees(pitch)

        return yaw, pitch, rvec, tvec, camera_matrix

    # ================================================================
    # Calibration API
    # ================================================================

    def reset_calibration(self):
        """Reset calibration data."""
        self.is_calibrated  = False
        self._calib_bounds  = None
        self._calib_samples = []

    def add_calibration_sample(self, rh: float, rv: float):
        """Record one calibration sample point."""
        self._calib_samples.append((rh, rv))

    def finalize_calibration(self):
        """
        Compute calibration bounds from 4 corner samples.

        Expected sample order:
          0: Top-Left     (look at screen top-left)
          1: Top-Right    (look at screen top-right)
          2: Bottom-Right (look at screen bottom-right)
          3: Bottom-Left  (look at screen bottom-left)
        """
        if len(self._calib_samples) < 4:
            print(f"[MakoVision] Calibration failed: only {len(self._calib_samples)} samples, need 4")
            return False

        rh_vals = [s[0] for s in self._calib_samples[:4]]
        rv_vals = [s[1] for s in self._calib_samples[:4]]

        rh_min = min(rh_vals)
        rh_max = max(rh_vals)
        rv_min = min(rv_vals)
        rv_max = max(rv_vals)

        rh_margin = (rh_max - rh_min) * 0.05
        rv_margin = (rv_max - rv_min) * 0.05

        self._calib_bounds = {
            "rh_min": max(0.0, rh_min - rh_margin),
            "rh_max": min(1.0, rh_max + rh_margin),
            "rv_min": max(0.0, rv_min - rv_margin),
            "rv_max": min(1.0, rv_max + rv_margin),
        }

        self.is_calibrated = True
        print(f"[MakoVision] Calibration OK: rh=[{self._calib_bounds['rh_min']:.3f}, {self._calib_bounds['rh_max']:.3f}], "
              f"rv=[{self._calib_bounds['rv_min']:.3f}, {self._calib_bounds['rv_max']:.3f}]")
        return True

    # ---------------------------------------------------------------
    # Bilinear Mapping: (rh, rv) → Screen Ratio (0~1, 0~1)
    # ---------------------------------------------------------------
    def _gaze_to_screen_ratio(self, rh: float, rv: float):
        if not self.is_calibrated or self._calib_bounds is None:
            return None, None, False

        b = self._calib_bounds
        is_in_bounds = (b["rh_min"] <= rh <= b["rh_max"]) and (b["rv_min"] <= rv <= b["rv_max"])

        rh_range = b["rh_max"] - b["rh_min"]
        rv_range = b["rv_max"] - b["rv_min"]

        if rh_range < 1e-6 or rv_range < 1e-6:
            return 0.5, 0.5, False

        ratio_x = (rh - b["rh_min"]) / rh_range
        ratio_y = (rv - b["rv_min"]) / rv_range

        ratio_x = max(0.0, min(1.0, ratio_x))
        ratio_y = max(0.0, min(1.0, ratio_y))

        return ratio_x, ratio_y, is_in_bounds

    # ---------------------------------------------------------------
    # Set Mako window bounding box (screen pixel coords)
    # ---------------------------------------------------------------
    def set_mako_window(self, x: int, y: int, w: int, h: int):
        """Set the on-screen bounding box of Mako's pet window."""
        self._mako_win_x = x
        self._mako_win_y = y
        self._mako_win_w = w
        self._mako_win_h = h

    # ---------------------------------------------------------------
    # State machine (simplified, head-angle independent)
    # ---------------------------------------------------------------
    def _update_gaze_fsm(self, is_looking_screen: bool,
                         looking_at_mako: bool = False) -> bool:
        if is_looking_screen and looking_at_mako:
            self._gaze_score = min(30, self._gaze_score + DEFAULT_SCORE_HIT)
        elif not is_looking_screen:
            self._gaze_score = max(0, self._gaze_score + DEFAULT_SCORE_MISS)

        if not self._is_staring and self._gaze_score > DEFAULT_SCORE_THRESH:
            self._is_staring = True
        elif self._is_staring and self._gaze_score < DEFAULT_SCORE_RESET:
            self._is_staring = False

        return self._is_staring

    # ================================================================
    # Main entry: check_presence
    # ================================================================
    def check_presence(self):
        """
        实时人脸检测 + 视线追踪 + HD 帧持久化 + Hands 并行检测。

        Returns:
            (result_dict, frame_or_None)
            result_dict 保持不变，新增字段：
            "has_hands": bool  — 是否检测到手
        """
        result = {
            "is_present": False,
            "face_count": 0,
            "faces": [],
            "gaze_rh":      0.5,
            "gaze_rv":      0.5,
            "head_yaw":     0.0,
            "head_pitch":   0.0,
            "screen_ratio_x": None,
            "screen_ratio_y": None,
            "screen_pixel_x": None,
            "screen_pixel_y": None,
            "is_looking_screen": False,
            "looking_at_mako": False,
            "gaze_score":   0,
            "is_staring":   self._is_staring,
            "landmarks_raw": [],
            "rvec":         None,
            "tvec":         None,
            "timestamp":    time.time(),
            "is_calibrated": self.is_calibrated,
            "calib_bounds":  self._calib_bounds,
            # ---- v3.0 新增字段 ----
            "has_hands":    False,
        }

        ret, frame = self.cap.read()
        if not ret:
            return result, None

        # ---- ⭐ v3.0 核心：保存原始高清帧的深拷贝 ----
        self._raw_hd_frame = frame.copy()

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ============================================================
        # A. FaceMesh 面部检测
        # ============================================================
        mp_results = self.face_mesh.process(rgb_frame)

        if mp_results.multi_face_landmarks is None or len(mp_results.multi_face_landmarks) == 0:
            self._latest_face_landmarks = None
            self._gaze_score = max(0, self._gaze_score + DEFAULT_SCORE_MISS)
            if self._is_staring and self._gaze_score < DEFAULT_SCORE_RESET:
                self._is_staring = False
            result["gaze_score"] = self._gaze_score
            result["is_staring"] = self._is_staring
        else:
            landmarks = mp_results.multi_face_landmarks[0].landmark
            self._latest_face_landmarks = landmarks

            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            result["is_present"] = True
            result["face_count"] = 1
            result["faces"] = [{
                "x": (min_x + max_x) / 2.0,
                "y": (min_y + max_y) / 2.0,
                "width":  max_x - min_x,
                "height": max_y - min_y,
                "confidence": 1.0,
            }]

            # Gaze estimation
            rh_raw, rv_raw = self._estimate_gaze(landmarks, w, h)
            yaw_raw, pitch_raw, rvec, tvec, _ = self._estimate_head_pose(landmarks, w, h)

            rh_smooth    = self._ema(rh_raw,   self._prev_rh)
            rv_smooth    = self._ema(rv_raw,   self._prev_rv)
            yaw_smooth   = self._ema(yaw_raw,  self._prev_yaw)
            pitch_smooth = self._ema(pitch_raw, self._prev_pitch)

            self._prev_rh    = rh_smooth
            self._prev_rv    = rv_smooth
            self._prev_yaw   = yaw_smooth
            self._prev_pitch = pitch_smooth

            result["gaze_rh"]    = rh_smooth
            result["gaze_rv"]    = rv_smooth
            result["head_yaw"]   = yaw_smooth
            result["head_pitch"] = pitch_smooth
            result["landmarks_raw"] = [(lm.x, lm.y) for lm in landmarks]
            result["rvec"] = rvec
            result["tvec"] = tvec

            # Calibrated screen mapping
            ratio_x, ratio_y, in_bounds = self._gaze_to_screen_ratio(rh_smooth, rv_smooth)
            if ratio_x is not None and self.is_calibrated:
                px = ratio_x * SCREEN_RES_X
                py = ratio_y * SCREEN_RES_Y
                result["screen_ratio_x"] = ratio_x
                result["screen_ratio_y"] = ratio_y
                result["screen_pixel_x"] = px
                result["screen_pixel_y"] = py
                result["is_looking_screen"] = in_bounds
                mx, my, mw, mh = self._mako_win_x, self._mako_win_y, self._mako_win_w, self._mako_win_h
                result["looking_at_mako"] = (mx <= px <= mx + mw) and (my <= py <= my + mh)
            else:
                result["is_looking_screen"] = False
                result["looking_at_mako"]   = False

            is_staring = self._update_gaze_fsm(
                result["is_looking_screen"],
                result["looking_at_mako"],
            )
            result["gaze_score"] = self._gaze_score
            result["is_staring"] = is_staring

        # ============================================================
        # B. Hands 手部检测 — v3.0 新增（与 FaceMesh 在同一帧 RGB 上运行）
        # ============================================================
        hand_results = self.hands.process(rgb_frame)
        self._latest_hand_results = hand_results

        if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) > 0:
            result["has_hands"] = True

        return result, frame

    # ================================================================
    # v3.0 核心武器：三个动态高清 ROI 裁切 API
    # ================================================================
    # ---- 工具方法：安全边界检查 + 带 padding 的 ROI 裁切 ----
    # ---------------------------------------------------------------

    def _expand_and_clamp_roi(self, x1: int, y1: int, x2: int, y2: int,
                              padding: float, img_w: int, img_h: int) -> tuple:
        """
        对 ROI 进行 padding 外扩 + 边界检查，返回 (x1, y1, x2, y2)。

        Parameters
        ----------
        x1, y1 : int
            左上角坐标。
        x2, y2 : int
            右下角坐标。
        padding : float
            外扩倍率（1.0 = 不扩，1.5 = 向外扩 50%）。
        img_w, img_h : int
            原始图像的宽度和高度。

        Returns
        -------
        tuple[int, int, int, int]
            (x1, y1, x2, y2) 安全 clamp 后的绝对坐标。
        """
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        half_w = int(((x2 - x1) // 2) * padding)
        half_h = int(((y2 - y1) // 2) * padding)

        x1 = max(0, center_x - half_w)
        y1 = max(0, center_y - half_h)
        x2 = min(img_w, center_x + half_w)
        y2 = min(img_h, center_y + half_h)

        # 确保最小裁切尺寸（防止极端情况）
        if x2 - x1 < 32:
            x1 = max(0, center_x - 16)
            x2 = min(img_w, center_x + 16)
        if y2 - y1 < 32:
            y1 = max(0, center_y - 16)
            y2 = min(img_h, center_y + 16)

        return x1, y1, x2, y2

    # ---------------------------------------------------------------
    # API 1: get_eye_roi_image — 眼部高清特写
    # ---------------------------------------------------------------

    def get_eye_roi_image(self) -> Image.Image | None:
        """
        从最新高清帧中裁切眼部区域（含眉毛 + 部分鼻梁）。

        使用 FaceMesh 的眼部关键点计算 BBox：
          - 左眼: inner/outer/top/bottom
          - 右眼: inner/outer/top/bottom
          - 取两眼的联合区域
        默认 1.5x padding 确保眉毛和鼻梁包入。

        Returns
        -------
        PIL.Image.Image | None
            裁切后的眼区域高清 PIL 图像；若无人脸或无高清帧则返回 None。
        """
        if self._raw_hd_frame is None:
            return None
        if self._latest_face_landmarks is None:
            return None

        h, w = self._raw_hd_frame.shape[:2]
        lm = self._latest_face_landmarks

        def _px(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # 左眼外/内/上/下
        l_ox, l_oy = _px(LEFT_EYE_OUTER)
        l_ix, l_iy = _px(LEFT_EYE_INNER)
        l_tx, l_ty = _px(LEFT_EYE_TOP)
        l_bx, l_by = _px(LEFT_EYE_BOTTOM)

        # 右眼外/内/上/下
        r_ox, r_oy = _px(RIGHT_EYE_OUTER)
        r_ix, r_iy = _px(RIGHT_EYE_INNER)
        r_tx, r_ty = _px(RIGHT_EYE_TOP)
        r_bx, r_by = _px(RIGHT_EYE_BOTTOM)

        # 两眼的联合 BBox
        all_x = [l_ox, l_ix, r_ox, r_ix, l_tx, l_bx, r_tx, r_bx]
        all_y = [l_oy, l_iy, r_oy, r_iy, l_ty, l_by, r_ty, r_by]

        x1 = min(all_x)
        y1 = min(all_y)
        x2 = max(all_x)
        y2 = max(all_y)

        # 1.5x padding + 边界 clamp
        x1, y1, x2, y2 = self._expand_and_clamp_roi(
            x1, y1, x2, y2, EYE_ROI_PADDING, w, h
        )

        # 从高清帧裁切
        roi = self._raw_hd_frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return Image.fromarray(roi_rgb)

    # ---------------------------------------------------------------
    # API 2: get_hands_roi_image — 手部高清特写
    # ---------------------------------------------------------------

    def get_hands_roi_image(self) -> Image.Image | None:
        """
        从最新高清帧中裁切手部区域。

        使用 MediaPipe Hands 结果：
          - 如果有两只手，取包含两手的最小外接矩形
          - 默认 2.0x padding 确保手腕/键盘背景包入

        Returns
        -------
        PIL.Image.Image | None
            裁切后的手部区域高清 PIL 图像；若未检测到手或无高清帧则返回 None。
        """
        if self._raw_hd_frame is None:
            return None
        if self._latest_hand_results is None:
            return None
        if not self._latest_hand_results.multi_hand_landmarks:
            return None

        h, w = self._raw_hd_frame.shape[:2]

        # 收集所有手的 landmark 坐标
        all_x = []
        all_y = []
        for hand_landmarks in self._latest_hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)
                all_x.append(px)
                all_y.append(py)

        if not all_x or not all_y:
            return None

        x1 = min(all_x)
        y1 = min(all_y)
        x2 = max(all_x)
        y2 = max(all_y)

        # 2.0x padding + 边界 clamp
        x1, y1, x2, y2 = self._expand_and_clamp_roi(
            x1, y1, x2, y2, HANDS_ROI_PADDING, w, h
        )

        roi = self._raw_hd_frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return Image.fromarray(roi_rgb)

    # ---------------------------------------------------------------
    # API 3: get_face_roi_image — 人脸高清特写
    # ---------------------------------------------------------------

    def get_face_roi_image(self) -> Image.Image | None:
        """
        从最新高清帧中裁切人脸区域。

        使用 FaceMesh 的全部 landmark 计算 BBox：
          - 取所有 478 个 landmark 的 min/max
          - 默认 1.3x padding 适度外扩

        Returns
        -------
        PIL.Image.Image | None
            裁切后的人脸高清 PIL 图像；若无人脸或无高清帧则返回 None。
        """
        if self._raw_hd_frame is None:
            return None
        if self._latest_face_landmarks is None:
            return None

        h, w = self._raw_hd_frame.shape[:2]
        lm = self._latest_face_landmarks

        xs = [lm[i].x * w for i in range(len(lm))]
        ys = [lm[i].y * h for i in range(len(lm))]

        x1 = int(min(xs))
        y1 = int(min(ys))
        x2 = int(max(xs))
        y2 = int(max(ys))

        # 1.3x padding + 边界 clamp
        x1, y1, x2, y2 = self._expand_and_clamp_roi(
            x1, y1, x2, y2, FACE_ROI_PADDING, w, h
        )

        roi = self._raw_hd_frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return Image.fromarray(roi_rgb)

    # ---------------------------------------------------------------
    # API 4: get_latest_frame_as_pil — 获取最新全景帧
    # ---------------------------------------------------------------

    def get_latest_frame_as_pil(self) -> Image.Image | None:
        """
        获取最新的原始全景摄像头帧作为 PIL 图像（默认摄像头画面基底）。

        Returns
        -------
        PIL.Image.Image | None
            全景摄像头 PIL 图像；若无可用的高清帧则返回 None。
        """
        if self._raw_hd_frame is None:
            return None
        rgb = cv2.cvtColor(self._raw_hd_frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------
    @property
    def is_staring(self) -> bool:
        return self._is_staring

    @property
    def gaze_score(self) -> int:
        return self._gaze_score

    @property
    def camera_resolution(self) -> tuple:
        """返回摄像头 (宽, 高)。"""
        return (self._cam_w, self._cam_h)

    # ================================================================
    # Resource management
    # ================================================================
    def release(self):
        if hasattr(self, "face_mesh") and self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
            print("[MakoVision] MediaPipe FaceMesh released")

        if hasattr(self, "hands") and self.hands is not None:
            self.hands.close()
            self.hands = None
            print("[MakoVision] MediaPipe Hands released")

        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[MakoVision] Camera released")

        self._raw_hd_frame = None
        self._latest_face_landmarks = None
        self._latest_hand_results = None
        print("[MakoVision v3.0] All resources released")

    def __del__(self):
        self.release()


# ============================================================
# Test Entry: OpenCV Live Preview + ROI 裁切测试
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MakoVision v3.0 Test Mode -- HD Servo Zoom Preview")
    print("  [C] Calibrate  [Space] Sample  [Q] Quit")
    print("  [E] Eye ROI crop  [H] Hands ROI crop  [F] Face ROI crop")
    print("=" * 70)

    print("\nInitializing MakoVision v3.0...")
    vision = MakoVision(camera_id=0)
    print(f"  Camera resolution: {vision.camera_resolution[0]}x{vision.camera_resolution[1]}")
    print("OK. Starting camera preview.\n")

    WINDOW_NAME = "MakoVision v3.0 - HD Servo Zoom Preview"

    # ---- Render colors ----
    COLOR_MESH     = (0, 200, 0)
    COLOR_IRIS     = (255, 100, 0)
    COLOR_HANDS    = (100, 255, 100)  # light green for hands
    COLOR_AXIS_X   = (0, 0, 255)
    COLOR_AXIS_Y   = (0, 255, 0)
    COLOR_AXIS_Z   = (255, 0, 255)
    COLOR_HUD_OK   = (0, 255, 0)
    COLOR_HUD_RED  = (0, 0, 255)
    COLOR_DOT      = (0, 100, 255)
    COLOR_CALIB    = (0, 255, 255)
    COLOR_ROI_BOX  = (0, 255, 255)  # cyan ROI bounding box

    # ---- Screen mini-map ----
    MINI_W = 120
    MINI_H = 75
    MINI_Y = 10

    # ---- Calibration state ----
    in_calibration = False
    calib_step     = 0
    calib_corners  = [
        "LOOK at TOP-LEFT corner of screen",
        "LOOK at TOP-RIGHT corner of screen",
        "LOOK at BOTTOM-RIGHT corner of screen",
        "LOOK at BOTTOM-LEFT corner of screen",
    ]
    calib_corner_short = ["TL", "TR", "BR", "BL"]

    # ---- 3D axis for visualization ----
    AXIS_LEN = 150
    axis_pts_3d = np.array([
        (0, 0, 0), (AXIS_LEN, 0, 0), (0, AXIS_LEN, 0), (0, 0, AXIS_LEN),
    ], dtype=np.float64)

    # ---- Mako window (simulated for test) ----
    VISION_MAKO_X = 100
    VISION_MAKO_Y = 100
    VISION_MAKO_W = 300
    VISION_MAKO_H = 200
    vision.set_mako_window(VISION_MAKO_X, VISION_MAKO_Y, VISION_MAKO_W, VISION_MAKO_H)

    # ---- Main loop ----
    try:
        frame_count = 0
        while True:
            frame_count += 1
            data, frame = vision.check_presence()

            if frame is None:
                time.sleep(0.03)
                continue

            h, w = frame.shape[:2]
            is_present  = data["is_present"]
            is_staring  = data["is_staring"]
            is_calib    = data.get("is_calibrated", False)
            has_hands   = data.get("has_hands", False)

            # ---- Camera matrix for axis projection ----
            focal = w
            cx = w / 2.0
            cy = h / 2.0
            cam_mat = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1],
            ], dtype=np.float64)
            dist = np.zeros((4, 1), dtype=np.float64)

            # ============================================================
            # A. Face mesh + iris (v3.0: also draw hand landmarks)
            # ============================================================
            if is_present:
                lms = data.get("landmarks_raw", [])
                for lx, ly in lms:
                    px, py = int(lx * w), int(ly * h)
                    cv2.circle(frame, (px, py), 1, COLOR_MESH, -1)

                if len(lms) > RIGHT_IRIS:
                    lix = int(lms[LEFT_IRIS][0] * w)
                    liy = int(lms[LEFT_IRIS][1] * h)
                    cv2.circle(frame, (lix, liy), 5, COLOR_IRIS, -1)

                    rix = int(lms[RIGHT_IRIS][0] * w)
                    riy = int(lms[RIGHT_IRIS][1] * h)
                    cv2.circle(frame, (rix, riy), 5, COLOR_IRIS, -1)

                # 3D axis
                rv = data.get("rvec")
                tv = data.get("tvec")
                if rv is not None and tv is not None:
                    proj, _ = cv2.projectPoints(axis_pts_3d, rv, tv, cam_mat, dist)
                    nose = tuple(proj[0].ravel().astype(int))
                    ax_x = tuple(proj[1].ravel().astype(int))
                    ax_y = tuple(proj[2].ravel().astype(int))
                    ax_z = tuple(proj[3].ravel().astype(int))
                    cv2.line(frame, nose, ax_x, COLOR_AXIS_X, 2)
                    cv2.line(frame, nose, ax_y, COLOR_AXIS_Y, 2)
                    cv2.line(frame, nose, ax_z, COLOR_AXIS_Z, 3)
                    cv2.circle(frame, ax_z, 5, COLOR_AXIS_Z, -1)

            # Draw hand landmarks (v3.0)
            if has_hands and vision._latest_hand_results:
                for hand_landmarks in vision._latest_hand_results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (px, py), 3, COLOR_HANDS, -1)
                    # Connect hand landmarks with lines
                    for connection in mp.solutions.hands.HAND_CONNECTIONS:
                        start = hand_landmarks.landmark[connection[0]]
                        end = hand_landmarks.landmark[connection[1]]
                        sx, sy = int(start.x * w), int(start.y * h)
                        ex, ey = int(end.x * w), int(end.y * h)
                        cv2.line(frame, (sx, sy), (ex, ey), COLOR_HANDS, 1)

            # ============================================================
            # B. HUD Panel (upper-left)
            # ============================================================
            if is_present:
                stat_col = COLOR_HUD_RED if is_staring else COLOR_HUD_OK

                overlay = frame.copy()
                cv2.rectangle(overlay, (8, 8), (360, 250), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.rectangle(frame, (8, 8), (360, 250), stat_col, 2)

                score = data["gaze_score"]
                yaw   = data["head_yaw"]
                pitch = data["head_pitch"]
                rh    = data["gaze_rh"]
                rv    = data["gaze_rv"]

                cv2.putText(frame, f"Score: {score:>2d}/30",      (18, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Yaw: {yaw:>+6.1f}  Pitch: {pitch:>+6.1f}", (18, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Rh: {rh:.3f}  Rv: {rv:.3f}", (18, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Hands: {'YES' if has_hands else 'no'}", (18, 101),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                if is_calib:
                    rx = data.get("screen_ratio_x", 0.0)
                    ry = data.get("screen_ratio_y", 0.0)
                    px = data.get("screen_pixel_x", 0.0)
                    py = data.get("screen_pixel_y", 0.0)
                    looking = data.get("is_looking_screen", False)
                    at_mako = data.get("looking_at_mako", False)

                    cv2.putText(frame, f"Ratio: ({rx:.2f}, {ry:.2f})", (18, 124),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    cv2.putText(frame, f"Pixel: ({int(px):>4d}, {int(py):>4d})", (18, 147),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                    screen_label = "Screen: YES" if looking else "Screen: no"
                    cv2.putText(frame, screen_label, (18, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                    mako_label = "MakoWin: HIT" if at_mako else "MakoWin: ---"
                    cv2.putText(frame, mako_label, (18, 193),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "NOT CALIBRATED - press [C]", (18, 124),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

                stare_label = "STARING" if is_staring else "---"
                cv2.putText(frame, stare_label, (18, 225),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, stat_col, 2)

            else:
                cv2.putText(frame, "No face detected", (18, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            # ============================================================
            # C. Calibration overlay
            # ============================================================
            if in_calibration:
                step = calib_step
                ctext = f"CALIBRATION STEP {step+1}/4: {calib_corners[step]}"
                cv2.putText(frame, ctext, (w // 2 - 250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CALIB, 2)

                if step == 0:
                    cv2.drawMarker(frame, (50, 50), COLOR_CALIB, cv2.MARKER_CROSS, 40, 3)
                elif step == 1:
                    cv2.drawMarker(frame, (w - 50, 50), COLOR_CALIB, cv2.MARKER_CROSS, 40, 3)
                elif step == 2:
                    cv2.drawMarker(frame, (w - 50, h - 50), COLOR_CALIB, cv2.MARKER_CROSS, 40, 3)
                elif step == 3:
                    cv2.drawMarker(frame, (50, h - 50), COLOR_CALIB, cv2.MARKER_CROSS, 40, 3)

                cv2.putText(frame, "Look at the crosshair and press SPACE to sample",
                            (w // 2 - 220, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CALIB, 2)

            # ============================================================
            # D. Screen mini-map
            # ============================================================
            ms_x1 = w - MINI_W - 15
            ms_y1 = MINI_Y
            ms_x2 = w - 15
            ms_y2 = MINI_Y + MINI_H

            ov2 = frame.copy()
            cv2.rectangle(ov2, (ms_x1, ms_y1), (ms_x2, ms_y2), (0, 0, 0), -1)
            cv2.addWeighted(ov2, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (ms_x1, ms_y1), (ms_x2, ms_y2), (200, 200, 200), 1)

            cv2.putText(frame, "Screen", (ms_x1 + 5, ms_y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            mk_rel_x = int((VISION_MAKO_X / SCREEN_RES_X) * MINI_W)
            mk_rel_y = int((VISION_MAKO_Y / SCREEN_RES_Y) * MINI_H)
            mk_rel_w = int((VISION_MAKO_W / SCREEN_RES_X) * MINI_W)
            mk_rel_h = int((VISION_MAKO_H / SCREEN_RES_Y) * MINI_H)
            cv2.rectangle(frame,
                          (ms_x1 + mk_rel_x, ms_y1 + mk_rel_y),
                          (ms_x1 + mk_rel_x + mk_rel_w, ms_y1 + mk_rel_y + mk_rel_h),
                          (200, 0, 200), 1)

            if is_calib:
                rx = data.get("screen_ratio_x", None)
                ry = data.get("screen_ratio_y", None)
                if rx is not None and ry is not None:
                    dot_x = int(ms_x1 + rx * MINI_W)
                    dot_y = int(ms_y1 + ry * MINI_H)
                    dot_x = max(ms_x1 + 2, min(ms_x2 - 2, dot_x))
                    dot_y = max(ms_y1 + 2, min(ms_y2 - 2, dot_y))
                    cv2.circle(frame, (dot_x, dot_y), 5, (255, 255, 255), -1)
                    cv2.circle(frame, (dot_x, dot_y), 3, COLOR_DOT, -1)

            cv2.putText(frame, f"Frame: {frame_count}", (w - 150, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # ---- Show ----
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            # ---- Key handling ----
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\nUser quit.")
                break

            elif key == ord('c') or key == ord('C'):
                if not in_calibration:
                    in_calibration = True
                    calib_step = 0
                    vision.reset_calibration()
                    print("[CALIB] Starting 4-corner calibration...")
                else:
                    in_calibration = False
                    vision.reset_calibration()
                    print("[CALIB] Cancelled.")

            elif key == ord(' ') or key == 32:
                if in_calibration and is_present:
                    rh_s = data["gaze_rh"]
                    rv_s = data["gaze_rv"]
                    cname = calib_corner_short[calib_step]
                    print(f"[CALIB] Step {calib_step+1} [{cname}]: sampled rh={rh_s:.3f}, rv={rv_s:.3f}")
                    vision.add_calibration_sample(rh_s, rv_s)
                    calib_step += 1
                    if calib_step >= 4:
                        ok = vision.finalize_calibration()
                        if ok:
                            print("[CALIB] Calibration complete!")
                            print(f"  Bounds: {vision._calib_bounds}")
                        else:
                            print("[CALIB] Calibration FAILED (need 4 samples)")
                        in_calibration = False

            elif key == ord('r') or key == ord('R'):
                vision.reset_calibration()
                in_calibration = False
                calib_step = 0
                print("[CALIB] Reset.")

            # ============================================================
            # v3.0 ROI 裁切测试快捷键
            # ============================================================
            elif key == ord('e') or key == ord('E'):
                # 眼部裁切
                eye_img = vision.get_eye_roi_image()
                if eye_img is not None:
                    save_path = "test_eye_roi.png"
                    eye_img.save(save_path)
                    print(f"[ROI] ✅ 眼部裁切保存: {save_path} (尺寸: {eye_img.size[0]}x{eye_img.size[1]})")
                else:
                    print("[ROI] ⚠ 眼部裁切失败: 未检测到人脸或高清帧不可用")

            elif key == ord('h') or key == ord('H'):
                # 手部裁切
                hands_img = vision.get_hands_roi_image()
                if hands_img is not None:
                    save_path = "test_hands_roi.png"
                    hands_img.save(save_path)
                    print(f"[ROI] ✅ 手部裁切保存: {save_path} (尺寸: {hands_img.size[0]}x{hands_img.size[1]})")
                else:
                    print("[ROI] ⚠ 手部裁切失败: 未检测到手或高清帧不可用")

            elif key == ord('f') or key == ord('F'):
                # 人脸裁切
                face_img = vision.get_face_roi_image()
                if face_img is not None:
                    save_path = "test_face_roi.png"
                    face_img.save(save_path)
                    print(f"[ROI] ✅ 人脸裁切保存: {save_path} (尺寸: {face_img.size[0]}x{face_img.size[1]})")
                else:
                    print("[ROI] ⚠ 人脸裁切失败: 未检测到人脸或高清帧不可用")

            elif key == ord('['):
                VISION_MAKO_X = max(0, VISION_MAKO_X - 20)
                vision.set_mako_window(VISION_MAKO_X, VISION_MAKO_Y,
                                       VISION_MAKO_W, VISION_MAKO_H)

            elif key == ord(']'):
                VISION_MAKO_X = min(SCREEN_RES_X - VISION_MAKO_W, VISION_MAKO_X + 20)
                vision.set_mako_window(VISION_MAKO_X, VISION_MAKO_Y,
                                       VISION_MAKO_W, VISION_MAKO_H)

    except KeyboardInterrupt:
        print("\nUser interrupted (Ctrl+C)")

    finally:
        vision.release()
        cv2.destroyAllWindows()
        print("MakoVision v3.0 test ended.")
        print("=" * 70)
