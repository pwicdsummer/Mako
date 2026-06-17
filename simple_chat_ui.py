"""
simple_chat_ui.py
===============
纯文本驱动的桌宠 UI 核心组件。
无任何外部依赖（只需 PyQt5），无 MessageBase / 数据库绑定。

组件：
  - BubbleMenu:     右键菜单（仅聊聊天 + 退出）
  - BubbleInput:    聊天输入弹窗（CSS 动画俱全）
  - SpeechBubble:   聊天气泡（paintEvent 自绘）
  - SpeechBubbleList: 气泡管理器（纯文本，自动排列 + 定时消失）
  - PetWidget:      茉子桌宠窗口（可拖动）

运行测试：
  python simple_chat_ui.py
"""

import sys
import os
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLineEdit, QPushButton,
    QHBoxLayout, QMenu, QLabel, QGraphicsOpacityEffect,

)
from PyQt5.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QTimer, QPoint, QRect, QSize,
    QSequentialAnimationGroup, pyqtProperty, pyqtSignal,
)
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPainterPath, QPixmap,
)

# ============================================================
# 常量
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCALE = 1.0

# 茉子图片
PET_IMAGE_PATH = os.path.join(PROJECT_ROOT, "img", "茉子a_1892_2558.png")
# 原始尺寸 970x3333，缩放 0.1 倍 → 约 97x333
PET_SCALE_FACTOR = 0.5


# ============================================================
# BubbleInput — 聊天输入弹窗（从原项目剥离，保留 CSS + 动画）
# ============================================================
class BubbleInput(QWidget):
    """聊天输入弹窗。纯文本，无任何外部依赖。"""

    def __init__(self, parent=None, on_send=None):
        super().__init__(parent)
        self.on_send_callback = on_send
        self._init_ui()
        self._init_style()
        self._init_animation()

    def _init_ui(self):
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("对我说点什么吧...")
        self.input_field.setMinimumWidth(250)
        self.input_field.setFont(QFont("Microsoft YaHei", 12))

        self.send_btn = QPushButton("发送")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFont(QFont("Microsoft YaHei", 12))

        layout = QHBoxLayout(self)
        layout.addWidget(self.input_field)
        layout.addWidget(self.send_btn)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(10)

        self.send_btn.clicked.connect(self._on_send)
        self.input_field.returnPressed.connect(self._on_send)

    def _init_style(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f0f8ff, stop:1 #e6f3ff);
                border-radius: 15px;
                border: 2px solid #a0d1eb;
            }}
            QLineEdit {{
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #c0ddec;
                border-radius: 10px;
                padding: 8px;
                font-size: 14px;
                color: #333;
            }}
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-weight: bold;
                min-width: 60px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5cb860, stop:1 #4CAF50);
            }}
        """)

    def _init_animation(self):
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.OutQuad)

    def showEvent(self, event):
        self._animate_show()
        super().showEvent(event)

    def _animate_show(self):
        self.anim.stop()
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.start()

    def _on_send(self):
        text = self.input_field.text().strip()
        if text and self.on_send_callback:
            self.on_send_callback(text)
        self.close()

    def close(self):
        self.input_field.clear()
        super().close()
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.start()

    def update_position(self):
        if not self.parent():
            return
        parent_rect = self.parent().geometry()
        screen = self.parent().screen().availableGeometry()

        x = parent_rect.center().x() - self.width() // 2
        y = parent_rect.bottom() + 10

        x = max(screen.left() + 10, min(x, screen.right() - self.width() - 10))
        if y + self.height() > screen.bottom():
            y = parent_rect.top() - self.height() - 10
        y = max(screen.top() + 10, y)

        self.move(x, y)


# ============================================================
# BubbleMenu — 右键菜单（定制化 QMenu，仅保留核心项）
# ============================================================
class BubbleMenu(QMenu):
    """自定义圆角右键菜单。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 5px;
                border: 1px solid #e0e0e0;
            }
            QMenu::item {
                padding: 8px 20px;
                color: #333;
                border-radius: 5px;
            }
            QMenu::item:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        w, h = float(self.width()), float(self.height())
        path.addRoundedRect(0, 0, w, h, 10, 10)
        painter.fillPath(path, QColor(255, 255, 255))
        super().paintEvent(event)


# ============================================================
# SpeechBubble — 聊天气泡（paintEvent 自绘 + 淡出动画）
# ============================================================
class SpeechBubble(QLabel):
    """单个聊天气泡。纯文字自绘，无图片 / MessageBase 依赖。"""

    def __init__(self, parent=None, text: str = "", msg_type: str = "received"):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.text_data = text
        self.msg_type = msg_type

        self._content_margin = 10
        self._image_text_spacing = 5

        if msg_type == "received":
            self.bg_color = QColor(240, 248, 255)
            self.follow_offset = QPoint(-100, -30)
        else:
            self.bg_color = QColor(200, 255, 200)
            self.follow_offset = QPoint(100, -30)

        self.text_color = QColor(70, 70, 70)
        self.setFont(QFont("Microsoft YaHei", 12))
        self.corner_radius = 10
        self.arrow_height = 10

        self.animation_group = QSequentialAnimationGroup(self)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        body_rect = self.rect().adjusted(0, 0, 0, -self.arrow_height)
        painter.setBrush(self.bg_color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(body_rect, self.corner_radius, self.corner_radius)

        path = QPainterPath()
        arrow_width = 20
        if self.msg_type == "received":
            center_x = 30
        else:
            center_x = self.width() - 30

        path.moveTo(center_x - arrow_width // 2, body_rect.height())
        path.lineTo(center_x, self.height())
        path.lineTo(center_x + arrow_width // 2, body_rect.height())
        painter.drawPath(path)

        if self.text_data:
            text_rect = QRect(10, 5, self.width() - 20, body_rect.height() - 5)
            painter.setPen(self.text_color)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.TextWordWrap, self.text_data)

    def calculate_bubble_size(self) -> QSize:
        min_text_width = 100
        text_width = 300
        text_height = 0

        if self.text_data:
            text_rect = self.fontMetrics().boundingRect(
                QRect(0, 0, text_width - 2 * self._content_margin, 0),
                Qt.TextWordWrap,
                self.text_data
            )
            text_height = text_rect.height()
            min_text_width = min(text_rect.width() + 2 * self._content_margin, min_text_width)

        content_width = max(min_text_width, text_width) + 2 * self._content_margin
        height = text_height + 2 * self._content_margin + self.arrow_height

        return QSize(content_width, height)

    def show_message(self):
        size = self.calculate_bubble_size()
        self.resize(size)
        self.show()

    def fade_out(self):
        if self.animation_group.state() == QPropertyAnimation.Running:
            self.animation_group.stop()
        self.animation_group.clear()

        fade_anim = QPropertyAnimation(self, b"windowOpacity")
        fade_anim.setDuration(500)
        fade_anim.setStartValue(1.0)
        fade_anim.setEndValue(0.0)
        fade_anim.finished.connect(self.deleteLater)

        self.animation_group.addAnimation(fade_anim)
        self.animation_group.start()


# ============================================================
# SpeechBubbleList — 气泡列表管理器（纯文本，无数据库）
# ============================================================
class SpeechBubbleList:
    _vertical_spacing = 5
    MAX_BUBBLES = 5  # 最多同时显示 5 个气泡

    def __init__(self, parent=None):
        self.parent = parent
        self._active_bubbles: deque[SpeechBubble] = deque()

    def add_message(self, text: str, msg_type: str = "received"):
        new_bubble = SpeechBubble(parent=self.parent, text=text, msg_type=msg_type)
        self._active_bubbles.append(new_bubble)
        new_bubble.show_message()
        # ★ 新增：超过 MAX_BUBBLES 个 → 自动删除最老的
        if len(self._active_bubbles) > self.MAX_BUBBLES:
            self.del_first_msg()
        self.update_position()


    def del_first_msg(self):
        if self._active_bubbles and self._active_bubbles[0]:
            self._active_bubbles[0].fade_out()
            del self._active_bubbles[0]

    def clear_all(self):
        for bubble in self._active_bubbles:
            bubble.deleteLater()
        self._active_bubbles.clear()

    def update_position(self):
        if not self.parent or not hasattr(self.parent, 'geometry'):
            return

        screen_geo = QApplication.primaryScreen().availableGeometry()
        parent_rect = self.parent.geometry()

        center_x = parent_rect.center().x()
        # 茉子的头部在图片上方区域，气泡显示在头顶偏上位置
        base_y = parent_rect.top() - 10

        total_height = 0
        visible_bubbles = [b for b in self._active_bubbles if b.isVisible()]

        for bubble in reversed(visible_bubbles):
            bubble_size = bubble.size()
            bubble_width = bubble_size.width()
            bubble_height = bubble_size.height()

            if bubble.msg_type == "received":
                x_pos = max(screen_geo.left() + 10,
                            center_x - 160 - bubble_width // 2)
            else:
                x_pos = min(screen_geo.right() - bubble_width - 10,
                            center_x + 160 - bubble_width // 2)

            y_pos = base_y - total_height - bubble_height

            if y_pos < screen_geo.top():
                y_pos = parent_rect.bottom() + total_height + 30
                bubble.arrow_height = -abs(bubble.arrow_height)
            else:
                bubble.arrow_height = abs(bubble.arrow_height)

            x_pos = max(screen_geo.left() + 5,
                        min(x_pos, screen_geo.right() - bubble_width - 5))
            y_pos = max(screen_geo.top() + 5,
                        min(y_pos, screen_geo.bottom() - bubble_height - 5))

            bubble.move(int(x_pos), int(y_pos))
            total_height += bubble_height + self._vertical_spacing

            if total_height > screen_geo.height() *2 // 3:
                self.del_first_msg()


# ============================================================
# PetWidget — 茉子桌宠窗口
# ============================================================
class PetWidget(QWidget):
    """以茉子图片为形象的桌面宠物。可拖动、右键菜单、聊天气泡联动。"""

    AUTO_DISMISS_MS = 90000  # 气泡 90 秒后消失

    # 语音切换信号：右键菜单点击时发射，供 main.py 连接 VoiceListener
    voice_toggled = pyqtSignal(bool)
    # 视觉切换信号：右键菜单点击时发射，供 main.py 连接视觉模块
    vision_toggled = pyqtSignal(bool)
    # TTS 音源切换信号：右键菜单点击时发射，供 main.py 切换后端
    tts_toggled = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._drag_pos = None
        self._mako_opacity = 1.0  # 透明度属性，用于 fade_switch_image 动画
        self._voice_enabled = False  # 语音状态跟踪
        self._vision_enabled = False  # 视觉状态跟踪（默认关闭，与实际加载状态对齐）
        self._tts_backend = "GS"  # TTS 音源："GS"=GPT-SoVITS, "FA"=Fish Audio

        # 跳跃动画组（防重复触发）
        self.jump_anim_group = None

        # 加载并缩放茉子图片
        self.original_pixmap = QPixmap(PET_IMAGE_PATH)
        self.scaled_pixmap = self.original_pixmap.scaled(
            int(self.original_pixmap.width() * PET_SCALE_FACTOR),
            int(self.original_pixmap.height() * PET_SCALE_FACTOR),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        pet_w = self.scaled_pixmap.width()
        pet_h = self.scaled_pixmap.height()

        # 窗口属性 — 严格恢复基准代码设置：无边框、透明背景
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(pet_w, pet_h)

        # 居中屏幕
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2,
        )

        # 气泡系统
        self.chat_bubbles = SpeechBubbleList(parent=self)
        self.bubble_input = BubbleInput(parent=self, on_send=self._on_user_send)
        self.bubble_input.hide()

    # ---- pyqtProperty 包装 _mako_opacity，供 QPropertyAnimation 驱动 ----
    @pyqtProperty(float)
    def mako_opacity(self):
        return self._mako_opacity

    @mako_opacity.setter
    def mako_opacity(self, value):
        self._mako_opacity = value
        self.update()  # 触发重绘

    # ---- 绘制：茉子图片（加入透明度控制） ----
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setOpacity(self._mako_opacity)
        painter.drawPixmap(0, 0, self.scaled_pixmap)

    # ---- 平滑切换图片（虚化 → 换图 → 恢复） ----
    def fade_switch_image(self, new_path: str):
        """
        平滑切换茉子立绘。
        动画：1.0 → 0.5（淡出），瞬间换图，0.5 → 1.0（淡入）。
        透明度最低为 0.5，角色不会完全消失。
        """
        # 停掉并清理之前的动画，防止信号堆积
        if hasattr(self, '_fade_anim') and self._fade_anim:
            try:
                self._fade_anim.finished.disconnect()
            except TypeError:
                pass
            self._fade_anim.stop()
        if hasattr(self, '_fade_in_anim') and self._fade_in_anim:
            try:
                self._fade_in_anim.finished.disconnect()
            except TypeError:
                pass
            self._fade_in_anim.stop()

        # 第一阶段动画：1.0 → 0.5
        self._fade_anim = QPropertyAnimation(self, b"mako_opacity")
        self._fade_anim.setDuration(200)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.5)
        self._fade_anim.setEasingCurve(QEasingCurve.InOutQuad)

        # 第二阶段动画：0.5 → 1.0（换图后自动触发）
        self._fade_in_anim = QPropertyAnimation(self, b"mako_opacity")
        self._fade_in_anim.setDuration(200)
        self._fade_in_anim.setStartValue(0.5)
        self._fade_in_anim.setEndValue(1.0)
        self._fade_in_anim.setEasingCurve(QEasingCurve.InOutQuad)

        # 当到达 0.5 时瞬间换图 → 再自动切入第二阶段
        self._fade_anim.finished.connect(lambda: self._do_switch_image(new_path))
        self._fade_anim.finished.connect(self._fade_in_anim.start)

        self._fade_anim.start()

    def _do_switch_image(self, new_path: str):
        """内部方法：在动画到达 0.5 时执行实际换图。"""
        new_pixmap = QPixmap(new_path)
        if new_pixmap.isNull():
            # 致命防御：图片无效时直接恢复透明度，绝不坍缩窗口
            self._mako_opacity = 1.0
            self.update()
            return
        self.original_pixmap = new_pixmap
        self.scaled_pixmap = self.original_pixmap.scaled(
            int(self.original_pixmap.width() * PET_SCALE_FACTOR),
            int(self.original_pixmap.height() * PET_SCALE_FACTOR),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setFixedSize(self.scaled_pixmap.width(), self.scaled_pixmap.height())
        self.update()

    # ---- 跳跃动画 ----
    def jump(self, times = 2):
        """
        茉子向上跳跃动画。

        参数
        ----------
        times : int
            连续跳跃次数（默认 1）。例如 times=2 为连跳两次。
        """
        # 防误触：如果当前正在跳跃，则忽略新的指令
        if self.jump_anim_group and self.jump_anim_group.state() == QSequentialAnimationGroup.Running:
            return

        jump_height = 80
        duration_up = 110
        duration_down = 200
        current_pos = self.pos()

        # 一次性把所有跳跃段加入同一个 QSequentialAnimationGroup
        self.jump_anim_group = QSequentialAnimationGroup()
        for _ in range(times):
            peak_pos = QPoint(current_pos.x(), current_pos.y() - jump_height)

            anim_up = QPropertyAnimation(self, b"pos")
            anim_up.setDuration(duration_up)
            anim_up.setStartValue(current_pos)
            anim_up.setEndValue(peak_pos)
            anim_up.setEasingCurve(QEasingCurve.OutQuad)

            anim_down = QPropertyAnimation(self, b"pos")
            anim_down.setDuration(duration_down)
            anim_down.setStartValue(peak_pos)
            anim_down.setEndValue(current_pos)
            anim_down.setEasingCurve(QEasingCurve.InQuad)

            self.jump_anim_group.addAnimation(anim_up)
            self.jump_anim_group.addAnimation(anim_down)

        # 全部跳完后释放引用，允许下一次 jump() 正常进入
        self.jump_anim_group.finished.connect(lambda: setattr(self, 'jump_anim_group', None))
        self.jump_anim_group.start()

    # ---- 害羞抖动动画 ----
    def shy_movement(self):
        # 防误触：如果当前正在播放害羞动画，则忽略
        if hasattr(self, 'shy_anim') and self.shy_anim.state() == QPropertyAnimation.Running:
            return

        self.shy_anim = QPropertyAnimation(self, b"pos")
        self.shy_anim.setDuration(400)
        self.shy_anim.setEasingCurve(QEasingCurve.Linear)

        current_pos = self.pos()
        x = current_pos.x()
        y = current_pos.y()
        offset = 15

        self.shy_anim.setKeyValueAt(0.0, QPoint(x, y))
        self.shy_anim.setKeyValueAt(0.2, QPoint(x - offset, y))
        self.shy_anim.setKeyValueAt(0.4, QPoint(x + offset, y))
        self.shy_anim.setKeyValueAt(0.6, QPoint(x - offset, y))
        self.shy_anim.setKeyValueAt(0.8, QPoint(x + offset, y))
        self.shy_anim.setKeyValueAt(1.0, QPoint(x, y))

        self.shy_anim.start()

    # ---- 鼠标拖动 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            self.chat_bubbles.update_position()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def _toggle_voice(self):
        """切换语音监听状态，更新菜单项文字并发射 signal。"""
        self._voice_enabled = not self._voice_enabled
        self.voice_toggled.emit(self._voice_enabled)

    def _toggle_vision(self):
        """切换视觉认知状态，更新菜单项文字并发射 signal。"""
        self._vision_enabled = not self._vision_enabled
        self.vision_toggled.emit(self._vision_enabled)

    def _toggle_tts(self):
        """切换 TTS 音源后端，更新菜单项文字并发射 signal。"""
        self._tts_backend = "FA" if self._tts_backend == "GS" else "GS"
        self.tts_toggled.emit(self._tts_backend)

    # ---- 右键菜单 ----
    def contextMenuEvent(self, event):
        menu = BubbleMenu(self)
        chat_action = menu.addAction("\u270f\ufe0f 聊聊天")
        chat_action.triggered.connect(self._show_chat_input)
        menu.addSeparator()
        # 视觉切换菜单项（动态文字）
        vision_label = "\U0001f441\ufe0f 视觉认知：开" if self._vision_enabled else "\U0001f441\ufe0f 视觉认知：关"
        vision_action = menu.addAction(vision_label)
        vision_action.triggered.connect(self._toggle_vision)
        menu.addSeparator()
        # TTS 音源切换菜单项（动态文字）
        tts_label = "\U0001f399\ufe0f TTS音源：GS" if self._tts_backend == "GS" else "\U0001f399\ufe0f TTS音源：FA"
        tts_action = menu.addAction(tts_label)
        tts_action.triggered.connect(self._toggle_tts)
        menu.addSeparator()
        # 语音切换菜单项（动态文字：监听中显示绿点状态）
        voice_label = "\U0001f3a7 语音输入：开" if self._voice_enabled else "\U0001f3a4 语音输入：关"
        voice_action = menu.addAction(voice_label)
        voice_action.triggered.connect(self._toggle_voice)
        menu.addSeparator()
        exit_action = menu.addAction("\u274c 退出")
        exit_action.triggered.connect(QApplication.instance().quit)
        menu.exec_(event.globalPos())
        menu.deleteLater()


    # ---- 交互链路 ----
    def _show_chat_input(self):
        self.bubble_input.show()
        self.bubble_input.update_position()
        self.bubble_input.input_field.setFocus()

    def _on_user_send(self, text: str):
        # 显示自己的消息（右侧，绿色）
        self.chat_bubbles.add_message(text, msg_type="sent")
        QTimer.singleShot(self.AUTO_DISMISS_MS, self.chat_bubbles.del_first_msg)

        # 模拟收到回复（左侧，蓝色）
        reply = f"你说：{text}\n—— 茉子收到了 \U0001f4ac"
        QTimer.singleShot(800, lambda: self._show_reply(reply))

    def _show_reply(self, text: str):
        self.chat_bubbles.add_message(text, msg_type="received")
        QTimer.singleShot(self.AUTO_DISMISS_MS, self.chat_bubbles.del_first_msg)


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = PetWidget()
    widget.show()
    sys.exit(app.exec_())
