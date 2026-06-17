import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl, QTimer, QPropertyAnimation, QRect
import os

class FullscreenPlayer(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        # 设置窗口属性以支持透明背景（关键点）
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) # 窗口底色透明
        self.showFullScreen()
        
        self.mediaPlayer = QMediaPlayer()
        self.audioOutput = QAudioOutput()
        self.mediaPlayer.setAudioOutput(self.audioOutput)
        
        self.videoWidget = QVideoWidget()
        self.setCentralWidget(self.videoWidget)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        
        self.mediaPlayer.setSource(QUrl.fromLocalFile(video_path))
        self.mediaPlayer.play()
        
        # 使用定时器监听播放进度 (单位：毫秒)
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_time)
        self.timer.start(100) # 每100ms检查一次

    def check_time(self):
        # 21000毫秒 = 21秒
        if self.mediaPlayer.position() >= 22500:
            self.timer.stop()
            self.start_fade_out()

    def start_fade_out(self):
        # 创建一个透明度动画，从 1.0 渐变到 0.0
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(1500) # 4秒淡出
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.finished.connect(QApplication.quit)
        self.animation.start()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app = QApplication(sys.argv)
    player = FullscreenPlayer(os.path.join(base_dir,"launch_mv.mp4"))
    sys.exit(app.exec())
