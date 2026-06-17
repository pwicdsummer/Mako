"""
mako_image_loader.py
====================
独立的图片供应模块。
只负责提供茉子立绘的路径字符串，与 UI / LLM 逻辑完全解耦。
禁止包含任何 PyQt5 代码。
"""

import os
import random
#"C:\Users\kz740\Desktop\MakoTalker\img\茉子b_2054_2216.png"
#"C:\Users\kz740\Desktop\MakoTalker\img\茉子a_1896_2559.png"
_base_dir = os.path.dirname(os.path.abspath(__file__))
_img_dir = os.path.join(_base_dir,"img")

_DEFAULT_PATH = r"C:\Users\kz740\Desktop\MakoTalker\img\茉子a_1892_2558.png"
current_path=_DEFAULT_PATH

lst_normal=list(os.listdir(_img_dir+"/校服"))
Lst_normal=[]
for i in range(len(lst_normal)):
    Lst_normal.append(_img_dir+"/校服"+"/"+lst_normal[i])
Lst_normal_c=random.sample(Lst_normal,len(Lst_normal))

lst_shy=list(os.listdir(_img_dir+"/害羞"))
Lst_shy=[]
for i in range(len(lst_shy)):
    Lst_shy.append(_img_dir+"/害羞"+"/"+lst_shy[i])
Lst_shy_c=random.sample(Lst_shy,len(Lst_shy))

def get_mako_image_path(state: str = "normal") -> str:
    """
    获取茉子立绘的图片路径。

    当前忽略 state 参数，始终返回默认路径。
    未来可根据 state 返回不同表情/动作的图片路径。

    Parameters
    ----------
    state : str
        茉子状态标识（如 "normal", "happy", "thinking" 等）。

    Returns
    -------
    str
        图片文件的绝对路径。
    """
    return current_path


def provide_default_path(emotion) -> str:
    """
    返回默认立绘的绝对路径。

    Returns
    -------
    str
        默认图片的绝对路径字符串。
    """
    global current_path
    global Lst_normal_c
    global Lst_shy_c
    if emotion == "normal":
        current_path=Lst_normal_c.pop(0)
        if len(Lst_normal_c) == 0:
            Lst_normal_c=random.sample(Lst_normal,len(Lst_normal))
    else:
        current_path=Lst_shy_c.pop(0)
        if len(Lst_shy_c) == 0:
            Lst_shy_c=random.sample(Lst_shy,len(Lst_shy))
    return current_path
