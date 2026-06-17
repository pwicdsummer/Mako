# class Stack:
#     def __init__(self):
#         self._storage = []
    
#     def push(self,item):
#         self._storage.append(item)
    
#     def pop(self):
#         if self.is_empty():
#             raise IndexError("空栈无法出栈")
#         return self._storage.pop()
    
#     def peek(self):
#         if self.is_empty():
#             return None
#         return self._storage[-1]
    
#     def is_empty(self) -> bool:
#         return len(self.storage) == 0
    
#     def size(self):
#         return len(self._storage)

# import os
# img_stack = Stack()
# base_dir = os.path.dirname(os.path.abspath(__file__))

# for i in os.listdir("img\校服"):
#     img_stack.push(i)

# print(img_stack.size())

# import math

# class MyMath:
#     def __init__(self,r):
#         self.r = r
    
#     def circle_area(self):
#         print(f"圆的面积是：{math.pi * self.r * 2:.2f}") 
    
#     def circle_volumn(self):
#         print(f"圆的体积是：{math.pi * self.r ** 2:.2f}") 
        
#     def sphere_area(self):
#         print(f"球的面积是：{ 4 * math.pi * self.r ** 2:.2f}") 
    
#     def sphere_volumn(self):
#         print(f"球的体积是：{4 / 3 * math.pi * self.r ** 3:.2f}") 

# r = float(input("请输入半径："))
# m = MyMath(r)
# m.circle_area()
# m.circle_volumn()
# m.sphere_area()
# m.sphere_volumn()

text = " [laughing] えへへ、真尋ったら、またそんなところでバグを出してるんですか？[sighing] もう～、仕方ないですね。[serious] 私が見てあげますよ。"

import re

def text_without_emotion_labels(text):
    clean_text = re.sub("\[.*?\]","",text)
    clean_text = re.sub(" ","",clean_text)
    return clean_text

print(text_without_emotion_labels(text))