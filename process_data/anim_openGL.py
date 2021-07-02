# 使用opengl线框显示模型
#! /usr/bin/python
import os
from utils import load_all_recon_obj, load_obj
import openmesh as om
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ctypes
import numpy as np

global mytimer,count, max_count, vs, faces, obj_list,mesh,gcount
IS_PERSPECTIVE = True                               # 透视投影
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])                 # 模型缩放比例
EYE = np.array([0.5, 0.0, 2.0])                     # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, 0.0])                 # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 1.0, 0.0])                  # 定义对观察者而言的上方（默认y轴的正方向）
WIN_W, WIN_H = 640, 480                             # 保存窗口宽度和高度的变量
LEFT_IS_DOWNED = False                              # 鼠标左键被按下
MOUSE_X, MOUSE_Y = 0, 0                             # 考察鼠标位移量时保存的起始位置

# 初始化
def init():
    glClearColor(0.5, 0.5, 0.8, 0.8)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(
        EYE[0], EYE[1], EYE[2],
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )

    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)

# 绘制obj
def draw():
    global mytimer, count, max_count, vs, faces, obj_list,mesh
    if (count == max_count):
        print("")
    else:
        if (mytimer % 2 == 0):
            mesh = om.TriMesh()
            vs, faces = load_obj(obj_list[count][1]) # 指定文件加载obj
            print(obj_list[count][1])
            # print(vs)
            # print(faces)
            mesh.add_vertices(vs)
            mesh.add_faces(faces)
            mesh.request_face_normals() #mesh计算出顶点法线
            mesh.update_normals() # 释放面法线
            mesh.release_face_normals()
            count += 1
            mytimer = 0

    mytimer += 1
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_CULL_FACE)
    glPolygonMode(GL_FRONT, GL_LINE)
    # glBegin(GL_TRIANGLES)
    # for face in faces:
    #     print(face)
    #     # 第一个点的法线，纹理，位置信息
    #     if(k==0): glColor4f(1.0,0.0,0.0,1.0)
    #     elif(k==1):
    #         glColor4f(1.0,1.0,0.0,1.0)
    #     elif(k==2): glColor4f(0.0,1.0,0.0,1.0)
    #     else: glColor4f(1.0,0.0,1.0,1.0)
    #     k+=1
    #     for i in range(3):
    #         cur_v=vs[face[i]]
    #         print(cur_v)
    #         glVertex3fv((ctypes.c_float * 3)(*cur_v))
    # glEnd()

    for face in mesh.faces():
        # 每个面
        # print(mesh.fv(face))
        glBegin(GL_TRIANGLES)
        cur_v = []
        for it in enumerate(mesh.fv(face)):
            cur_v.append(it)
        for fv in cur_v:
            glVertex3fv(mesh.point(fv[1]))
            glNormal3fv(mesh.normal(fv[1]))
            # print(mesh.normal(fv[1]))
            # print(mesh.point(fv[1]))
        glEnd()

    glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

def reshape(width, height):
    glutPostRedisplay()

apath = os.getcwd()
print(apath)
walkpath = apath + '/checkpoints/guitar'
print(walkpath)
obj_list = load_all_recon_obj(walkpath)
count=0
mytimer=0
gcount=0
max_count=len(obj_list)

glutInit()
displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
glutInitDisplayMode(displayMode)

glutInitWindowSize(800,600)
glutInitWindowPosition(300, 200)
glutCreateWindow('Quidam Of OpenGL')

init()  # 初始化画布
glutDisplayFunc(draw)  # 注册回调函数draw()
glutIdleFunc(draw)
glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
glutMainLoop()  # 进入glut主循环


