# 可视化点云
# 点云+高斯噪声
# 输入平滑obj，导出点云ply, 增加噪声, 导出噪声.ply
import os
import torch
import argparse
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes, save_ply
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import read_pts_from_pc
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

"""
def plot_mesh_to_pointcloud(mesh, title=""):
    # 从mesh中均匀采样点
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()
"""
def plot_pointcloud(filepath):
    # 读取ply, 点云可视化
    vs, normal=read_pts(filepath)
    v2=vs
    x, y, z = v2[:,0],v2[:,1],v2[:,2]
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title("pointcloud vis")
    ax.view_init(190, 30)
    plt.show()

def plot_pointcloud_from_array(vs):
    # 从顶点数组进行可视化
    x, y, z = vs[:, 0], vs[:, 1], vs[:, 2]
    fig = plt.figure(figsize=(20, 20))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title("pointcloud vis")
    ax.view_init(190, 30)
    plt.show()

def add_noise_to_pc(vs):
    w,h=vs.shape
    np.random.seed(0)
    g_noise=np.random.normal(0,5,(w,h))
    vs+=g_noise
    return vs

apath = os.getcwd()
# pc_path=apath+"/data/g.ply"
# vs, normal=read_pts(pc_path)
# # add_noise_to_pc(vs)
# plot_pointcloud_from_array(vs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get original and noise ply')
    parser.add_argument('--i', type=str, required=True,
                        help='path to read .obj from')
    parser.add_argument('--samples-num', type=int, required=True, help='target of points for the points cloud')

    parser.add_argument('--o', type=str, required=True,
                        help='path to output original ply to')
    parser.add_argument('--o-noise', type=str, required=True,
                        help='path to output ply with noise to')
    args = parser.parse_args()


input_path = apath + args.i
output_path = apath + args.o
output_noise_path = apath + args.o_noise

device = torch.device('cpu')
mesh = load_objs_as_meshes([input_path], device=device)
num_samples = args.samples_num # 采样数
xyz, normals = sample_points_from_meshes(mesh, num_samples=num_samples, return_normals=True)
print(xyz.shape)
save_ply(output_path, verts=xyz[0, :], verts_normals=normals[0, :]) # obj转光滑点云
vs, normal=read_pts_from_pc(output_path)
plot_pointcloud_from_array(vs) # 显示smooth点云
vs=add_noise_to_pc(vs) # 加噪声
plot_pointcloud_from_array(vs) # 显示噪声点云
xyz=torch.from_numpy(vs).unsqueeze(0)
print(xyz.shape)
save_ply(output_noise_path,verts=xyz[0,:],verts_normals=normals[0, :]) # 存储noise.ply
