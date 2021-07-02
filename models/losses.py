import torch
import pytorch3d
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.knn import knn_gather, knn_points
from typing import Union
import torch.nn.functional as F

# 判断输入是否正确
def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
):
    # batch_reduction只接受["mean", "sum"]或None
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    # point_reduction只接受["mean", "sum"]
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

# 确定输入的点云是否为Pointclouds的实例
def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    # 若是就填充每一批点的数量和法向
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # 为张量或是None
    # 否则返回每个点云点的个数(基于第二个维度)
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals

# 计算倒角距离(用于估计两个点云的偏差)
def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    unoriented=False,
): # 输入两个点云以及相关缩减规模的参数

    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # 检查输入是否异构并创建长度掩码
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]
    cham_y = y_nn.dists[..., 0]

    eps = 0
    cham_x = torch.sqrt(cham_x + eps)
    cham_y = torch.sqrt(cham_y + eps)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # 根据索引得到法向
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        cham_norm_y = F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        if unoriented:
            cham_norm_x = abs(cham_norm_x)
            cham_norm_y = abs(cham_norm_y)

        cham_norm_x = -1*cham_norm_x
        cham_norm_y = -1*cham_norm_y

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # 减少点的数量
    cham_x = cham_x.sum(1)
    cham_y = cham_y.sum(1)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)
        cham_norm_y = cham_norm_y.sum(1)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    # 减少批的规模
    if batch_reduction is not None:
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    # 返回距离差以及法线之间的余弦距离差
    return cham_dist, cham_normals


class ZeroNanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad[grad != grad] = 0
        return grad

# 计算Beam Loss
class BeamGapLoss:
    def __init__(self, device):
        self.device = device
        self.points, self.masks = None, None

    def update_pm(self, pmesh, target_pc):
        points, masks = [], []
        target_pc.to(self.device)
        total_mask = torch.zeros(pmesh.main_mesh.vs.shape[0])
        for i, m in enumerate(pmesh):
            p, mask = m.discrete_project(target_pc, thres=0.99, cpu=True)
            p, mask = p.to(target_pc.device), mask.to(target_pc.device)
            points.append(p[:, :3])
            masks.append(mask)
            temp = torch.zeros(m.vs.shape[0])
            if (mask != False).any():
                temp[m.faces[mask]] = 1
                total_mask[pmesh.sub_mesh_index[i]] += temp
        self.points, self.masks = points, masks

    def __call__(self, pmesh, j):
        losses = self.points[j] - pmesh[j].vs[pmesh[j].faces].mean(dim=1)
        losses = ZeroNanGrad.apply(losses)
        losses = torch.norm(losses, dim=1)[self.masks[j]]
        l2 = losses.mean().float()
        return l2 * 1e1
