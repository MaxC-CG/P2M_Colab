import torch
from models.layers.mesh import Mesh, PartMesh
from models.networks import init_net, sample_surface, local_nonuniform_penalty
import utils
import numpy as np
from models.losses import chamfer_distance, BeamGapLoss
from options import Options
import time
import os


options = Options()
opts = options.args

torch.manual_seed(opts.torch_seed)
if torch.cuda.is_available():
  device=torch.device(f'cuda:{opts.gpu}')
else:
  device=torch.device('cpu')
print(f'device: {device}')

# get initial mesh: Mesh初始网格
mesh = Mesh(opts.initial_mesh, device=device, hold_history=True)

# input point cloud : 点云输入返回nparray
input_point, input_normals = utils.read_pts_from_pc(opts.input_pc)
# normalize point cloud based on initial mesh: 基于初始网格对点云进行坐标变换
input_point /= mesh.scale
input_point += mesh.translations[None, :]
input_point = torch.Tensor(input_point).type(options.dtype()).to(device)[None, :, :]
input_normals = torch.Tensor(input_normals).type(options.dtype()).to(device)[None, :, :]

# create part-mesh: 创建part-mesh
part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
print(f'number of parts {part_mesh.n_submeshes}')
# init net
net, optimizer, rand_c_l, scheduler = init_net(mesh, part_mesh, device, opts) 
# 创建beamGapLoss类
beamgap_loss = BeamGapLoss(device) 

if opts.beamgap_iterations > 0: # default :0
    print('beamgap on')
    beamgap_loss.update_pm(part_mesh, torch.cat([input_point, input_normals], dim=-1))

for i in range(opts.iterations):
    # for the i iter needs sample :cur_num_samples
    cur_num_samples = options.get_num_samples(i % opts.upsamp) 
    if opts.global_step: # default :false
        optimizer.zero_grad()
    start_time = time.time()
    for part_i, est_verts in enumerate(net(rand_c_l, part_mesh)): # forward
        if not opts.global_step:
            optimizer.zero_grad()
        part_mesh.update_verts(est_verts[0], part_i)
        cur_num_samples = options.get_num_samples(i % opts.upsamp) # the i iter needs sample num_samples
        # Sampler->Y
        recon_point, recon_normals = sample_surface(part_mesh.main_mesh.faces, part_mesh.main_mesh.vs.unsqueeze(0), cur_num_samples)
        # calc chamfer loss w/ normals
        recon_point, recon_normals = recon_point.type(options.dtype()), recon_normals.type(options.dtype())
        point_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_point, input_point, x_normals=recon_normals, y_normals=input_normals,
                                                                    unoriented=opts.unoriented) # chamfer dis->loss

        if (i < opts.beamgap_iterations) and (i % opts.beamgap_modulo == 0): # the iter needs beam-gap loss
            loss = beamgap_loss(part_mesh, part_i) 
        else:
            loss = (point_chamfer_loss + (opts.ang_wt * normals_chamfer_loss))
        if opts.local_non_uniform > 0:
            loss += opts.local_non_uniform * local_nonuniform_penalty(part_mesh.main_mesh).float()
        loss.backward()
        if not opts.global_step:
            optimizer.step()
            scheduler.step()
        part_mesh.main_mesh.vs.detach_()
        # end-for-batch(part-mesh)
    if opts.global_step:
        optimizer.step()
        scheduler.step()
    end_time = time.time()

    # each iter
    print(f'{os.path.basename(opts.input_pc)} || iter: {i} / {opts.iterations} || loss: {loss.item():.4f} ||'
        f' sample count: {cur_num_samples} || time: {end_time - start_time:.2f}')
    if i % opts.export_interval == 0 and i > 0:
        print('exporting reconstruction... current LR: {}'.format(optimizer.param_groups[0]['lr']))
        with torch.no_grad():
            part_mesh.export(os.path.join(opts.save_path, f'recon_iter_{i}.obj')) # export OBJ

    if (i > 0 and (i + 1) % opts.upsamp == 0): # the iter needs UP-SAMPLE: RWM
        mesh = part_mesh.main_mesh

        need_num_faces= len(mesh.faces) * 1.5
        min_faces=min(len(mesh.faces),opts.max_faces)
        max_faces=max(len(mesh.faces),opts.max_faces)
        if(need_num_faces<min_faces):
            need_num_faces=min_faces
        if(need_num_faces>max_faces):
            need_num_faces=max_faces
        need_num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), opts.max_faces))

        if need_num_faces > len(mesh.faces) or opts.manifold_always:
            # up-sample mesh
            mesh = utils.manifold_upsample(mesh, opts.save_path, Mesh, 
                                           num_faces=min(need_num_faces, opts.max_faces),
                                           res=opts.manifold_res, simplify=True)

            part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
            print(f'upsampled to {len(mesh.faces)} faces; number of parts {part_mesh.n_submeshes}')
            net, optimizer, rand_c_l, scheduler = init_net(mesh, part_mesh, device, opts) # init-net for next level
            if i < opts.beamgap_iterations:
                print('beam-gap updated')
                beamgap_loss.update_pm(part_mesh, input_point)
    # end-for-iteration
    
with torch.no_grad():
    mesh.export(os.path.join(opts.save_path, 'last_recon.obj')) # final result
