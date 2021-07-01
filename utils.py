import os
import numpy as np
import torch
import uuid
from options import MANIFOLD_DIR
import glob

def random_file_name(suffix, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{suffix}'


def read_pts_from_pc(pts_file):
    '''
    :param pts_file: file path of a .ply or .xyz
    such that a particular line has 6 float values: x, y, z, nx, ny, nz
    :return: xyz, normals
    '''
    xyz, normals = [], []

    f = open(pts_file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        try:
          temp=np.array(splitted_line,dtype=np.float32)
          xyz.append(temp[:3])
          normals.append(temp[3:])
        except:
          pass
    f.close()

    return np.array(xyz, dtype=np.float32), np.array(normals, dtype=np.float32)


def load_obj(file):
    '''
    :param pts_file: file path of .obj 
    record position of vertice, indice of vertice for each face
    :return: vertice(array), faces(array)
    '''
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def export(file, vs, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))


def manifold_upsample(mesh, save_path, Mesh, num_faces=2000, res=3000, simplify=True):
    '''
    :param: mesh(Mesh), save_path(path to save), Mesh(method to init), res(resolution)
    :return: up-sampled mesh(Mesh)
    '''
    # export before upsample
    num_face_before_RWM=len(mesh.faces)
    print(f"current num of faces [before RWM]: {num_face_before_RWM}")
    fname = os.path.join(save_path, f'recon_{num_face_before_RWM}.obj')
    mesh.export(fname) # Mesh.export(filepath)

    temp_file = os.path.join(save_path, random_file_name('obj'))
    opts = ' ' + str(res) if res is not None else ''

    manifold_script_path = os.path.join(MANIFOLD_DIR, 'manifold')
    if not os.path.exists(manifold_script_path):
        raise FileNotFoundError(f'{manifold_script_path} not found')
    cmd = f"{manifold_script_path} {fname} {temp_file + opts}"
    os.system(cmd)

    if simplify:
        cmd = "{} -i {} -o {} -f {}".format(os.path.join(MANIFOLD_DIR, 'simplify'), temp_file,
                                                             temp_file, num_faces)
        os.system(cmd)

    m_out = Mesh(temp_file, hold_history=True, device=mesh.device)
    fname = os.path.join(save_path, 'recon_{}_after.obj'.format(len(m_out.faces)))
    m_out.export(fname)
    [os.remove(_) for _ in list(glob.glob(os.path.splitext(temp_file)[0] + '*'))]
    return m_out
