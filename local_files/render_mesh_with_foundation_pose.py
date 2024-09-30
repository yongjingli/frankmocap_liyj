import os
import shutil

import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'black'

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
# !wget https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/docs/tutorials/utils/plot_image_grid.py
from plot_image_grid import image_grid
from pytorch3d.renderer import PerspectiveCameras
import pytorch3d
import time



def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0,255,0), linewidth=2):
  '''Revised from 6pack dataset/inference_dataset_nocs.py::projection
  @bbox: (2,3) min/max
  @line_color: RGB
  '''
  min_xyz = bbox.min(axis=0)
  xmin, ymin, zmin = min_xyz
  max_xyz = bbox.max(axis=0)
  xmax, ymax, zmax = max_xyz

  def draw_line3d(start,end,img):
    pts = np.stack((start,end),axis=0).reshape(-1,3)
    pts = (ob_in_cam@to_homo(pts).T).T[:,:3]   #(2,3)
    projected = (K@pts.T).T
    uv = np.round(projected[:,:2]/projected[:,2].reshape(-1,1)).astype(int)   #(2,2)
    img = cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)
    return img

  for y in [ymin,ymax]:
    for z in [zmin,zmax]:
      start = np.array([xmin,y,z])
      end = start+np.array([xmax-xmin,0,0])
      img = draw_line3d(start,end,img)

  for x in [xmin,xmax]:
    for z in [zmin,zmax]:
      start = np.array([x,ymin,z])
      end = start+np.array([0,ymax-ymin,0])
      img = draw_line3d(start,end,img)

  for x in [xmin,xmax]:
    for y in [ymin,ymax]:
      start = np.array([x,y,zmin])
      end = start+np.array([0,0,zmax-zmin])
      img = draw_line3d(start,end,img)

  return img


def getcamera(width, height, R, T, K, device):
    T = T.reshape(3)
    R[0, :] = -R[0, :]
    T[0] = -T[0]
    R[1, :] = -R[1, :]
    T[1] = -T[1]
    R = R.t()
    fx, _, cx, _, fy, cy, _, _, _ = K.reshape(9)
    cameras = PerspectiveCameras(
        image_size=[[height, width]],
        R=R[None],
        T=T[None],
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32),
        in_ndc=False,
        device=device
    )
    return cameras

def load_mesh_without_texture(mesh_path, device, color=[0.0, 0.0, 0.0]):
    verts, faces_idx, _ = load_obj(mesh_path, device=device)
    faces = faces_idx.verts_idx

    # -Initialize each vertex to bewhite in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    verts_rgb[:, :, 0] = color[0]
    verts_rgb[:, :, 1] = color[1]
    verts_rgb[:, :, 2] = color[2]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    return mesh

def render_mesh_with_foundation_pose():
    d_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/foundation_pose_data"

    obj_img_path = os.path.join(d_root, "obj_img.jpg")
    cam_K_path = os.path.join(d_root, "cam_K.npy")
    obj_pose_path = os.path.join(d_root, "obj_pose.npy")
    obj_box_path = os.path.join(d_root, "obj_box.npy")
    obj_mesh_path = os.path.join(d_root, "obj_mesh.obj")
    hand_mesh_path = os.path.join(d_root, "shadow_hand.obj")

    img = cv2.imread(obj_img_path)
    cam_k = np.load(cam_K_path)
    obj_pose = np.load(obj_pose_path)
    obj_box = np.load(obj_box_path)
    obj_mesh = trimesh.load(obj_mesh_path)

    vis = draw_posed_3d_box(cam_k, img=img, ob_in_cam=obj_pose, bbox=obj_box)

    plt.subplot(2, 1, 1)
    plt.imshow(vis[:, :, ::-1])

    # render mesh obj
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    img_h, img_w, _ = img.shape
    # verts, faces = pytorch3d.io.load_ply(obj_mesh_path)
    # verts, faces = obj_mesh.vertices, obj_mesh.faces
    # verts = verts.astype(np.float32)
    # faces = faces.astype(np.float32)
    # verts = torch.from_numpy(verts)
    # faces = torch.from_numpy(faces)
    #
    # if hasattr(obj_mesh.visual, 'uv') and hasattr(obj_mesh.visual, 'material'):
    #     texture_image = obj_mesh.visual.material.image
    #     uv = torch.tensor(obj_mesh.visual.uv, dtype=torch.float32)
    #     textures = TexturesUV(maps=[texture_image], faces_uvs=[faces], verts_uvs=[uv])
    # else:
    #     textures = None
    #
    # mesh = pytorch3d.structures.Meshes(
    #     verts=[verts],
    #     faces=[faces],
    #     # textures=TexturesVertex(verts_features=torch.ones_like(verts)[None]),
    #     textures=[textures],
    # )
    # mesh = mesh.to(device)

    # mesh = load_objs_as_meshes([obj_mesh_path], device=device)
    # mesh = load_objs_as_meshes([hand_mesh_path], device=device)
    mesh = load_mesh_without_texture(hand_mesh_path, device, color=[255/255, 0/255, 0/255])

    # obj_pose = np.linalg.inv(obj_pose)
    obj_pose = torch.from_numpy(obj_pose)
    cam_k = torch.from_numpy(cam_k)

    R = obj_pose[:3, :3]
    T = obj_pose[:3, 3]

    cameras = getcamera(img_w, img_h, R, T, cam_k, device)

    # 获取深度图
    # rasterizer = MeshRasterizer(
    #     cameras=cameras,
    #     raster_settings=RasterizationSettings(
    #         image_size=((img_h, img_w)),
    #     ),
    # )
    # fragments = rasterizer(meshes_world=mesh)
    # depth_img = fragments.zbuf[0, :, :, 0].detach().cpu().numpy()

    # images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off");
    # plt.imshow(depth_img)

    # 获取渲染图
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    # Change specular color to green and change material shininess
    materials = Materials(
        device=device,
        specular_color=[[0.0, 1.0, 0.0]],
        shininess=10.0
    )

    raster_settings = RasterizationSettings(
        image_size=[img_h, img_w],
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)
    rendered_image = images[0, ..., :3].cpu().numpy()
    rendered_image = rendered_image[:, :, ::-1]

    # existing_image = Image.open(obj_img_path)
    # background_pixels = np.all(rendered_image < 10, axis=-1)
    result_image = vis.copy()
    # result_image[~background_pixels] = rendered_image[~background_pixels]
    mask = np.all(rendered_image ==1, axis=-1)

    result_image[~mask] = rendered_image[~mask] * 255
    plt.subplot(2, 1, 2)
    # plt.imshow(mask)
    plt.imshow(result_image[:, :, ::-1])
    plt.axis("off");
    plt.show()

    # cv2.imwrite(os.path.join(d_root, "obj_img_render.jpg"), result_image[:, :, ::-1])


def render_mesh_with_foundation_pose_multi():
    root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855"
    obj_mesh_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/untitled.obj"
    # hand_mesh_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/shadow_hand.obj"
    hand_mesh_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/tora_hand.obj"

    img_root = os.path.join(root, "colors")
    depth_root = os.path.join(root, "depths")

    vis_img_root = os.path.join(root, "colors_vis")
    pose_root = os.path.join(root, "poses")
    # render_root = os.path.join(root, "render")
    render_root = os.path.join(root, "render_tora")

    if os.path.exists(render_root):
        shutil.rmtree(render_root)
    os.mkdir(render_root)

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)
    cam_k = torch.from_numpy(cam_k)

    mask_root = os.path.join(root, "masks")

    # render mesh obj
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # mesh = load_mesh_without_texture(hand_mesh_path, device, color=[255/255, 0/255, 0/255])
    mesh = load_mesh_without_texture(hand_mesh_path, device, color=[200/255, 200/255, 200/255])
    # mesh = load_objs_as_meshes([obj_mesh_path], device=device)

    for img_name in img_names:

        img_path = os.path.join(img_root, img_name)
        vis_img_path = os.path.join(vis_img_root, img_name)
        depth_path = os.path.join(depth_root, img_name.replace("_color.jpg", "_depth.npy"))
        pose_path = os.path.join(pose_root, img_name.replace("_color.jpg", "_pose.npy"))

        img = cv2.imread(img_path)
        vis_img = cv2.imread(vis_img_path)
        depth = np.load(depth_path)
        obj_pose = np.load(pose_path)

        img_h, img_w, _ = img.shape

        obj_pose = torch.from_numpy(obj_pose)

        if np.all(obj_pose == np.eye(4)):
            result_image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        else:
            t1 = time.time()
            R = obj_pose[:3, :3]
            T = obj_pose[:3, 3]

            cameras = getcamera(img_w, img_h, R, T, cam_k, device)
            # lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
            # Change specular color to green and change material shininess
            materials = Materials(
                device=device,
                # specular_color=[[0.0, 1.0, 0.0]],
                specular_color=[[1.0, 1.0, 1.0]],
                shininess=10.0
            )

            raster_settings = RasterizationSettings(
                image_size=[img_h, img_w],
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0,
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device,
                    cameras=cameras,
                    lights=lights
                )
            )

            t2 = time.time()
            images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)
            rendered_image = images[0, ..., :3].cpu().numpy()
            rendered_image = rendered_image[:, :, ::-1]

            t3 = time.time() - t2
            print("t3:", t3)

            result_image = img.copy()
            mask = np.all(rendered_image == 1, axis=-1)

            result_image[~mask] = result_image[~mask] * 0.1 + rendered_image[~mask] * 255 * 0.9

        render_path = os.path.join(render_root, img_name.replace("_color.jpg", "_render.jpg"))
        cv2.imwrite(render_path, result_image)

        # plt.subplot(2, 1, 1)
        # plt.imshow(vis_img[:, :, ::-1])
        #
        # plt.subplot(2, 1, 2)
        # # plt.imshow(mask)
        # plt.imshow(result_image[:, :, ::-1])

        # plt.imshow(result_image[:, :, ::-1])
        #
        # plt.axis("off");
        # plt.show()
        #
        # exit(1)


def load_mesh():
    mesh_file = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/untitled.obj"
    mesh = trimesh.load(mesh_file, force='mesh')  # 从blender导出的模型需要设置force='mesh', 要不然没有vertices和vertex_normals这些属性
    mesh.apply_scale(0.001)  # mesh模型的单位是毫米，需要将其设置为米的单位 深度图输入的单位也是米
    mesh.export("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/untitled2.obj")

def combine_imgs():
    root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855"
    # root0 = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855/colors_vis"
    root0 = os.path.join(root, "colors_vis")
    root1 = os.path.join(root, "render_tora")
    root2 = os.path.join(root, "vis_pose")

    s_root = os.path.join(root, "combine_render_tora2")
    os.makedirs(s_root, exist_ok=True)

    img_names0 = [name for name in os.listdir(root0) if name[-4:] in [".jpg", ".png"]]
    img_names0 = list(sorted(img_names0, key=lambda x: int(x.split(".")[0].split("_")[0])))

    img_names1 = [name for name in os.listdir(root1) if name[-4:] in [".jpg", ".png"]]
    img_names1 = list(sorted(img_names1, key=lambda x: int(x.split(".")[0].split("_")[0])))

    img_names2 = [name for name in os.listdir(root2) if name[-4:] in [".jpg", ".png"]]
    img_names2 = list(sorted(img_names2, key=lambda x: int(x.split(".")[0].split("_")[0])))

    for img_name0, img_name1, img_name2 in zip(img_names0, img_names1, img_names2):
        img_path0 = os.path.join(root0, img_name0)
        img_path1 = os.path.join(root1, img_name1)
        img_path2 = os.path.join(root2, img_name2)

        img0 = cv2.imread(img_path0)
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        img_comb = cv2.hconcat([img0, img1, img2])
        s_img_path = os.path.join(s_root, img_name0)
        cv2.imwrite(s_img_path, img_comb)


def save_img_2_video():
    # root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855/combine_render"
    # root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855/combine_render_tora"
    root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855/combine_render_tora2"
    images = [img for img in os.listdir(root) if img.endswith(".jpg") or img.endswith(".png")]
    images = list(sorted(images, key=lambda x: int(x.split(".")[0].split("_")[0])))

    frame = cv2.imread(os.path.join(root, images[0]))
    height, width, layers = frame.shape

    video_path = os.path.join(root, "output.mp4")
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for image in images:
        img_path = os.path.join(root, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    print("STart")
    # render_mesh_with_foundation_pose()
    # render_mesh_with_foundation_pose_multi()
    # combine_imgs()
    save_img_2_video()
    print("End")