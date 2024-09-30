from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

import os
import torch
import matplotlib.pyplot as plt

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


def office_example():
    # Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
    sphere_mesh = ico_sphere(level=3)
    verts, faces, _ = load_obj("/home/pxn-lyj/Downloads/models/020/textured.obj")
    test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

    # Differentiably sample 5k points from the surface of each mesh and then compute the loss.
    sample_sphere = sample_points_from_meshes(sphere_mesh, 5000)
    sample_test = sample_points_from_meshes(test_mesh, 5000)
    loss_chamfer, _ = chamfer_distance(sample_sphere, sample_test)
    print(loss_chamfer)


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


def render_mesh_office():
    # 官网教程
    # https://pytorch3d.org/tutorials/render_textured_meshes
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    # DATA_DIR = "./data"
    # obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

    # Load obj file
    # obj_filename = "/home/pxn-lyj/Downloads/models/020/textured.obj"
    # obj_filename = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/meshs/cow.obj"
    # mesh = load_objs_as_meshes([obj_filename], device=device)

    obj_filename = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/foundation_pose_data/shadow_hand.obj"
    mesh = load_mesh_without_texture(obj_filename, device, color=[0.0, 1.0, 0.0])

    # show mesh textures maps
    # plt.figure(figsize=(7, 7))
    # texture_image = mesh.textures.maps_padded()
    # plt.imshow(texture_image.squeeze().cpu().numpy())
    # # texturesuv_image_matplotlib(mesh.textures, subsample=None)
    # plt.axis("off");
    # plt.show()

    # 有关camera和pytorch3d坐标系的相关关系
    # https://zhuanlan.zhihu.com/p/651937759
    # https://blog.csdn.net/qq_26239785/article/details/139008893
    # camera  相机的Z轴超前、X轴朝右、Y轴朝下
    # pytorch3d 相机的Z轴超前、X轴朝左、Y轴朝上（LUF)

    # 还有一个最重要的问题：pytorch3d中的3D点是按照Nx3的格式存储的，因此 在计算过程中 Point= point@R+t 而不是 Point=R @ point+t,所以在传入参数时应使用 R.t()
    # ps：如果使用R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    # 生成的R和T已经是转置好了的，因此就不必再转置

    # create renderer
    # dist (1), (N) – distance of the camera from the object
    # elev – angle in degrees or radians.
    # This is the angle between the vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
    # azim – angle in degrees or radians.
    # The vector from the object to the camera is projected onto a horizontal plane y = 0. azim is the angle between the projected vector and a reference vector at (0, 0, 1) on the reference plane (the horizontal plane).

    # 这里的坐标基准都是以物体的坐标系而言
    # 2.7 设置object为camera前面的2.7m
    # 将z轴调转180度，设置为-z
    # elev 与xz平面 也就是pitch角的方向， 不改变
    R, T = look_at_view_transform(2.7, 0, 180)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # 设置渲染的参数
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
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

    images = renderer(mesh)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off");
    plt.show()

    # 将光源放置到物体的背后
    # Now move the light so it is on the +Z axis which will be behind the cow.
    lights.location = torch.tensor([0.0, 0.0, +1.0], device=device)[None]
    images = renderer(mesh, lights=lights)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off");
    plt.show()


    # 修改相机的位置
    # Rotate the object by increasing the elevation and azimuth angles
    # R, T = look_at_view_transform(dist=2.7, elev=10, azim=-150)
    R, T = look_at_view_transform(dist=2.7, elev=10, azim=-180)    # 修改相机pitch角
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Move the light location so the light is shining on the cow's face.
    lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=device)

    # Change specular color to green and change material shininess
    materials = Materials(
        device=device,
        specular_color=[[0.0, 1.0, 0.0]],
        shininess=10.0
    )

    # Re render the mesh, passing in keyword arguments for the modified components.
    images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)

    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.show()

    # 一下子采用20个相机的参数进行batchsize的渲染
    # Set batch size - this is the number of different viewpoints from which we want to render the mesh.
    batch_size = 20

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(batch_size)

    # Get a batch of viewing angles.
    elev = torch.linspace(0, 180, batch_size)
    azim = torch.linspace(-180, 180, batch_size)

    # All the cameras helper methods support mixed type inputs and broadcasting. So we can
    # view the camera from the same distance and specify dist=2.7 as a float,
    # and then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Move the light back in front of the cow which is facing the -z direction.
    lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)

    images = renderer(meshes, cameras=cameras, lights=lights)
    image_grid(images.cpu().numpy(), rows=4, cols=5, rgb=True)

    plt.show()


def plotly_visualizetion():
    # 主要是显示在网页上
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    # DATA_DIR = "./data"
    # obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

    # Load obj file
    # obj_filename = "/home/pxn-lyj/Downloads/models/020/textured.obj"
    # obj_filename = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/meshs/cow.obj"
    # mesh = load_objs_as_meshes([obj_filename], device=device)

    obj_filename = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/foundation_pose_data/shadow_hand.obj"
    # mesh = load_mesh_without_texture(obj_filename, device, color=[0.0, 1.0, 0.0])


    verts, faces_idx, _ = load_obj(obj_filename)
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object
    # mesh = Meshes(
    #     verts=[verts.to(device)],
    #     faces=[faces.to(device)],
    #     textures=textures    # 全部设置为白色了
    # )

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)]   # plotly的默认颜色
    )

    # Render the plotly figure
    fig = plot_scene({
        "subplot1": {
            "cow_mesh": mesh
        }
    })
    fig.show()

    # 多个mesh 只是改变verts就可以达到将xyz都平移2米的效果
    mesh_batch = Meshes(
        verts=[verts.to(device), (verts + 2).to(device)],
        faces=[faces.to(device), faces.to(device)]
    )

    # plot mesh batch in the same trace
    fig = plot_scene({
        "subplot1": {
            "cow_mesh_batch": mesh_batch
        }
    })
    fig.show()

    # 在同一网页视图里显示不同的mesh模型，可以分开显示操作旋转视图
    # plot batch of meshes in different subplots
    fig = plot_scene({
        "subplot1": {
            "cow_mesh1": mesh_batch[0]
        },
        "subplot2": {
            "cow_mesh2": mesh_batch[1]
        }
    })
    fig.show()

    # 直接对mesh模型进行复制的操作
    # extend the batch to have 4 meshes
    mesh_4 = mesh_batch.extend(2)

    # visualize the batch in different subplots, 2 per row
    fig = plot_batch_individually(mesh_4)
    # we can update the figure height and width
    fig.update_layout(height=1000, width=500)
    fig.show()

    # 将坐标轴平面用不同的颜色进行显示
    fig2 = plot_scene({
        "cow_plot1": {
            "cows": mesh_batch
        }
    },
        xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
        yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
        zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
        axis_args=AxisArgs(showgrid=True))
    fig2.show()

    fig3 = plot_batch_individually(
        mesh_4,
        ncols=2,
        subplot_titles=["cow1", "cow2", "cow3", "cow4"],  # customize subplot titles
        xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
        yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
        zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
        axis_args=AxisArgs(showgrid=True))
    fig3.show()


def render_mesh_in_camera_coordinate():
    # 已知物体在相机坐标系下的坐标,得到该相机坐标系下的渲染图像
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

    def getdepth(model_path, R, T, K, device=torch.device('cuda')):
        width, height = 640, 480
        verts, faces = pytorch3d.io.load_ply(model_path)
        mesh = pytorch3d.structures.Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts)[None]),
        )
        mesh = mesh.to(device)

        cameras = getcamera(width, height, R, T, K)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=((height, width)),
            ),
        )
        fragments = rasterizer(meshes_world=mesh)

        return fragments.zbuf[0, :, :, 0], cameras

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # getdepth(model_path, R, T, K, device=device)


if __name__ == "__main__":
    print("Start")
    # office_example()
    # render_mesh_office()
    plotly_visualizetion()
    print("End")