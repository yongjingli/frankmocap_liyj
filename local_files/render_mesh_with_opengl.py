import numpy as np
import cv2
import os
from renderer.visualizer import Visualizer
import trimesh
os.chdir("../")


def save_2_ply(file_path, x, y, z, color=None):
    points = []
    if color == None:
        color = [[255, 255, 255]] * len(x)
    for X, Y, Z, C in zip(x, y, z, color):
        points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, C[2], C[1], C[0]))

    # for X, Y, Z, C in zip(x, y, z, color):
    #     points.append("%f %f %f %d %d %d 0\n" % (Z, X, Y, C[0], C[1], C[2]))

    file = open(file_path, "w")
    file.write('''ply
          format ascii 1.0
          element vertex %d
          property float x
          property float y
          property float z
          property uchar red
          property uchar green
          property uchar blue
          property uchar alpha
          end_header
          %s
          ''' % (len(points), "".join(points)))
    file.close()


def render_dex_grasp():



    faces = np.load(
        "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_data/faces.npy")
    vertices = np.load(
        "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_data/vertices.npy")
    vertices = vertices * 1000
    vertices[:, 0] = vertices[:, 0] + 1000

    # vertices[:, 2] = vertices[:, 2] * 10000
    # vertices = vertices * 1000
    # pred_mesh_list.append({"vertices": vertices, "faces": faces})

    faces2 = np.load(
        "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_data/1/faces.npy")
    vertices2 = np.load(
        "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_data/1/vertices.npy")

    mesh3 = trimesh.load('/home/pxn-lyj/Downloads/models/020/textured.obj')
    vertices3 = mesh3.vertices
    faces3 = mesh3.faces
    vertices3 = vertices3 * 1000
    faces3 = faces3


    save_2_ply("vertices.ply", vertices[:, 0], vertices[:, 1], vertices[:, 2])
    save_2_ply("vertices2.ply", vertices2[:, 0], vertices2[:, 1], vertices2[:, 2])
    save_2_ply("vertices3.ply", vertices3[:, 0], vertices2[:, 1], vertices3[:, 2])

    visualizer = Visualizer("opengl")
    img_original_bgr = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    pred_mesh_list = []
    pred_mesh_list.append({"vertices": vertices, "faces": faces})
    pred_mesh_list.append({"vertices": vertices2, "faces": faces2})
    pred_mesh_list.append({"vertices": vertices3, "faces": faces3})
    res_img = visualizer.visualize(
        img_original_bgr,
        pred_mesh_list=pred_mesh_list,)

    cv2.imshow("res_img", res_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    print("Start")
    render_dex_grasp()
    print("End")
