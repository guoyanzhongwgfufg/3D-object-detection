# from detect import run
import copy
#standard import
from nxlib.item import NxLibItem
from nxlib.command import NxLibCommand
from nxlib.exception import NxLibException
import nxlib.api as api
# import all constants 
from nxlib.constants import *
#from urx import Robot
from scipy.spatial.transform import Rotation as R
#import common 
import time  
import open3d as o3d
import numpy as np
import json
import cv2
import os
import sys
from pathlib import Path
import logging

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

##########################################################################
#提前初始化相机函数
camera_serial = '232966'#args.serial、
api.initialize()
api.open_tcp_port(0,0)
PI=3.1415927
#  准备相机参数
cmd = NxLibCommand(CMD_OPEN)  
cmd.parameters()[ITM_CAMERAS] = camera_serial
cmd.execute()
json_param=''
with open('CameraSettings/camera_settings_232608.json', 'r') as json_file:
# 加载JSON数据
    data = json.load(json_file)
json_param=data['Parameters']    
json_param_text=json.dumps(json_param, indent=0)
camera = NxLibItem()[ITM_CAMERAS][camera_serial]
camera[ITM_PARAMETERS].set_json(json_param_text,True)
coodinate= None
# 提前初始化推理模型
from detectdemo import YoloV5Detect
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
detect = YoloV5Detect(ROOT / 'best.pt',data=ROOT / 'data/coco128.yaml')
detect.OpenModel()

#———————————————————————————相机拍照函数————————————————————————————————————
def acquire_image():
    start_camera_time =time.time()
    camera=NxLibItem()[ITM_CAMERAS][camera_serial]
   # camera[ITM_PARAMETERS][ITM_CAPTURE][ITM_EXPOSURE].set_double(5.0)

#———————————————————————— 第一次拍照，获得灰度图——————————————————————————————
#  拍照执行命令
    capture = NxLibCommand(CMD_CAPTURE)
    capture.parameters()[ITM_CAMERAS] = camera_serial
    capture.parameters()[ITM_TIMEOUT] = 6000
    # capture.execute()
    #  只获得2d图像
    captureParams = NxLibItem()[ITM_CAMERAS][camera_serial][ITM_PARAMETERS][ITM_CAPTURE]
    captureParams[ITM_PROJECTOR]=False
    captureParams[ITM_FRONT_LIGHT]=True
    #  拍照执行命令
    # capture = NxLibCommand(CMD_CAPTURE)
    # capture.parameters()[ITM_CAMERAS] = camera_serial
    capture.execute()
    ComputeDisparityMap= NxLibCommand(CMD_COMPUTE_DISPARITY_MAP)
    ComputeDisparityMap.parameters()[ITM_CAMERAS] = camera_serial
    ComputeDisparityMap.execute()
    #  获取灰度修正图
    rectified_map =  NxLibItem()[ITM_CAMERAS][camera_serial][ITM_IMAGES][ITM_RECTIFIED][ITM_LEFT]. get_binary_data()
    #  观察显采集是否成功
    # cv2.imshow("rectified_map",rectified_map )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("2D图像尺寸为:",rectified_map.shape)
    # 保存图像
    # cv2.imwrite('rectified_map.png',rectified_map)
# ————————————————————————第二次拍照，获得深度图tiff————————————————————————————
    #获得深度图像
    captureParams = NxLibItem()[ITM_CAMERAS][camera_serial][ITM_PARAMETERS][ITM_CAPTURE]
    captureParams[ITM_PROJECTOR]=True
    captureParams[ITM_FRONT_LIGHT]=True
    #拍照执行命令
    capture = NxLibCommand(CMD_CAPTURE)
    capture.parameters()[ITM_CAMERAS] = camera_serial
    capture.parameters()[ITM_TIMEOUT] = 6000
    capture.execute()
    #计算像差图
    ComputeDisparityMap= NxLibCommand(CMD_COMPUTE_DISPARITY_MAP)
    ComputeDisparityMap.parameters()[ITM_CAMERAS] = camera_serial
    ComputeDisparityMap.execute()
    #计算点云
    ComputePointMap= NxLibCommand(CMD_COMPUTE_POINT_MAP)
    ComputePointMap.parameters()[ITM_CAMERAS] = camera_serial  
    ComputePointMap.execute()
    # 获取左右原始图
    left_image_raw=  NxLibItem()[ITM_CAMERAS][camera_serial][ITM_IMAGES][ITM_RAW][ITM_LEFT]. get_binary_data()
    right_image_raw = NxLibItem()[ITM_CAMERAS][camera_serial][ITM_IMAGES][ITM_RAW][ITM_RIGHT].get_binary_data()
    binaryinfo=NxLibItem()[ITM_CAMERAS][camera_serial][ITM_IMAGES][ITM_RAW][ITM_LEFT] .get_binary_data_info() 
    #获取左右校正图
    left_image_Rectified=  NxLibItem()[ITM_CAMERAS][camera_serial][ITM_IMAGES][ITM_RECTIFIED][ITM_LEFT]. get_binary_data()
    right_image_Rectified = NxLibItem()[ITM_CAMERAS][camera_serial][ITM_IMAGES][ITM_RECTIFIED][ITM_RIGHT].get_binary_data()
    binaryinfo=NxLibItem()[ITM_CAMERAS][camera_serial][ITM_IMAGES][ITM_RECTIFIED][ITM_LEFT] .get_binary_data_info()
    width=binaryinfo[0]
    heigh=binaryinfo[1]
    #获取点云数据
    point_map=  NxLibItem()[ITM_CAMERAS][camera_serial][ITM_IMAGES][ITM_POINT_MAP]. get_binary_data()
    print("point_map.shape=",point_map.shape)
    print("point_map.dtype=",point_map.dtype)
    #观察深度图像
    # cv2.imshow("深度图", point_map) 
    # cv2.waitKey(6000)
    # cv2.destroyAllWindows()
    #转换画幅的长和宽为np数组
    img=np.array(rectified_map,dtype=np.uint8).reshape(heigh, width)
    #转换为32位浮点数
    image_point_map_np=np.array(point_map,dtype=np.float32).reshape( heigh , width,3)
    image_point_map= image_point_map_np#cv2.UMat(image_point_map_np)
    print(image_point_map.shape)
    # print(image_np.shape)
    end_camera_time=time.time()
    print("相机调用消耗时间=",end_camera_time-start_camera_time,"秒")

    return  rectified_map, image_point_map
# ——————————————————————————执行2次拍照命令结束—————————————————————————————————
#———————————————————————————相机拍照函数———————————————————————————————————————


# ——————————————————————————点云转换函数————————————————————————————————————————
def filter_nans(point_map):
    """ Filter NaN values. """
    return point_map[~np.isnan(point_map).any(axis=1)]
def reshape_point_map(point_map):
    """ Reshape the point map array from (m x n x 3) to ((m*n) x 3). """
    return point_map.reshape(
        (point_map.shape[0] * point_map.shape[1]), point_map.shape[2])
def convert_to_open3d_point_cloud(point_map):
    """ Convert numpy array to Open3D format. """
    point_map = reshape_point_map(point_map)
    point_map = filter_nans(point_map)
    open3d_point_cloud = o3d.geometry.PointCloud()
    open3d_point_cloud.points = o3d.utility.Vector3dVector(point_map)
    return open3d_point_cloud
# ——————————————————————————点云转换函数————————————————————————————————————————


# ———————————————————————————图像检测算法———————————————————————————————————————
if  __name__ == "__main__":
    
    # ----------------------take photo----------------------------
    camera_start_time = time.time()
    #采集图像
    image = acquire_image()[0]
    print("image.shape=",image.shape)
    #采集点云
    point_map=acquire_image()[1]
    print(f"深度图的尺寸:{point_map.shape},深度图的类型为{point_map.dtype}")
    pointcloud=convert_to_open3d_point_cloud(point_map)
    camrea_end_time = time.time()
    camera_consume_time = camrea_end_time -camera_start_time

     # ----------------------take photo----------------------------
    

    # ---------------single detection model detection--------------
    single_model_start_time = time.time() 
    three_channel_image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)#单通道变为3通道
    print("单通道变为3通道",three_channel_image.shape)
    if  three_channel_image is not None:
    # 显示图像
    #   cv2.imshow('Image', image)
    #   cv2.waitKey(6000)
    #   cv2.destroyAllWindows()
    # model run time
    #   yolo_start_time = time.time()
      target_x,target_y,target_w,target_h,target_label,target_scorse,result_image=detect.Run(three_channel_image,0.5,1)
      if(len(target_x)==0):
        print("没有检测到目标")
      else: 
        print(f"矩形型框右上值x点坐标为{target_x[0]},")
    #   yolo_end_time = time.time()
    #   yolo_consume_time =yolo_end_time- yolo_start_time
    single_model_end_time = time.time()
    single_model_consume_time = single_model_end_time -single_model_start_time
    
    # ---------------single detection model detection--------------
        
    # o3d.visualization.draw_geometries([pointcloud])
    # o3d.io.write_point_cloud("full scenes.pcd",pointcloud)
    # cv2.imwrite('5.png',point_map)
   
#方法一映射深度图，直接深度图中心点
    # index_cols=int((target_x[0]-target_w[0])/2+target_w[0])
    # index_rows=int((target_y[0]-target_h[0])/2+target_h[0])
    # print(f"列索引={index_cols},行索引{index_rows}") 
    # print('圆环中心点提取初步的x,y,z坐标是:',point_map[index_rows, index_cols][0], \
    #       point_map[index_rows, index_cols][1],point_map[index_rows, index_cols][2])


#方法二：点云裁切求ransac中心点
refine_point_start_time = time.time()
#----------------------------------提取点云----------------------------------------------
# 裁剪深度图和灰度图
left_x = int(target_x[0] - target_w[0] / 2)
right_x = int(target_x[0] + target_w[0] / 2)
top_y = int(target_y[0] - target_h[0] / 2)
bottom_y = int(target_y[0] + target_h[0] / 2)
crop=three_channel_image[top_y:bottom_y, left_x:right_x].copy()
# cv2.imshow('CropImage', crop)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('crop_image.jpg',crop)
# 创建一个与原始图像相同大小的空掩码
blank_image = np.zeros((571, 909,3), dtype=np.uint8) # 取一个通道的尺寸

# 指定放置图像的位置
blank_image[top_y :bottom_y, left_x :right_x , :] = 1
print("裁切放回置图的尺寸和类型={blank_image.shape}{blank_image.dtype}")
new_img = image * blank_image
cv2.imwrite('composite_img.png',new_img )
cv2.imshow('composite_img', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("new_img",new_img)
print(f"组合图图的尺寸和类型={new_img.shape}{new_img.dtype}")


#进一步提取轮廓
blur = cv2.GaussianBlur(new_img, (5, 5), 0)
ret, mask = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)  
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('mask.png',mask)
print(f"掩码图的尺寸和类型={mask.shape}{mask.dtype}")


# 应用掩码到深度图,非零区域等于1
mask[mask!=0]=1
mask_bool=mask.astype(bool)
#数组相乘得到理想roi数组
result=np.where(mask_bool,point_map,0.0)
#转换为深度为点云
maskpointcloud=convert_to_open3d_point_cloud(result)
o3d.visualization.draw_geometries([maskpointcloud])

# 点云转为numpy
points = np.asarray(maskpointcloud.points)
# 去除所有0值
arr = result[result != 0].reshape(-1,3)
mark_point_cloud = o3d.geometry.PointCloud()
mark_point_cloud.points = o3d.utility.Vector3dVector(arr) 
#可视化
o3d.visualization.draw_geometries([mark_point_cloud])
print("mark点云数量",mark_point_cloud.scale)


#求最小包围盒
aabb= mark_point_cloud.get_axis_aligned_bounding_box()
aabb.color=(1,0,0)
obb = mark_point_cloud.get_oriented_bounding_box()
obb.color=(0,1,0)
# print(f"obb={obb}")
print("最小包围盒=",aabb)


#可视化
# o3d.visualization.draw_geometries([mark_point_cloud,obb])
# 将点云转换为 NumPy 数组
# points = np.asarray(maskpointcloud.points)
# arr = points[points!=0].reshape(-1, 3)


# 计算质心
centroid = np.mean(arr, axis=0)
print("centroid=", centroid)
# 创建一个包含质心的点云
centroid_pcd = o3d.geometry.PointCloud()
centroid_pcd = o3d.geometry.PointCloud()
centroid_pcd.points = o3d.utility.Vector3dVector([centroid])
# 为质心点设置颜色（例如红色）
centroid_color = [1, 0, 0]  # 红色
centroid_pcd.colors = o3d.utility.Vector3dVector([centroid_color])
# 可以为原始点云设置不同的颜色，以便区分
mark_point_cloud.paint_uniform_color([0, 0, 1])  # 蓝色
pointcloud.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色


#观察点云
# o3d.visualization.draw_geometries([mark_point_cloud,centroid_pcd])
# o3d.io.write_point_cloud("crop_point_cloud0.pcd",mark_point_cloud)
# 使用 Open3D 绘制两个点云
o3d.visualization.draw_geometries([pointcloud, mark_point_cloud, aabb,centroid_pcd])
# 获取旋转矩阵
rotation_matrix = obb.R
R = copy.copy(rotation_matrix)
print(f"mark的最小包围盒旋转矩阵=:\n{R}")
refine_point_end_time = time.time()
refine_point_consume_time = refine_point_end_time - refine_point_start_time
run_time = refine_point_end_time - camera_start_time

logging.info(f'相机消耗时间：{camera_consume_time}秒')
# logging.info(f'yolo消耗时间：{yolo_consume_time}秒')
logging.info(f'单步目标检测网络消耗时间：{single_model_consume_time }秒')
logging.info(f'点云细化消耗运行时间：{refine_point_consume_time}秒')
logging.info(f'合计相机+算法运行时间：{run_time}秒')
# ———————————————————————————图像检测算法———————————————————————————————————————



