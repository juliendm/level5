
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import tqdm
from tqdm import tqdm as prog_bar

from skimage import io
from PIL import Image

import pandas as pd
import plotly.graph_objects as go
import numpy as np

import os, glob
import pickle

from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

import random

from second.core import box_np_ops
from second.core.point_cloud.point_cloud_ops import bound_points_jit
from second.data import kitti_common as kitti

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from itertools import compress

import numba


name_map = {
    "car"               : "car",
    "truck"             : "truck",
    "bus"               : "bus",
    "pedestrian"        : "pedestrian",
    "bicycle"           : "bicycle",
    "motorcycle"        : "motorcycle",
    "other_vehicle"     : "other_vehicle",
    "emergency_vehicle" : "emergency_vehicle",
    "animal"            : "animal"
}

name_map_reverse = {
    "car"               : "car",
    "truck"             : "truck",
    "bus"               : "bus",
    "pedestrian"        : "pedestrian",
    "bicycle"           : "bicycle",
    "motorcycle"        : "motorcycle",
    "other_vehicle"     : "other_vehicle",
    "emergency_vehicle" : "emergency_vehicle",
    "animal"            : "animal"
}

# # CAMERA DATA
# cs_record = lyftdata.get("calibrated_sensor", lyftdata.get("sample_data", sample['data']['CAM_FRONT'])["calibrated_sensor_token"])
# print(cs_record)


def create_level5_infos(level5_data_train, level5_data_test, lyftdata_train, lyftdata_test):

    level5_infos_train = []
    level5_infos_val = []
    level5_infos_test = []

    random.seed(42)
    random_index = np.arange(len(level5_data_train))
    random.shuffle(random_index)
    sep = int(0.8*len(level5_data_train))

    train_index = random_index[:sep]
    val_index = random_index[sep:]

    for index in prog_bar(train_index):
        sample_data = create_sample_data(level5_data_train,lyftdata_train,index,annotations=True)
        level5_infos_train.append(sample_data)

    for index in prog_bar(val_index):
        sample_data = create_sample_data(level5_data_train,lyftdata_train,index,annotations=True)
        level5_infos_val.append(sample_data)

    for index in prog_bar(range(len(level5_data_test))):
        sample_data = create_sample_data(level5_data_test,lyftdata_test,index,annotations=False)
        level5_infos_test.append(sample_data)

    return level5_infos_train, level5_infos_val, level5_infos_test

def create_sample_data(level5_data,lyftdata,index,annotations=False):

    token = level5_data.iloc[index]['Id']

    sample = lyftdata.get('sample', token)

    sample_data = {}
    sample_data['image_idx'] = index
    sample_data['level5_token'] = token
    sample_data['pointcloud_num_features'] = 5

    sample_image = lyftdata.get('sample_data',sample['data']['CAM_FRONT'])
    sample_data['img_path'] = sample_image['filename']

    sample_lidar = lyftdata.get('sample_data',sample['data']['LIDAR_TOP'])
    sample_data['velodyne_path'] = os.path.join(lyftdata.data_path,sample_lidar['filename'])

    if annotations:

        annos = {}

        # print(level5_data.iloc[index]['PredictionString'])

        annos['name'] =  []
        annos['dimensions'] = []
        annos['location'] =  []
        annos['rotation_y'] = []
        annos['truncated'] = []
        annos['occluded'] = []
        annos['bbox'] = []
        annos['score'] = []
        annos['alpha'] = []
        annos['difficulty'] = []

        _, boxes, _ = lyftdata.get_sample_data(sample['data']['LIDAR_TOP'], flat_vehicle_coordinates=False)

        for box in boxes:

            # lidar_to_world(box,sample['data']['LIDAR_TOP'])

            annos['name'].append(name_map[box.name])
            annos['dimensions'].append([box.wlh[1],box.wlh[2],box.wlh[0]])   #       # l, h, w --> w, l, h   # Check that kitty_info_val.pkl has l, h, w
            #annos['location'].append([-box.center[1],-box.center[2],box.center[0]])
            annos['location'].append([box.center[0],box.center[1],box.center[2]-box.wlh[2]/2.0])
            annos['rotation_y'].append(-box.orientation.yaw_pitch_roll[0]+np.pi/2.0)
            annos['truncated'].append(0.0)
            annos['occluded'].append(0)
            annos['bbox'].append([0,0,0,0])
            annos['score'].append(0)
            annos['alpha'].append(0.0) # Does not seem to be used in training
            annos['difficulty'].append(0) # done by add_difficulty_to_annos(sample_data)

        annos['name'] = np.array(annos['name'])
        annos['dimensions'] = np.array(annos['dimensions'])
        annos['location'] = np.array(annos['location'])
        annos['rotation_y'] = np.array(annos['rotation_y'])
        annos['truncated'] = np.array(annos['truncated'])
        annos['occluded'] = np.array(annos['occluded'])
        annos['bbox'] = np.array(annos['bbox'])
        annos['score'] = np.array(annos['score'])
        annos['alpha'] = np.array(annos['alpha'])
        annos['difficulty'] = np.array(annos['difficulty'])

        num_objects = len(annos['name'])
        
        annos['index'] = np.arange(num_objects, dtype=np.int32)
        annos['group_ids'] = np.arange(num_objects, dtype=np.int32)

        # annos['num_points_in_gt'] = np.array([]) # done by _calculate_num_points_in_gt , remove outside False

        sample_data['annos'] = annos
      
  
    sample_data['calib/R0_rect'] = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                                             [ 0.        ,  1.        ,  0.        ,  0.        ],
                                             [ 0.        ,  0.        ,  1.        ,  0.        ],
                                             [ 0.        ,  0.        ,  0.        ,  1.        ]])     
    sample_data['calib/Tr_velo_to_cam'] = np.array([[ 1.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
                                                    [ 0.000000e+00,  1.000000e+00,  0.000000e+00,  0.000000e+00],
                                                    [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  0.000000e+00],
                                                    [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])

    # IMPROVE !!!!!!!!!!!!!
    sample_data['img_shape'] = np.array([ 375, 1242], dtype=np.int32) # image_info['img_shape'] = np.array(io.imread(img_path).shape[:2], dtype=np.int32)
    sample_data['calib/P2'] = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                                        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                                        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
                                        [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])

    return sample_data

def lidar_to_world(box,lyftdata,lidar_top_token): # sample['data']['LIDAR_TOP']
    
    sd_record = lyftdata.get("sample_data", lidar_top_token)
    cs_record = lyftdata.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = lyftdata.get("ego_pose", sd_record["ego_pose_token"])

    box.rotate(Quaternion(cs_record["rotation"]))
    box.translate(np.array(cs_record["translation"]))

    box.rotate(Quaternion(pose_record["rotation"]))
    box.translate(np.array(pose_record["translation"]))

def pred_to_submission(submission,level5_infos,lyftdata,result,phi=0.0):

    for index in prog_bar(range(len(level5_infos))):
        
        sample = lyftdata.get('sample', level5_infos[index]['level5_token'])
        case = result[index]
        
        boxes = create_boxes(case,phi)
        score = case['score']

        world_boxes = []
        for box in boxes:
            lidar_to_world(box,lyftdata,sample['data']['LIDAR_TOP'])
            world_boxes.append(box)
      
        pred_str = ''
        for box_index,box in enumerate(world_boxes):
            pred_str += '%f %f %f %f %f %f %f %f %s ' % (score[box_index],box.center[0],box.center[1],box.center[2],box.wlh[0],box.wlh[1],box.wlh[2],box.orientation.radians,box.name)
        
        key = level5_infos[index]['level5_token']
        
        if key in submission:
            submission[key] += pred_str
        else:
            submission[key] = pred_str


def show_scene(level5_infos,lyftdata,result,index):

    v_filename = level5_infos[index]['velodyne_path'].split('/')
    v_filename[-2] += "_reduced"
    v_filename = '/'.join(v_filename)

    points_v = np.fromfile(v_filename, dtype=np.float32, count=-1).reshape([-1, 5])

    sample = lyftdata.get('sample', level5_infos[index]['level5_token'])
    _, boxes, _ = lyftdata.get_sample_data(sample['data']['LIDAR_TOP'], flat_vehicle_coordinates=False)

    boxes_pred = create_boxes(result[index])

    # Fig 1

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set_facecolor('black')
    ax.grid(False)

    ax.scatter(points_v[:, 0], points_v[:, 1], s=0.1, c="white", cmap='grey')

    for box in boxes:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[0,:],points[1,:],'r')
        
    for box in boxes_pred:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[0,:],points[1,:],'b')

    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)

    # Fig 2

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_facecolor('black')
    ax.grid(False)

    filter_index = [value for value in np.where(points_v[:, 0]> -10)[0] if value in np.where(points_v[:, 0]< 100)[0]]  
    ax.scatter(points_v[filter_index, 1], points_v[filter_index, 2], s=0.1, c="white", cmap='grey')

    for box in boxes:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[1,:],points[2,:],'r')

    for box in boxes_pred:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[1,:],points[2,:],'b')
        
    ax.set_xlim(-100,100)
    ax.set_ylim(-8,6)


def show_scene_3d(level5_infos,lyftdata,result,index):

    points_v = np.fromfile(level5_infos[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 5])

    df_tmp = pd.DataFrame(points_v[:, :3], columns=['x', 'y', 'z'])
    df_tmp['norm'] = np.sqrt(np.power(df_tmp[['x', 'y', 'z']].values, 2).sum(axis=1))
    scatter = go.Scatter3d(
        x=df_tmp['x'],
        y=df_tmp['y'],
        z=df_tmp['z'],
        mode='markers',
        marker=dict(
            size=1,
            color=df_tmp['norm'],
            opacity=0.8
        )
    )


    case = result[index]
    boxes = create_boxes(case)

    x_lines = []
    y_lines = []
    z_lines = []

    def f_lines_add_nones():
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    ixs_box_0 = [0, 1, 2, 3, 0]
    ixs_box_1 = [4, 5, 6, 7, 4]

    for box in boxes:
        points = view_points(box.corners(), view=np.eye(3), normalize=False)
        x_lines.extend(points[0, ixs_box_0])
        y_lines.extend(points[1, ixs_box_0])
        z_lines.extend(points[2, ixs_box_0])    
        f_lines_add_nones()
        x_lines.extend(points[0, ixs_box_1])
        y_lines.extend(points[1, ixs_box_1])
        z_lines.extend(points[2, ixs_box_1])
        f_lines_add_nones()
        for i in range(4):
            x_lines.extend(points[0, [ixs_box_0[i], ixs_box_1[i]]])
            y_lines.extend(points[1, [ixs_box_0[i], ixs_box_1[i]]])
            z_lines.extend(points[2, [ixs_box_0[i], ixs_box_1[i]]])
            f_lines_add_nones()

    lines = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        name='lines'
    )

    fig = go.Figure(data=[scatter, lines])
    fig.update_layout(scene_aspectmode='data')
    fig.show()

def create_boxes(case,phi=0.0):

    score = case['score']
    number = len(np.where(score>0.5)[0])

    loc = case['location']
    dim = case['dimensions']
    yaw = case['rotation_y']
    name = case['name']

    return create_boxes_from_val(loc,dim,yaw,name,number,phi)

def create_boxes_from_val(loc,dim,yaw,name,number,phi=0.0):

    boxes = []

    for box_index in range(number):

        x =  np.cos(phi)*loc[box_index][0] + np.sin(phi)*loc[box_index][1]
        y = -np.sin(phi)*loc[box_index][0] + np.cos(phi)*loc[box_index][1]
        angle = (yaw[box_index]+phi-np.pi/2.0)/2.0

        box = Box(
                [x,y,loc[box_index][2]+dim[box_index][1]/2.0],

                [dim[box_index][2],dim[box_index][0],dim[box_index][1]], # dim == lhw; need wlh
          
                Quaternion(scalar=np.cos(angle), vector=[0, 0, np.sin(angle)]).inverse,
          
                name=name_map_reverse[name[box_index]],
                token="token",
            )
        boxes.append(box)

    return boxes

def show_grounds(level5_infos,lyftdata,index):

    points_v = np.fromfile(level5_infos[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 5])

    sample = lyftdata.get('sample', level5_infos[index]['level5_token'])
    _, boxes, _ = lyftdata.get_sample_data(sample['data']['LIDAR_TOP'], flat_vehicle_coordinates=False)

      
    loc = level5_infos[index]['annos']['location']
    dim = level5_infos[index]['annos']['dimensions']
    yaw = level5_infos[index]['annos']['rotation_y']
    # yaw = []
    # for box in boxes:
    #   yaw.append(-box.orientation.yaw_pitch_roll[0]+np.pi/2.0)
    
    names = level5_infos[index]['annos']['name']

    rect = level5_infos[index]['calib/R0_rect']
    P2 = level5_infos[index]['calib/P2']
    Trv2c = level5_infos[index]['calib/Tr_velo_to_cam']
    annos = level5_infos[index]['annos']
    num_obj = np.sum(annos["index"] >= 0)
    rbbox_cam = kitti.anno_to_rbboxes(annos)[:num_obj]
    rbbox_lidar = box_np_ops.box_camera_to_lidar(rbbox_cam, rect, Trv2c)
    
    rbbox_corners = box_np_ops.center_to_corner_box3d(rbbox_lidar[:, :3], rbbox_lidar[:, 3:6], rbbox_lidar[:, 6], origin=[0.5, 0.5, 0], axis=2)

    boxes_pred = create_boxes_from_val(loc, dim, yaw, names, len(loc))

    # Fig 1

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set_facecolor('black')
    ax.grid(False)

    ax.scatter(points_v[:, 0], points_v[:, 1], s=0.1, c="white", cmap='grey')

    for box in boxes:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[0,:],points[1,:],'r')
        
    for box in boxes_pred:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[0,:],points[1,:],'b')

    for idx in range(len(rbbox_corners)):
      ax.plot(rbbox_corners[idx,:,0],rbbox_corners[idx,:,1],'g')

    for index_obj in range(len(boxes)):
      obj = np.fromfile(str(lyftdata.data_path) + '/../gt_database/'+str(level5_infos[index]['image_idx'])+'_'+names[index_obj]+'_'+str(index_obj)+'.bin', dtype=np.float32, count=-1).reshape([-1, 4])
      ax.scatter(obj[:, 0]+loc[index_obj, 0], obj[:, 1]+loc[index_obj, 1], s=0.1, c="red", cmap='grey')

    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)

    # Fig 2

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_facecolor('black')
    ax.grid(False)

    filter_index = [value for value in np.where(points_v[:, 0]> -10)[0] if value in np.where(points_v[:, 0]< 100)[0]]  
    ax.scatter(points_v[filter_index, 1], points_v[filter_index, 2], s=0.1, c="white", cmap='grey')

    for box in boxes:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[1,:],points[2,:],'r')

    for box in boxes_pred:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[1,:],points[2,:],'b')
        
    for idx in range(len(rbbox_corners)):
      ax.plot(rbbox_corners[idx,:,1],rbbox_corners[idx,:,2],'g')

    for index_obj in range(len(boxes)):
      obj = np.fromfile(str(lyftdata.data_path) + '/../gt_database/'+str(level5_infos[index]['image_idx'])+'_'+names[index_obj]+'_'+str(index_obj)+'.bin', dtype=np.float32, count=-1).reshape([-1, 4])
      ax.scatter(obj[:, 1]+loc[index_obj, 1], obj[:, 2]+loc[index_obj, 2], s=0.1, c="red", cmap='grey')

    ax.set_xlim(-100,100)
    ax.set_ylim(-8,6)

def show_slice(data,res,index):


    # loc = case['location']
    # dim = case['dimensions']
    # angle = case['rotation_y']


    # box = np.concatenate((loc,dim,angle.reshape(-1,1)),axis=1)
    # box = torch.as_tensor(box, dtype=torch.float16, device = torch.device("cuda:0"))
    # rect = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ],
    #                  [ 0.        ,  1.        ,  0.        ,  0.        ],
    #                  [ 0.        ,  0.        ,  1.        ,  0.        ],
    #                  [ 0.        ,  0.        ,  0.        ,  1.        ]])     
    # rect = torch.as_tensor(rect, dtype=torch.float16, device = torch.device("cuda:0"))
    # Trv2c = np.array([[ 1.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
    #                   [ 0.000000e+00,  1.000000e+00,  0.000000e+00,  0.000000e+00],
    #                   [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  0.000000e+00],
    #                   [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
    # Trv2c = torch.as_tensor(Trv2c, dtype=torch.float16, device = torch.device("cuda:0"))
    # #box = box_torch_ops.box_lidar_to_camera(box, rect, Trv2c)
    # box = box.cpu().numpy()
    # loc = box[:, :3]
    # dim = box[:, 3:6]
    # angle = box[:, 6]


    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.set_facecolor('black')
    ax.grid(False)

    points_v = np.fromfile('/level5_data/'+data[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 5])

    case = res[index]

    score = case['score']
    number = len(np.where(score>0.4)[0])
      
    ax.scatter(points_v[:, 0], points_v[:, 1], s=0.01, c="white", cmap='grey')

    #points_v = np.fromfile('/level5_data/train_lidar/host-a004_lidar1_1232815254300468606.bin', dtype=np.float32, count=-1).reshape([-1, 5])
    #loc = res[index]['location']
    #ax.scatter(points_v[:, 0], points_v[:, 3], s=0.01, c="green", cmap='grey')



    ax.scatter(loc[:number,2], -loc[:number,0], s=10, c="red", cmap='grey')

    ax.scatter(loc[:number,2]-dim[:number,0]/2.0, -loc[:number,0]-dim[:number,2]/2.0, s=10, c="green", cmap='grey')
    ax.scatter(loc[:number,2]+dim[:number,0]/2.0, -loc[:number,0]-dim[:number,2]/2.0, s=10, c="green", cmap='grey')
    ax.scatter(loc[:number,2]-dim[:number,0]/2.0, -loc[:number,0]+dim[:number,2]/2.0, s=10, c="green", cmap='grey')
    ax.scatter(loc[:number,2]+dim[:number,0]/2.0, -loc[:number,0]+dim[:number,2]/2.0, s=10, c="green", cmap='grey')

    ax.set_xlim(-20,100)
    ax.set_ylim(-50,25)


def average_size(level5_data,lyftdata):

    w = {"car":0,"truck":0,"bus":0,"pedestrian":0,"bicycle":0,"motorcycle":0,"other_vehicle":0,"emergency_vehicle":0,"animal":0}
    l = {"car":0,"truck":0,"bus":0,"pedestrian":0,"bicycle":0,"motorcycle":0,"other_vehicle":0,"emergency_vehicle":0,"animal":0}
    h = {"car":0,"truck":0,"bus":0,"pedestrian":0,"bicycle":0,"motorcycle":0,"other_vehicle":0,"emergency_vehicle":0,"animal":0}
    m = {"car":0,"truck":0,"bus":0,"pedestrian":0,"bicycle":0,"motorcycle":0,"other_vehicle":0,"emergency_vehicle":0,"animal":0}

    for index in prog_bar(range(len(level5_data))):
        token = level5_data.iloc[index]['Id']
        sample = lyftdata.get('sample', token)
        _, boxes, _ = lyftdata.get_sample_data(sample['data']['LIDAR_TOP'], flat_vehicle_coordinates=False)
        for box in boxes:
          w[box.name] += box.wlh[0]
          l[box.name] += box.wlh[1]
          h[box.name] += box.wlh[2]
          m[box.name] += 1        

    for key in m.keys():
      if m[key]:
        print(key,"[%.2f,%.2f,%.2f]" % (w[key]/m[key],l[key]/m[key],h[key]/m[key]))


def cloud_range(level5_data,lyftdata):

    cloud = []

    for index in prog_bar(range(len(level5_data))):
        token = level5_data.iloc[index]['Id']
        sample = lyftdata.get('sample', token)
        _, boxes, _ = lyftdata.get_sample_data(sample['data']['LIDAR_TOP'], flat_vehicle_coordinates=False)
        for box in boxes:
          cloud.append(list(box.center))
          
    cloud = np.array(cloud)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.scatter(cloud[:, 0], cloud[:, 1])
    size = 110
    ax.plot([-size,size,size,-size,-size],[size,size,-size,-size,size])
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.scatter(cloud[:, 0], cloud[:, 2])
    ax.plot([-100,100],[6,6])
    ax.plot([-100,100],[-8,-8])
    plt.show()


def rotate(phi):

    # x =  np.cos(phi)*x + np.sin(phi)*y
    # y = -np.sin(phi)*x + np.cos(phi)*y

    # angle += phi

    return

def filter_points(velodyne_path):

    points_v = np.fromfile(velodyne_path, dtype=np.float32, count=-1).reshape([-1, 5])
    
    x_img, d_lidar = _project_points(points_v[:,:3])
    tri = Delaunay(x_img)
    angles, phi, area = _filter_points(points_v[:,:3],tri.simplices)
    
    max_angle = np.max(angles, axis=1)
    min_angle = np.min(angles, axis=1)
    skew = np.maximum((max_angle-np.pi/3.0)/(np.pi-np.pi/3.0) , (np.pi/3.0-min_angle)/(np.pi/3.0))
    
    filtered_simplices = tri.simplices[~((skew > 0.92) | (abs(phi) > 70.0*np.pi/180.0) | ((area < 0.002) & (abs(phi) > 65.0*np.pi/180.0)))]

    fil = np.zeros((len(points_v),1),dtype=np.int32)
    for index in range(3): fil[filtered_simplices[:,index]] = 1
    
    fil[(d_lidar<30.0) & (points_v[:,2]>-1.0)] = 1
    fil[(points_v[:,2]>3)] = 0

    points_v_filtered = np.array(list(compress(points_v, fil)))

    save_filename = velodyne_path.split('/')
    save_filename[-2] += "_reduced_bis"
    save_filename = '/'.join(save_filename)

    with open(save_filename, 'w') as f:
        points_v_filtered.tofile(f)

@numba.jit(nopython=True)
def _project_points(points_v):

    x_lidar = points_v[:, 0]
    y_lidar = points_v[:, 1]
    z_lidar = points_v[:, 2]
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    x_img = np.zeros((len(points_v),2))
    x_img[:,0] = np.arctan2(y_lidar, -x_lidar)
    x_img[:,1] = np.arctan2(z_lidar, d_lidar)

    return x_img, d_lidar

@numba.jit(nopython=True)
def _filter_points(points_v,simplices):

    p1 = points_v[simplices[:,0]]
    p2 = points_v[simplices[:,1]]
    p3 = points_v[simplices[:,2]]

    edges = np.zeros((len(simplices),3))
    edges[:,0] = np.sqrt((p1[:,0]-p2[:,0])**2+(p1[:,1]-p2[:,1])**2+(p1[:,2]-p2[:,2])**2)
    edges[:,1] = np.sqrt((p1[:,0]-p3[:,0])**2+(p1[:,1]-p3[:,1])**2+(p1[:,2]-p3[:,2])**2)
    edges[:,2] = np.sqrt((p2[:,0]-p3[:,0])**2+(p2[:,1]-p3[:,1])**2+(p2[:,2]-p3[:,2])**2)

    angles = np.zeros((len(simplices),3))
    angles[:,0] = np.arccos((edges[:,1]**2+edges[:,2]**2-edges[:,0]**2)/(2*edges[:,1]*edges[:,2]))
    angles[:,1] = np.arccos((edges[:,0]**2+edges[:,2]**2-edges[:,1]**2)/(2*edges[:,0]*edges[:,2]))
    angles[:,2] = np.arccos((edges[:,0]**2+edges[:,1]**2-edges[:,2]**2)/(2*edges[:,0]*edges[:,1]))

    U = p2-p1
    V = p3-p1
    normals = np.zeros((len(simplices),3))
    normals[:,0] = U[:,1]*V[:,2]-U[:,2]*V[:,1]
    normals[:,1] = U[:,2]*V[:,0]-U[:,0]*V[:,2]
    normals[:,2] = U[:,0]*V[:,1]-U[:,1]*V[:,0]

    phi = np.arcsin(normals[:,2]/np.sqrt(normals[:,0]**2+normals[:,1]**2+normals[:,2]**2))

    area = 0.5*edges[:,0]*edges[:,1]*np.sin(angles[:,2])
    
    return angles, phi, area
    
    








