
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm

try:
    import plotly.graph_objects as go
except:
    pass

import matplotlib.image as mpimg
import tqdm
from tqdm import tqdm as prog_bar

from skimage import io
from PIL import Image

import pandas as pd
import numpy as np

import os, sys, glob
import pickle

from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

import random, copy

from second.core import box_np_ops
from second.core.point_cloud.point_cloud_ops import bound_points_jit
from second.data import kitti_common as kitti

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from itertools import compress

import numba


sys.path.append('/content/AB3DMOT/')
try:
    from main import AB3DMOT, associate_detections_to_trackers, convert_3dbox_to_8corner, KalmanBoxTracker
except:
    pass

det_id2str = {1:'car', 2:'other_vehicle', 3:'pedestrian', 4:'bicycle', 5:'bus', 6:'truck', 7:'motorcycle', 8:'emergency_vehicle', 9:'animal'}
det_str2id = {'car':1, 'other_vehicle':2, 'pedestrian':3, 'bicycle':4, 'bus':5, 'truck':6, 'motorcycle':7, 'emergency_vehicle':8, 'animal':9}

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


def create_level5_infos(level5_data_train, level5_data_test, lyftdata_train, lyftdata_test, bounds=[-500.0, -500.0, -5, 500.0, 500.0, 3]):

    level5_infos_train = []
    level5_infos_val = []
    level5_infos_test = []

    random.seed(42)
    random_index = np.arange(len(level5_data_train))
    random.shuffle(random_index)
    sep = int(0.8*len(level5_data_train))

    train_index = random_index[:sep]
    val_index = random_index[sep:]

    phis = [0.0] #[0.0,90.0,180.0,270.0]

    for index in prog_bar(train_index):
        for phi in phis:
            sample_data = create_sample_data(level5_data_train,lyftdata_train,index,bounds,phi=phi,annotations=True)
            level5_infos_train.append(sample_data)

    for index in prog_bar(val_index):
        for phi in phis:
            sample_data = create_sample_data(level5_data_train,lyftdata_train,index,bounds,phi=phi,annotations=True)
            level5_infos_val.append(sample_data)

    for index in prog_bar(range(len(level5_data_test))):
        for phi in phis:
            sample_data = create_sample_data(level5_data_test,lyftdata_test,index,bounds,phi=phi,annotations=False)
            level5_infos_test.append(sample_data)

    return level5_infos_train, level5_infos_val, level5_infos_test

def create_sample_data(level5_data,lyftdata,index,bounds,phi=0.0,annotations=False):

    token = level5_data.iloc[index]['Id']

    sample = lyftdata.get('sample', token)

    sample_data = {}
    sample_data['image_idx'] = index
    sample_data['level5_token'] = token
    sample_data['orientation'] = phi
    sample_data['pointcloud_num_features'] = 6

    sample_image = lyftdata.get('sample_data',sample['data']['CAM_FRONT'])
    sample_data['img_path'] = sample_image['filename']

    sample_lidar = lyftdata.get('sample_data',sample['data']['LIDAR_TOP'])

    # Create New Name
    velodyne_path = os.path.join(lyftdata.data_path,sample_lidar['filename'])
    # velodyne_path = velodyne_path.split('/')
    # #velodyne_path[-2] += "_reduced"
    # velodyne_path.insert(-1,velodyne_path[-1].split('_')[0])
    # velodyne_path.insert(-1,str(phi))
    # velodyne_path = '/'.join(velodyne_path)


    sample_data['velodyne_path'] = velodyne_path

    # save_filename = velodyne_path.split('/')
    # save_filename[-2] += "_reduced"
    # save_filename = '/'.join(save_filename)

    # filter_points(velodyne_path,save_filename,bounds,phi)

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

            x =  np.cos(phi*np.pi/180.0)*box.center[0] + np.sin(phi*np.pi/180.0)*box.center[1]
            y = -np.sin(phi*np.pi/180.0)*box.center[0] + np.cos(phi*np.pi/180.0)*box.center[1]

            if x >= bounds[0] and y >= bounds[1] and x <= bounds[3] and y <= bounds[4]:
                annos['name'].append(name_map[box.name])
                annos['dimensions'].append([box.wlh[1],box.wlh[2],box.wlh[0]])   #       # l, h, w --> w, l, h   # Check that kitty_info_val.pkl has l, h, w
                annos['location'].append([x,y,box.center[2]-box.wlh[2]/2.0])
                annos['rotation_y'].append(-box.orientation.yaw_pitch_roll[0]+phi*np.pi/180.0+np.pi/2.0)
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

def world_to_lidar(box,lyftdata,lidar_top_token): # sample['data']['LIDAR_TOP']
    
    sd_record = lyftdata.get("sample_data", lidar_top_token)
    cs_record = lyftdata.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = lyftdata.get("ego_pose", sd_record["ego_pose_token"])

    box.translate(-np.array(pose_record["translation"]))
    box.rotate(Quaternion(pose_record["rotation"]).inverse)

    box.translate(-np.array(cs_record["translation"]))
    box.rotate(Quaternion(cs_record["rotation"]).inverse)

def pred_to_submission(submission,level5_infos,lyftdata,result,phi=0.0,min_score=0.5):

    # submission_array = []
    # for key, value in submission.items():
    #     submission_array.append([key,value])
    # df_submission = pd.DataFrame(submission_array,columns=['Id','PredictionString'])
    # df_submission.to_csv('submission.csv',index=False)

    for index in prog_bar(range(len(level5_infos))):
        
        sample = lyftdata.get('sample', level5_infos[index]['level5_token'])
        case = result[index]
        
        boxes = create_boxes(case,phi=phi,min_score=min_score)
        score = case['score']

        world_boxes = []
        for box in boxes:
            lidar_to_world(box,lyftdata,sample['data']['LIDAR_TOP'])
            world_boxes.append(box)

        pred_str = ''
        for box_index,box in enumerate(world_boxes):
            # if abs(box.orientation.yaw_pitch_roll[0]-box.orientation.radians) > 0.0000001: raise ValueError
            pred_str += '%f %f %f %f %f %f %f %f %s ' % (score[box_index],box.center[0],box.center[1],box.center[2],box.wlh[0],box.wlh[1],box.wlh[2],box.orientation.yaw_pitch_roll[0],box.name)

        key = level5_infos[index]['level5_token']
        
        if key in submission:
            submission[key] += pred_str
        else:
            submission[key] = pred_str

def show_sample(sample_token,lyftdata,train=False,result=None,level5_infos=None,submission_dfs=None,phi=0.0,min_score=0.5,sub_folders=False):

    sample_record = lyftdata.get('sample', sample_token)

    sample_lidar = lyftdata.get('sample_data',sample_record['data']['LIDAR_TOP'])
    velodyne_path = os.path.join(lyftdata.data_path,sample_lidar['filename'])

    if sub_folders:
        velodyne_path = velodyne_path.split('/')
        velodyne_path.insert(-1,velodyne_path[-1].split('_')[0])
        velodyne_path = '/'.join(velodyne_path)

    points_v = np.fromfile(velodyne_path, dtype=np.float32).reshape((-1, 6))

    # Fig 1

    fig1, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    ax1.grid(False)
    ax1.set_xlim(-100,100)
    ax1.set_ylim(-100,100)
    ax1.scatter(points_v[:, 0], points_v[:, 1], s=5.0, c=points_v[:,3:6])

    # Fig 2

    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 2))
    mask = np.ones(points_v.shape[0], dtype=bool)
    mask = np.logical_and(mask, points_v[:, 0] > -10)
    mask = np.logical_and(mask, points_v[:, 0] < 100)
    ax2.scatter(points_v[mask, 1], points_v[mask, 2], s=5.0, c=points_v[mask,3:6])
    ax2.grid(False)
    ax2.set_xlim(-15,15)
    ax2.set_ylim(-3,1)

    # Boxes

    if train:
        _, boxes, _ = lyftdata.get_sample_data(sample_record['data']['LIDAR_TOP'], flat_vehicle_coordinates=False)
        for box in boxes:

            # box.center[0] = - box.center[0]
            # box.center[1] = - box.center[1]
            # box.orientation = Quaternion(axis=[0,0,1], angle=box.orientation.yaw_pitch_roll[0]-np.pi)

            points = view_points(box.corners(), view=np.eye(3), normalize=False)
            ax1.plot(points[0,:],points[1,:],'r')
            ax2.plot(points[1,:],points[2,:],'r')

    if result is not None:
        for index in range(len(level5_infos)):
            if level5_infos[index]['level5_token'] == sample_token:
                break
        boxes_pred = create_boxes(result[index],phi=phi,min_score=min_score)
        for box in boxes_pred:
            points = view_points(box.corners(), view=np.eye(3), normalize=False)
            ax1.plot(points[0,:],points[1,:],'b')
            ax2.plot(points[1,:],points[2,:],'b')

    if submission_dfs is not None:
        colors = 'rbgcky'
        for index,submission_df in enumerate(submission_dfs):
          boxes_subm = submission_to_boxes(submission_df,sample_token,lyftdata,min_score=min_score)
          for box in boxes_subm:
              points = view_points(box.corners(), view=np.eye(3), normalize=False)
              ax1.plot(points[0,:],points[1,:],c=colors[index%len(colors)],linestyle='dashed')
              ax2.plot(points[1,:],points[2,:],c=colors[index%len(colors)],linestyle='dashed')


    plt.show()

def show_scene(level5_infos,lyftdata,result,index,phi=0.0,min_score=0.5):

    # v_filename = level5_infos[index]['velodyne_path'].split('/')
    # v_filename[-2] += "_reduced"
    # v_filename = '/'.join(v_filename)

    points_v = np.fromfile(level5_infos[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 6])

    sample = lyftdata.get('sample', level5_infos[index]['level5_token'])
    _, boxes, _ = lyftdata.get_sample_data(sample['data']['LIDAR_TOP'], flat_vehicle_coordinates=False)

    boxes_pred = create_boxes(result[index],phi=phi,min_score=min_score)

    # Fig 1

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    ax.scatter(points_v[:, 0], points_v[:, 1], s=5.0, c=points_v[:,3:6])

    for box in boxes:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[0,:],points[1,:],'r')
        
    for box in boxes_pred:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[0,:],points[1,:],'b')

    ax.grid(False)
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)

    # Fig 2

    fig, ax = plt.subplots(1, 1, figsize=(15, 2))

    mask = np.ones(points_v.shape[0], dtype=bool)
    mask = np.logical_and(mask, points_v[:, 0] > -10)
    mask = np.logical_and(mask, points_v[:, 0] < 100)
    ax.scatter(points_v[mask, 1], points_v[mask, 2], s=5.0, c=points_v[mask,3:6])

    for box in boxes:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[1,:],points[2,:],'r')

    for box in boxes_pred:
      points = view_points(box.corners(), view=np.eye(3), normalize=False)
      ax.plot(points[1,:],points[2,:],'b')
        
    ax.grid(False)
    ax.set_xlim(-15,15)
    ax.set_ylim(-3,1)


def show_scene_3d(level5_infos,lyftdata,result,index):

    points_v = np.fromfile(level5_infos[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 6])

    df_tmp = pd.DataFrame(points_v[:, :6], columns=['x', 'y', 'z', 'r', 'g', 'b'])
    # df_tmp['norm'] = np.sqrt(np.power(df_tmp[['x', 'y', 'z']].values, 2).sum(axis=1))
    scatter = go.Scatter3d(
        x=df_tmp['x'],
        y=df_tmp['y'],
        z=df_tmp['z'],
        mode='markers',
        marker=dict(
            size=1,
            color=df_tmp[['r', 'g', 'b']].values, # df_tmp['norm'],
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

def create_boxes(case,phi=0.0,min_score=0.5):

    score = case['score']
    number = len(np.where(score>min_score)[0])

    loc = case['location']
    dim = case['dimensions']
    yaw = case['rotation_y']
    name = case['name']

    return create_boxes_from_val(loc,dim,yaw,name,number,phi=phi)

def create_boxes_from_val(loc,dim,yaw,name,number,phi=0.0):

    boxes = []

    for box_index in range(number):

        x =  np.cos(phi*np.pi/180.0)*loc[box_index][0] + np.sin(phi*np.pi/180.0)*loc[box_index][1]
        y = -np.sin(phi*np.pi/180.0)*loc[box_index][0] + np.cos(phi*np.pi/180.0)*loc[box_index][1]

        # angle = (yaw[box_index]+phi*np.pi/180.0-np.pi/2.0)/2.0
        # Quaternion(scalar=np.cos(angle), vector=[0, 0, np.sin(angle)]).inverse,
        
        box = Box(
                [x,y,loc[box_index][2]+dim[box_index][1]/2.0],

                [dim[box_index][2],dim[box_index][0],dim[box_index][1]], # dim == lhw; need wlh
          
                Quaternion(axis=[0,0,1], angle=-(yaw[box_index]+phi*np.pi/180.0)+np.pi/2.0),

                name=name_map_reverse[name[box_index]],
                token="token",
            )
        boxes.append(box)

    return boxes

def submission_to_boxes(submission_df,sample_token,lyftdata,min_score=0.5,switch_to_lidar=True):

    predictions = submission_df.loc[submission_df['Id'] == sample_token].iloc[0]['PredictionString']
    predictions = np.array(predictions.split()).reshape(-1,9)
    # predictions = np.array(predictions.split()).reshape(-1,8)

    boxes = []
    for val in predictions:

        # val = np.insert(val,0,1.0)

        score,x,y,z,w,l,h,yaw,name = float(val[0]),float(val[1]),float(val[2]),float(val[3]),float(val[4]),float(val[5]),float(val[6]),float(val[7]),val[8]
        if w > l:
            w,l = l,w
            yaw += np.pi/2.0

        if score >= min_score:
            box = Box([x,y,z], [w,l,h], Quaternion(axis=[0,0,1], angle=yaw),
                    name=name_map_reverse[name], token="token")

            if switch_to_lidar:
                sample = lyftdata.get('sample', sample_token)
                world_to_lidar(box,lyftdata,sample['data']['LIDAR_TOP'])

            boxes.append(box)

    return boxes


def show_grounds(level5_infos,lyftdata,index):

    points_v = np.fromfile(level5_infos[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 6])

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
      obj = np.fromfile(str(lyftdata.data_path) + '/../gt_database/'+str(level5_infos[index]['image_idx'])+'_'+names[index_obj]+'_'+str(index_obj)+'.bin', dtype=np.float32, count=-1).reshape([-1, 6])
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
      obj = np.fromfile(str(lyftdata.data_path) + '/../gt_database/'+str(level5_infos[index]['image_idx'])+'_'+names[index_obj]+'_'+str(index_obj)+'.bin', dtype=np.float32, count=-1).reshape([-1, 6])
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

    points_v = np.fromfile('/level5_data/'+data[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 6])

    case = res[index]

    score = case['score']
    number = len(np.where(score>0.4)[0])
      
    ax.scatter(points_v[:, 0], points_v[:, 1], s=0.01, c="white", cmap='grey')

    #points_v = np.fromfile('/level5_data/train_lidar/host-a004_lidar1_1232815254300468606.bin', dtype=np.float32, count=-1).reshape([-1, 6])
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


# [0, -39.68, -5, 108.8, 39.68, 3]

def filter_points(points_v,save_filename='',bounds=[-500.0, -500.0, -5.0, 500.0, 500.0, 3.0],phi=0.0):

    #points_v = np.fromfile(velodyne_path, dtype=np.float32, count=-1).reshape([-1, 6])

    x_img, d_lidar = _project_points(points_v[:,:3])
    tri = Delaunay(x_img)
    angles, normal_phi, area = _filter_points(points_v[:,:3],tri.simplices)
    
    max_angle = np.max(angles, axis=1)
    min_angle = np.min(angles, axis=1)
    skew = np.maximum((max_angle-np.pi/3.0)/(np.pi-np.pi/3.0) , (np.pi/3.0-min_angle)/(np.pi/3.0))
    
    filtered_simplices = tri.simplices[~((skew > 0.92) | (abs(normal_phi) > 70.0*np.pi/180.0) | ((area < 0.002) & (abs(normal_phi) > 65.0*np.pi/180.0)))]


    mask = np.zeros((len(points_v),1),dtype=np.int32)
    # mask = np.ones(depths.shape[0], dtype=bool)

    for index in range(3): mask[filtered_simplices[:,index]] = 1
    
    mask[(d_lidar<30.0) & (points_v[:,2]>-1.0)] = 1


    x_saved = points_v[:,0]
    y_saved = points_v[:,1]

    points_v[:,0] =  np.cos(phi*np.pi/180.0)*x_saved + np.sin(phi*np.pi/180.0)*y_saved
    points_v[:,1] = -np.sin(phi*np.pi/180.0)*x_saved + np.cos(phi*np.pi/180.0)*y_saved

    mask[(points_v[:,0]<bounds[0])] = 0
    mask[(points_v[:,0]>bounds[3])] = 0

    mask[(points_v[:,1]<bounds[1])] = 0
    mask[(points_v[:,1]>bounds[4])] = 0

    mask[(points_v[:,2]<bounds[2])] = 0
    mask[(points_v[:,2]>bounds[5])] = 0



    # mask = np.logical_and(mask, depths > 0)
    # mask = np.logical_and(mask, points[0, :] > 1)
    # mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    # mask = np.logical_and(mask, points[1, :] > 1)
    # mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)



    points_v_filtered = np.array(list(compress(points_v, mask)))
    # points_v_filtered = points_v[mask]

    return points_v_filtered

    # with open(save_filename, 'w') as f:
    #     points_v_filtered.tofile(f)

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

    normal_phi = np.arcsin(normals[:,2]/np.sqrt(normals[:,0]**2+normals[:,1]**2+normals[:,2]**2))

    area = 0.5*edges[:,0]*edges[:,1]*np.sin(angles[:,2])
    
    return angles, normal_phi, area
    



def reder_rgb_all(level5_data,lyftdata,index,filter_v=False):

    sample_token = level5_data.iloc[index]['Id']

    pointsensor_channel = "LIDAR_TOP"
    sample_record = lyftdata.get("sample", sample_token)
    pointsensor_token = sample_record["data"][pointsensor_channel]
    pointsensor = lyftdata.get("sample_data", pointsensor_token)
    pcl_path = lyftdata.data_path / pointsensor["filename"]

    point_v = np.fromfile(str(pcl_path), dtype=np.float32).reshape((-1, 5))[:,:LidarPointCloud.nbr_dims()]
    
    if filter_v:
        point_v = filter_points(point_v)

    pc = LidarPointCloud(point_v.copy().T)
    #pc = LidarPointCloud.from_file(pcl_path)

    cs_record = lyftdata.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))
    poserecord = lyftdata.get("ego_pose", pointsensor["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))
    
    point_v = np.lib.pad(point_v[:,:3],((0,0),(0,3)),'constant',constant_values=(0))  
    count = np.zeros((len(point_v),1))
    for camera_chanel in ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT"]:
        mask, cloud_color, eccentricity = render_rgb_on_pointcloud(lyftdata, sample_token, copy.deepcopy(pc), dot_size = 15, camera_channel = camera_chanel)
        point_v[mask,3:6] += cloud_color
        count[mask] += 1.0
    point_v[:,3:6] /= np.clip(count,1.0,max(count))

    # Show

    x_lidar = point_v[:, 0]
    y_lidar = point_v[:, 1]
    z_lidar = point_v[:, 2]
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    x_img = np.zeros((len(point_v),2))
    x_img[:,0] = np.arctan2(y_lidar, -x_lidar)
    x_img[:,1] = np.arctan2(z_lidar, d_lidar)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(x_img[:,0],x_img[:,1], s=1.0, c=point_v[:,3:6])
    ax.axis('scaled')
    fig.show()
    

def render_rgb_on_pointcloud(lyftdata, sample_token, pc, dot_size = 2, camera_channel= "CAM_FRONT"):

    sample_record = lyftdata.get("sample", sample_token)
    camera_token = sample_record["data"][camera_channel]

    points, mask, im = map_pointcloud_to_image(lyftdata, pc, camera_token)
    points = points[:, mask]
    colors = np.array(im)
    
    cloud_color, eccentricity = _render_rgb_on_pointcloud(np.array(np.round(points[0:2, :]),dtype=np.int32),colors)

    return mask, cloud_color, eccentricity 

@numba.jit(nopython=True)
def _render_rgb_on_pointcloud(indices,colors):

    cloud_color = np.zeros((len(indices[0, :]),3))    
    for index in range(len(indices[0, :])):
        cloud_color[index,:] = np.sum(np.sum(colors[indices[1,index]-2:indices[1,index]+2+1,indices[0,index]-1:indices[0,index]+1+1,:],axis=0),axis=0)/(5*3)/255
    eccentricity = np.sqrt((indices[1,:]-colors.shape[0]/2.0)**2.0 + (indices[0,:]-colors.shape[1]/2.0)**2.0) 

    return cloud_color, eccentricity

def map_pointcloud_to_image(lyftdata, pc, camera_token):

    cam = lyftdata.get("sample_data", camera_token)
    im = Image.open(str(lyftdata.data_path / cam["filename"]))

    poserecord = lyftdata.get("ego_pose", cam["ego_pose_token"])
    pc.translate(-np.array(poserecord["translation"]))
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)
    cs_record = lyftdata.get("calibrated_sensor", cam["calibrated_sensor_token"])
    pc.translate(-np.array(cs_record["translation"]))
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    depths = pc.points[2, :]
    points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

    return points, mask, im



def show_scene_all_frames(scene_token,submission_df,lyftdata): # lyftdata = lyftdata_test / lyftdata_train

    # sample_token = level5_data_train.iloc[100]['Id']
    # sample_record = lyftdata.get("sample", sample_token)
    # scene_token = sample_record['scene_token']

    cmap = matplotlib.cm.jet
    norm = Normalize(vmin=0, vmax=100)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.grid(False)

    for scene_token in [scene_token]:

        scene_record = lyftdata.get("scene", scene_token)
        sample_token = scene_record['first_sample_token']

        index = 0
        while sample_token:

            try:

                boxes_subm = submission_to_boxes(submission_df,sample_token,lyftdata,min_score=0.0,switch_to_lidar=False)
                
                for box in boxes_subm:
                
                    points = view_points(box.corners(), view=np.eye(3), normalize=False)
                    ax.plot(points[0,:],points[1,:],c=cmap(norm(index)),alpha=0.2)
                    
                    sample_record = lyftdata.get('sample', sample_token)
                    sd_record = lyftdata.get("sample_data", sample_record['data']['LIDAR_TOP'])
                    cs_record = lyftdata.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
                    pose_record = lyftdata.get("ego_pose", sd_record["ego_pose_token"])
                    pos = np.array(cs_record["translation"])+np.array(pose_record["translation"])
                    ax.plot(pos[0],pos[1],'*',c=cmap(norm(index)))

                    # circle = plt.Circle((pos[0],pos[1]), 100.0, color=cmap(norm(index)), fill=False)
                    # ax.add_artist(circle)

                index += 1
                sample_token = lyftdata.get("sample", sample_token)['next']

            except:
                break
            
    ax.axis('scaled')
    plt.show()




# submission_kf = {}
# for index in range(len(lyftdata_test.scene)):
#     scene_token = lyftdata_test.scene[index]['token']
#     submission_kf.update(submission_kalman_filter_bis(scene_token,lyftdata_test,submission_df,std_threshold=0.3,sub_folders=True))

# submission_array = []
# for key, value in submission_kf.items():
#     submission_array.append([key,value])
# submission_df_filtered = pd.DataFrame(submission_array,columns=['Id','PredictionString'])
# submission_df_filtered.to_csv('floydhub_saved/level5_model_01/submission_filtered.csv',index=False)



def submission_kalman_filter_bis(scene_token,lyftdata,pred_df,std_threshold=0.5,max_age=5,sub_folders=False):

    # git clone https://github.com/xinshuoweng/AB3DMOT.git  
    # pip install -r /content/AB3DMOT/requirements.txt

    # submission_array = []
    # for key, value in submission.items():
    #     submission_array.append([key,value])
    # df_submission = pd.DataFrame(submission_array,columns=['Id','PredictionString'])
    # df_submission.to_csv('submission.csv',index=False)

    submission_kf = {}
    object_types = ['car','other_vehicle','bus','truck']


    for scene_token in [scene_token]:
    
        scene_record = lyftdata.get("scene", scene_token)
        print("Processing %s" % scene_token)
        print('WARNING: CHANGE IOU TO 0.001, use std 0.3')
        
        history = {}

        #
        # Forward Pass
        #
        print('Forward Pass')

        KalmanBoxTracker.count = 0
        mot_trackers = []
        for object_type in object_types:
            mot_tracker = AB3DMOT(max_age=max_age,min_hits=0) 
            mot_tracker.reorder = [0, 1, 2, 3, 4, 5, 6]
            mot_tracker.reorder_back = [0, 1, 2, 3, 4, 5, 6]
            mot_trackers.append(mot_tracker)

        sample_token = scene_record['first_sample_token']
        sample_index = -1
        while sample_token:
            sample_index += 1
            sample_record = lyftdata.get("sample", sample_token)
 
            for mot_tracker,object_type in zip(mot_trackers,object_types):
                dets_all = arrange_pred_score(pred_df,sample_token,object_type)
                mot_tracker.update(dets_all)
            
            for mot_tracker,object_type in zip(mot_trackers,object_types):
                for tracker in mot_tracker.trackers:
                    #print(tracker.kf.x.shape)
                    if not tracker.id in history:
                        history[tracker.id] = {'hits':[0]*(mot_tracker.frame_count-1),
                                               'states':[[np.nan]*len(tracker.get_state())]*(mot_tracker.frame_count-1),
                                               'infos':[[np.nan]*len(tracker.info)]*(mot_tracker.frame_count-1)}
                    history[tracker.id]['hits'].append(1 if tracker.time_since_update == 0 else 0)
                    history[tracker.id]['states'].append(list(tracker.get_state()))
                    history[tracker.id]['infos'].append(list(tracker.info))

            last_sample_token = sample_token

            sample_token = sample_record['next']

        #
        # Intermediate Processing
        #
        print('Intermediate Processing')

        for tracker_id in history.keys():
            missing_length = (mot_tracker.frame_count-len(history[tracker_id]['hits']))
            history[tracker_id]['hits'] += [0]*missing_length
            history[tracker_id]['states'] += [[np.nan]*len(tracker.get_state())]*missing_length
            history[tracker_id]['infos'] += [[np.nan]*len(tracker.info)]*missing_length

        for tracker_id in history.keys():
            history[tracker_id]['hits'] = np.array(history[tracker_id]['hits'])
            history[tracker_id]['states'] = np.array(history[tracker_id]['states'])
            history[tracker_id]['infos'] = np.array(history[tracker_id]['infos'])
        
        history = filter_single(history) # Might get them back in Backward Pass, and if not, probably False Positive

        # history = filter_yaw_jumps(history) # Definitely False Positives !!!!!!!!!!!!

        # TODO: Play with Value!!!!!!!!!!!
        
        history_stationary, history_moving = separate_history(history, std_threshold=std_threshold) # Based on Std        
        history_moving, history_misclassified = filter_moving(history_moving)

        override_with_average_dims(history_stationary)
        override_with_average_locs(history_stationary)
        override_with_average_dims(history_moving)

        # for tracker_id in history_moving.keys():
        #     print(history_moving[tracker_id]['hits'],history_moving[tracker_id]['std'])
        
        #
        # Backward Pass
        #
        print('Backward Pass')
        
    

        KalmanBoxTracker.count = 1000000000000000 # In order to guarantee that we don't mix with manually created tracker
        mot_trackers = []
        for object_type in object_types:
            mot_tracker = AB3DMOT(max_age=max_age,min_hits=0) 
            mot_tracker.reorder = [0, 1, 2, 3, 4, 5, 6]
            mot_tracker.reorder_back = [0, 1, 2, 3, 4, 5, 6]
            mot_trackers.append(mot_tracker)

        sample_token = last_sample_token # scene_record['last_sample_token']
        biggest_sample_index = sample_index

        trackers_already_created = []
        
        while sample_token:
            sample_record = lyftdata.get("sample", sample_token)

            trackers_just_created = []

            try:

                for history_local in [history_moving]: # ADD history_stationary ???????????????????????
                    # Create one tracker for each in history
                    for tracker_id in history_local.keys():
                        if tracker_id not in trackers_already_created:
                            state = history_local[tracker_id]['states'][sample_index]
                            if not np.isnan(state).any():
                                trackers_already_created.append(tracker_id)
                                trackers_just_created.append(tracker_id)
                                info = history_local[tracker_id]['infos'][sample_index]
                                tracker = KalmanBoxTracker(state, info)
                                tracker.id = tracker_id
                                object_type = det_id2str[info[1]]
                                index_tracker = object_types.index(object_type)
                                mot_trackers[index_tracker].trackers.append(tracker)
                    
                    if sample_index < biggest_sample_index:
                        for mot_tracker,object_type in zip(mot_trackers,object_types):
                            dets_all = arrange_history(history_local,sample_index,pred_df,sample_token,trackers_already_created,object_type) # If history has NaNs and tracker already created: dets_all = arrange_pred(pred_df,sample_token)
                            mot_tracker.update(dets_all) # Maybe ok to do it for all since only modify NaNs from history_local, and trackers are garanteed momentum from multiple samples

                    for mot_tracker,object_type in zip(mot_trackers,object_types):
                        for tracker in mot_tracker.trackers: # And then never Create a new tracker. Or in practice, ignore any trackers beyond given length
                            if tracker.id in history_local:
                                if np.isnan(history_local[tracker.id]['states'][sample_index]).any():
                                    history_local[tracker.id]['hits'][sample_index] = 1 if tracker.time_since_update == 0 else 0
                                    history_local[tracker.id]['states'][sample_index][:4] = tracker.get_state()[:4]
                                    history_local[tracker.id]['infos'][sample_index] = history_local[tracker.id]['infos'][sample_index+1]

            except:
                break

            sample_index -= 1
            sample_token = sample_record['prev']

        # override_with_average_dims(history_stationary)
        # override_with_average_locs(history_stationary)
        override_with_average_dims(history_moving)


        


        #
        # Remove Extrapolation - Keep only interpolation
        #
        print('Removing Extrapolation')

        # for tracker_id in value_stationary.keys():
        #     no_extrapolation(value_stationary[tracker_id]['hits'])
        # for tracker_id in history_stationary.keys():
        #     no_extrapolation(history_stationary[tracker_id]['hits'])
        # Maybe can exterpolate stationary !!!!!!!!!!!!!!!!!!!!!!! And check !!!!!!!!!!!!!


        # remove_nan_positions(history_moving)
        for tracker_id in history_moving.keys():
            no_extrapolation(history_moving[tracker_id]['hits'])




        # fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        # for tracker_id in history_stationary.keys():
        #     hits = history_stationary[tracker_id]['hits']
        #     mask = np.ones(hits.shape[0], dtype=bool)
        #     mask = np.logical_and(mask, hits != -1)
        #     plt.plot(history_stationary[tracker_id]['states'][mask,5],'r')
        # for tracker_id in history_moving.keys():
        #     hits = history_moving[tracker_id]['hits']
        #     mask = np.ones(hits.shape[0], dtype=bool)
        #     mask = np.logical_and(mask, hits != -1)          
        #     plt.plot(history_moving[tracker_id]['states'][mask,5],'b')
        # plt.ylim(0,5)
        # plt.show()


        #  
        # Use Candidates to make submission
        #
        print('Using Candidates to make submission')
        
        sample_token = scene_record['first_sample_token']
        sample_index = 0
        while sample_token:
            sample_record = lyftdata.get("sample", sample_token)
            
            sample_lidar = lyftdata.get('sample_data',sample_record['data']['LIDAR_TOP'])
            velodyne_path = os.path.join(lyftdata.data_path,sample_lidar['filename'])
            if sub_folders:
                velodyne_path = velodyne_path.split('/')
                velodyne_path.insert(-1,velodyne_path[-1].split('_')[0])
                velodyne_path = '/'.join(velodyne_path)
            points_v = np.fromfile(velodyne_path, dtype=np.float32).reshape((-1, 6))

            pred_str = ''

            for history_local in [history_stationary,history_moving]:

                validate_candidates(history_local,sample_token,sample_index,points_v,lyftdata)
  
                for tracker_id in history_local.keys():

                    hit = history_local[tracker_id]['hits'][sample_index]

                    if hit == 1:

                        state = history_local[tracker_id]['states'][sample_index]
                        info  = history_local[tracker_id]['infos'][sample_index]
                        x,y,z,yaw,l,w,h = state[0],state[1],state[2],state[3],state[4],state[5],state[6]
                        score = info[0]
                        name = det_id2str[info[1]]
                        pred_str += '%f %f %f %f %f %f %f %f %s ' % (score,x,y,z,w,l,h,yaw,name)

            submission_kf[sample_token] = pred_str

            sample_index += 1
            sample_token = sample_record['next']

    return submission_kf

def validate_candidates(history,sample_token,sample_index,points_v,lyftdata):

    tracker_ids_to_check = []
    lidar_boxes_to_check = []

    for tracker_id in history.keys():
        hit = history[tracker_id]['hits'][sample_index]
        if hit == 0:
            state = history[tracker_id]['states'][sample_index]
            lidar_box = state_to_lidar_box(state,sample_token,lyftdata)
            if tracker_range(lidar_box) > 100.0:
                continue
            tracker_ids_to_check.append(tracker_id)
            lidar_boxes_to_check.append(lidar_box)

    if len(lidar_boxes_to_check) > 0:
        n_points_checked = number_of_points(lidar_boxes_to_check,points_v)

        for tracker_id,n_points in zip(tracker_ids_to_check,n_points_checked):
            if n_points >= 2:
                history[tracker_id]['hits'][sample_index] = 1

def remove_nan_positions(history_moving):
    for tracker_id in history_moving.keys():
        states = history_moving[tracker_id]['states']
        history_moving[tracker_id]['hits'][np.isnan(states).any(axis=1)] = -1

def no_extrapolation(vec, padding=0):

    index = 0
    while vec[index] != 1:
        vec[index] = -1
        index += 1
    vec[max(0,index-padding):index] = 0

    index = len(vec)-1
    while vec[index] != 1:
        vec[index] = -1
        index -= 1
    vec[index+1:min(len(vec),index+padding+1)] = 0

def normalize_angle(angle):

    while (angle <= -np.pi):
        angle += 2.0*np.pi
    while (angle > np.pi):
        angle -= 2.0*np.pi
    return angle

def clip_angle(angle): # Since scoring solely depends on IOU, orientation doesn't matter

    if angle < 0:
        return angle + np.pi
    return angle

def state_to_lidar_box(state,sample_token,lyftdata):

    x,y,z,yaw,l,w,h = state[0],state[1],state[2],state[3],state[4],state[5],state[6]
    box = Box([x,y,z], [w,l,h], Quaternion(axis=[0,0,1], angle=yaw), name='None', token="token")
    sample_record = lyftdata.get('sample', sample_token)
    world_to_lidar(box,lyftdata,sample_record['data']['LIDAR_TOP'])

    return box

def tracker_range(lidar_box):

    x = lidar_box.center[0]
    y = lidar_box.center[1]

    return np.sqrt(x**2.0+y**2.0)
    
def number_of_points(lidar_boxes,points_v):

    rect = np.array([[ 1., 0., 0., 0.],
                     [ 0., 1., 0., 0.],
                     [ 0., 0., 1., 0.],
                     [ 0., 0., 0., 1.]])
      
    Trv2c = np.array([[ 1., 0., 0., 0.],
                      [ 0., 1., 0., 0.],
                      [ 0., 0., 1., 0.],
                      [ 0., 0., 0., 1.]])
    rbbox_cam = []
    for lidar_box in lidar_boxes:
        rbbox_cam.append([lidar_box.center[0], \
                          lidar_box.center[1], \
                          lidar_box.center[2]-lidar_box.wlh[2]/2.0, \
                          lidar_box.wlh[1], \
                          lidar_box.wlh[2], \
                          lidar_box.wlh[0], \
                          -lidar_box.orientation.yaw_pitch_roll[0]+np.pi/2.0])
    rbbox_cam = np.array(rbbox_cam)

    rbbox_lidar = box_np_ops.box_camera_to_lidar(rbbox_cam, rect, Trv2c)
    indices = box_np_ops.points_in_rbbox(points_v[:, :3], rbbox_lidar)
    num_points_in_gt = indices.sum(0)

    return num_points_in_gt

def filter_single(history):
    # Delete ones with only 1 hit     
    # Delete all [0 ... 0 1 0 ... 0] sum = 1

    # Most likely False Positives. If not, will get them back during Backward Pass

    history_filtered = {}
    for tracker_id in history.keys():
        if sum(history[tracker_id]['hits']) > 1:
            history_filtered[tracker_id] = history[tracker_id]

    return history_filtered

def separate_history(history,std_threshold=0.5):
    history_stationary = {}
    history_moving = {}

    for tracker_id in history.keys():

        x = history[tracker_id]['states'][:,0]
        x = x[~np.isnan(x)]
        y = history[tracker_id]['states'][:,1]
        y = y[~np.isnan(y)]

        if np.sqrt(np.std(x)**2.0+np.std(y)**2.0) < std_threshold: # Move less than std_threshold meter over all scene
            history_stationary[tracker_id] = history[tracker_id]
        else:
            history_moving[tracker_id] = history[tracker_id]
            history_moving[tracker_id]['std'] = np.sqrt(np.std(x)**2.0+np.std(y)**2.0)

    return history_stationary, history_moving

def filter_moving(history_moving):
    # Delete all [0 ... 0 1 0 0 0 1 0 ... 0] if Non stationary sum > 1 and no 2 neighbors if std > 0.1

    # history_moving_filtered = {}
    # history_misclassified = {}

    # for tracker_id in history_moving.keys():
    #     biggest_hit_streak = 0
    #     hit_streak = 0
    #     for hit in history_moving[tracker_id]['hits']:
    #         if hit == 1:
    #             hit_streak += 1
    #             if hit_streak > biggest_hit_streak:
    #                 biggest_hit_streak = hit_streak
    #         else:
    #             hit_streak = 0

    #     if biggest_hit_streak > 1:
    #         history_moving_filtered[tracker_id] = history_moving[tracker_id]
    #     else:
    #         history_misclassified[tracker_id] = history_moving[tracker_id]

    # return history_moving_filtered, history_misclassified

    return history_moving, {}

def keep_last_one_or_avergare_first_two(history_stationary, history_misclassified):
    value_stationary = {}

    for tracker_id in history_stationary.keys(): # Std is already filtered under certain threshold
        
        hits = history_stationary[tracker_id]['hits']
        states = history_stationary[tracker_id]['states']
        infos = history_stationary[tracker_id]['infos']

        if np.isnan(hits).any(): 
          raise ValueError
          # hits[np.isnan(hits)] = 0
        
        value_stationary[tracker_id] = {'hits':hits,
                                        'state':states[~np.isnan(states).any(axis=1)][-1],
                                        'info':infos[~np.isnan(infos).any(axis=1)][-1]}

    for tracker_id in history_misclassified.keys():
        first_hit = True
        for index,hit in enumerate(history_misclassified[tracker_id]['hits']):
            if hit == 1:
                if first_hit:
                    state = history_misclassified[tracker_id]['states'][index]
                    first_hit = False
                else:
                    state = (state+history_misclassified[tracker_id]['states'][index])/2.0
                    break
        hits = history_misclassified[tracker_id]['hits']
        if np.isnan(hits).any(): hits[np.isnan(hits)] = 0
        value_stationary[tracker_id] = {'hits':hits,
                                        'state':state,
                                        'info':history_misclassified[tracker_id]['infos'][-1]}

    return value_stationary

def override_with_last_dims(history_moving):
    
    for tracker_id in history_moving.keys():
        states = history_moving[tracker_id]['states']
        state = states[~np.isnan(states).any(axis=1)][-1]
        length = len(states)
        history_moving[tracker_id]['states'][:,4] = np.array([state[4]]*length)
        history_moving[tracker_id]['states'][:,5] = np.array([state[5]]*length)
        history_moving[tracker_id]['states'][:,6] = np.array([state[6]]*length)

def average_max_mode(vec,bins=5):

    hist, bin_edges = np.histogram(vec,bins=bins)
    index_max = np.argmax(hist)
    edges = bin_edges[index_max:index_max+2]
    mask = np.ones(vec.shape[0], dtype=bool)
    mask = np.logical_and(mask, vec >= edges[0])
    mask = np.logical_and(mask, vec <= edges[1])
    return np.mean(vec[mask])

def override_with_average_dims(history):
    
    for tracker_id in history.keys():

        states = history[tracker_id]['states']
        length = len(states)
        states = states[~np.isnan(states).any(axis=1)]
        history[tracker_id]['states'][:,4] = np.array([average_max_mode(states[:,4])]*length)
        history[tracker_id]['states'][:,5] = np.array([average_max_mode(states[:,5])]*length)
        history[tracker_id]['states'][:,6] = np.array([average_max_mode(states[:,6])]*length)

        infos = history[tracker_id]['infos']
        length = len(infos)
        infos = infos[~np.isnan(infos).any(axis=1)]
        history[tracker_id]['infos'][:,0] = np.array([average_max_mode(infos[:,0])]*length)
        history[tracker_id]['infos'][:,1] = np.array([infos[0,1]]*length)

        if len(set(infos[:,1])) != 1:
            raise ValueError

def override_with_average_locs(history):
    
    for tracker_id in history.keys():
        states = history[tracker_id]['states']
        length = len(states)
        states = states[~np.isnan(states).any(axis=1)]
        history[tracker_id]['states'][:,0] = np.array([average_max_mode(states[:,0])]*length)
        history[tracker_id]['states'][:,1] = np.array([average_max_mode(states[:,1])]*length)
        history[tracker_id]['states'][:,2] = np.array([average_max_mode(states[:,2])]*length)
        history[tracker_id]['states'][:,3] = np.array([average_max_mode(states[:,3])]*length)

def arrange_history(history,sample_index,pred_df,sample_token,trackers_already_created,object_type):

    dets = []
    additional_info = []

    has_nan = False

    for tracker_id in history.keys():
        state = history[tracker_id]['states'][sample_index]
        if not np.isnan(state).any():
            info = history[tracker_id]['infos'][sample_index]
            if det_id2str[info[1]] == object_type:
                dets.append(list(state))
                additional_info.append(list(info))
        elif tracker_id in trackers_already_created:
            has_nan = True  

    trks_all = {'dets': np.array(dets).reshape(-1,7), 'info': np.array(additional_info).reshape(-1,2)}

    if has_nan:

        dets_all = arrange_pred_score(pred_df,sample_token,object_type)

        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets_all['dets']]
        if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
        else: dets_8corner = []

        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks_all['dets']]
        if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
        else: trks_8corner = []

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner)

        # Concatenate trks_all and unmatched_dets
        if len(unmatched_dets) > 0:
            trks_all = {'dets': np.concatenate((trks_all['dets'],dets_all['dets'][unmatched_dets].reshape(-1,7)),axis=0),
                        'info': np.concatenate((trks_all['info'],dets_all['info'][unmatched_dets].reshape(-1,2)),axis=0)}

    return trks_all

def arrange_pred(pred_df,sample_token):

    predictions = pred_df.loc[pred_df['Id'] == sample_token].iloc[0]['PredictionString']
    predictions = np.array(predictions.split()).reshape(-1,8)

    dets = []
    additional_info = []
    for val in predictions:
        x,y,z,w,l,h,yaw,name = float(val[0]),float(val[1]),float(val[2]),float(val[3]),float(val[4]),float(val[5]),float(val[6]),val[7]
        # if w > l:
        #     w,l = l,w
        #     yaw += np.pi/2.0

        # 1. +/- 180
        # 2. All - add 180 (check with a submission)
        yaw = normalize_angle(yaw)
        yaw = clip_angle(yaw)

        dets.append([x,y,z,yaw,l,w,h])
        additional_info.append([det_str2id[name]])

    dets_all = {'dets': np.array(dets).reshape(-1,7), 'info': np.array(additional_info).reshape(-1,1)}

    return dets_all

def arrange_pred_score(pred_df,sample_token,object_type):

    predictions = pred_df.loc[pred_df['Id'] == sample_token].iloc[0]['PredictionString']
    predictions = np.array(predictions.split()).reshape(-1,9)

    dets = []
    additional_info = []
    for val in predictions:
        name = val[8]
        if name == object_type:
            score,x,y,z,w,l,h,yaw = float(val[0]),float(val[1]),float(val[2]),float(val[3]),float(val[4]),float(val[5]),float(val[6]),float(val[7])
        
            # if w > l:
            #     w,l = l,w
            #     yaw += np.pi/2.0

            # 1. +/- 180
            # 2. All - add 180 (check with a submission)

            yaw = normalize_angle(yaw)
            yaw = clip_angle(yaw)

            dets.append([x,y,z,yaw,l,w,h])
            additional_info.append([score,det_str2id[name]])

    dets_all = {'dets': np.array(dets).reshape(-1,7), 'info': np.array(additional_info).reshape(-1,2)}

    return dets_all

def dets_to_8corner(dets_all):

    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets_all['dets']]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []
    return dets_8corner

def trackers_to_str(trackers):

    pred_str = ''
    for d in trackers:
        x,y,z,yaw,l,w,h = d[0],d[1],d[2],d[3],d[4],d[5],d[6]
        object_id = int(d[7])
        name = det_id2str[d[8]]
        pred_str += '%f %f %f %f %f %f %f %s ' % (x,y,z,w,l,h,yaw,name)

    return pred_str





















def submission_kalman_filter(level5_data,lyftdata,pred_df,max_age=1,min_hits=0):

    # git clone https://github.com/xinshuoweng/AB3DMOT.git  
    # pip install -r /content/AB3DMOT/requirements.txt

    # submission_array = []
    # for key, value in submission.items():
    #     submission_array.append([key,value])
    # df_submission = pd.DataFrame(submission_array,columns=['Id','PredictionString'])
    # df_submission.to_csv('submission.csv',index=False)

    submission_kf = {}

    scenes = set([])
    for index in range(len(level5_data)):
        sample_token = level5_data.iloc[index]['Id']
        sample_record = lyftdata.get("sample", sample_token)
        scenes.add(sample_record['scene_token'])

    for index,scene_token in enumerate(scenes):
    
        mot_tracker = AB3DMOT(max_age=max_age,min_hits=min_hits) 
        mot_tracker.reorder = [0, 1, 2, 3, 4, 5, 6]
        mot_tracker.reorder_back = [0, 1, 2, 3, 4, 5, 6]

        scene_record = lyftdata.get("scene", scene_token)
        sample_token = scene_record['first_sample_token']
      
        print(index,"Processing %s" % scene_token)

        while sample_token:
            sample_record = lyftdata.get("sample", sample_token)
 
            dets_all = arrange_pred(pred_df,sample_token)

            # print(len(dets_all['dets']))
            # dets_all_2 = arrange_pred(pred_df_2,sample_token)
            # matched, unmatched_dets, unmatched_dets_2 = associate_detections_to_trackers(dets_to_8corner(dets_all), dets_to_8corner(dets_all_2))
            # dets_all['dets'], dets_all['info'] = dets_all['dets'][matched[:,0]], dets_all['info'][matched[:,0]]
            # print(len(dets_all['dets']))

            trackers = mot_tracker.update(dets_all)
            
            submission_kf[sample_token] = trackers_to_str(trackers)
            
            sample_token = sample_record['next']

    return submission_kf



def dets_to_8corner(dets_all):

    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets_all['dets']]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []
    return dets_8corner

def trackers_to_str_score(trackers):

    pred_str = ''
    for d in trackers:
        x,y,z,yaw,l,w,h = d[0],d[1],d[2],d[3],d[4],d[5],d[6]
        object_id = int(d[7])
        score = d[8]
        name = det_id2str[d[9]]
        pred_str += '%f %f %f %f %f %f %f %f %s ' % (score,x,y,z,w,l,h,yaw,name)

    return pred_str








# folder = 'level5_model/eval_results/step_330252/'

# pickle_filename = folder + 'result_000.pkl'
# with open(pickle_filename, 'rb') as pickle_file:
#     result_000 = pickle.load(pickle_file)

# pickle_filename = folder + 'result_045.pkl'
# with open(pickle_filename, 'rb') as pickle_file:
#     result_045 = pickle.load(pickle_file)

# pickle_filename = folder + 'result_090.pkl'
# with open(pickle_filename, 'rb') as pickle_file:
#     result_090 = pickle.load(pickle_file)

# pickle_filename = folder + 'result_135.pkl'
# with open(pickle_filename, 'rb') as pickle_file:
#     result_135 = pickle.load(pickle_file)

# pickle_filename = folder + 'result_180.pkl'
# with open(pickle_filename, 'rb') as pickle_file:
#     result_180 = pickle.load(pickle_file)

# pickle_filename = folder + 'result_225.pkl'
# with open(pickle_filename, 'rb') as pickle_file:
#     result_225 = pickle.load(pickle_file)

# pickle_filename = folder + 'result_270.pkl'
# with open(pickle_filename, 'rb') as pickle_file:
#     result_270 = pickle.load(pickle_file)

# pickle_filename = folder + 'result_315.pkl'
# with open(pickle_filename, 'rb') as pickle_file:
#     result_315 = pickle.load(pickle_file)


# result_0 = concatenate_results([process_result(result_000,           0.0),
#                                 process_result(result_180,        -np.pi)])
# result_1 = concatenate_results([process_result(result_090,    -np.pi/2.0),
#                                 process_result(result_270,-3.0*np.pi/2.0)])
# result_2 = concatenate_results([process_result(result_045,    -np.pi/4.0),
#                                 process_result(result_225,-5.0*np.pi/4.0)])
# result_3 = concatenate_results([process_result(result_135,-3.0*np.pi/4.0),
#                                 process_result(result_315,-7.0*np.pi/4.0)])

# print('1')
# new_res = concatenate_no_overlap(result_0,result_1)
# print('2')
# new_res = concatenate_no_overlap(new_res, result_2)
# print('3')
# new_res = concatenate_no_overlap(new_res, result_3)



def process_result(result,phi):
    result_copy = copy.deepcopy(result)
    for res in result_copy:
        new_x =  np.cos(phi)*res['location'][:,0] + np.sin(phi)*res['location'][:,1]
        new_y = -np.sin(phi)*res['location'][:,0] + np.cos(phi)*res['location'][:,1]
        res['location'][:,0] = new_x
        res['location'][:,1] = new_y
        res['rotation_y'] += phi
    return result_copy

def concatenate_results(results):
    length = len(results[0])
    new_result = copy.deepcopy(results[0])
    for index in range(length):
        for key in new_result[index].keys():
            arg = (results[0][index][key],)
            for result in results[1:]:
                arg += (result[index][key],)
            new_result[index][key] = np.concatenate(arg)
        sort_index = np.argsort(new_result[index]['score'])[::-1]
        for key in new_result[index].keys():
            new_result[index][key] = new_result[index][key][sort_index]

    return new_result



def concatenate_no_overlap(primary_result,secondary_result):

    length = len(primary_result)
    new_result = copy.deepcopy(primary_result)

    for index in prog_bar(range(length)):

        prim_8corner = arrange_case(primary_result[index])
        sec_8corner  = arrange_case(secondary_result[index])

        matched, unmatched_prim, unmatched_sec = associate_detections_to_trackers(prim_8corner, sec_8corner, iou_threshold=0.1)

        if len(unmatched_sec) > 0:
            for key in new_result[index].keys():
                new_result[index][key] = np.concatenate((primary_result[index][key],secondary_result[index][key][unmatched_sec]))
        sort_index = np.argsort(new_result[index]['score'])[::-1]
        for key in new_result[index].keys():
            new_result[index][key] = new_result[index][key][sort_index]

    return new_result

def arrange_case(case):
    dets = []
    for index in range(len(case['score'])):
        loc = case['location']
        dim = case['dimensions']
        yaw = case['rotation_y']

        x,y,z = loc[index][0],loc[index][1],loc[index][2]+dim[index][1]/2.0
        w,l,h = dim[index][2],dim[index][0],dim[index][1] # dim == lhw; need wlh
        yaw = -yaw[index]+np.pi/2.0
        dets.append([x,y,z,yaw,l,w,h])

    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in np.array(dets)]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []

    return dets_8corner




# for angle_deg in [0.0,45.0,90.0,135.0,180.0,225.0,270.0,315.0]:
#     evaluate(config_path,model_dir,predict_test=False,ckpt_path=parameters_path,angle_deg=angle_deg)






