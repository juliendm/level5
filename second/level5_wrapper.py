
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

import glob
import pickle

from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

import random




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


def create_level5_infos(level5_data_train, level5_data_test, lyftdata):

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
        sample_data = create_sample_data(level5_data_train,lyftdata,index,annotations=True)
        level5_infos_train.append(sample_data)

    for index in prog_bar(val_index):
        sample_data = create_sample_data(level5_data_train,lyftdata,index,annotations=True)
        level5_infos_val.append(sample_data)

    for index in prog_bar(range(len(level5_data_test))):
        sample_data = create_sample_data(level5_infos_test,lyftdata,index,annotations=False)
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
    sample_data['velodyne_path'] = sample_lidar['filename']

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
            annos['dimensions'].append(list(box.wlh))   #[box.wlh[1],box.wlh[2],box.wlh[0]]       # l, h, w --> w, l, h   # Check that kitty_info_val.pkl has l, h, w
            annos['location'].append(list(box.center))
            annos['rotation_y'].append(box.orientation.radians)               # Check if this is ok
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

      
def lidar_to_world(box,lidar_top_token): # sample['data']['LIDAR_TOP']
    
    sd_record = lyftdata.get("sample_data", lidar_top_token)
    cs_record = lyftdata.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = lyftdata.get("ego_pose", sd_record["ego_pose_token"])

    box.rotate(Quaternion(cs_record["rotation"]))
    box.translate(np.array(cs_record["translation"]))

    box.rotate(Quaternion(pose_record["rotation"]))
    box.translate(np.array(pose_record["translation"]))


def pred_to_submission(data,res):
  
    submission = {}

    for index in prog_bar(range(len(data))):
        
        sample = lyftdata.get('sample', data[index]['level5_token'])
        case = res[index]
        score = case['score']
        number = len(np.where(score>0.3)[0])
        loc = case['location']
        dim = case['dimensions']
        yaw = case['rotation_y']

        boxes = []
        for box_index in range(number):
            box = Box([loc[box_index][2],-loc[box_index][0],-loc[box_index][1]],
                    [dim[box_index][0],dim[box_index][2],dim[box_index][1]],
                    Quaternion(scalar=np.cos(yaw[box_index]/2),vector=[0,0,np.sin(yaw[box_index]/2)]).inverse,
                    name=name_map_reverse[case['name'][box_index]],
                    token="token")
            lidar_to_world(box,sample['data']['LIDAR_TOP'])
            boxes.append(box)
      
        pred_str = ''
        for box_index,box in enumerate(boxes):
            pred_str += '%f %f %f %f %f %f %f %f %s ' % (score[box_index],box.center[0],box.center[1],box.center[2],box.wlh[0],box.wlh[1],box.wlh[2],box.orientation.radians,box.name)
        
        submission[data[index]['level5_token']] = pred_str
      
    df_submission = pd.DataFrame(submission.items(),columns=['Id','PredictionString'])

    return df_submission
  

def show_scene(data,res,index):

    points_v = np.fromfile('/content/data/'+data[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 5])

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


    case = res[index]
    score = case['score']
    number = len(np.where(score>0.5)[0])
    loc = case['location']
    dim = case['dimensions']
    yaw = case['rotation_y']

    boxes = []

    for index in range(number):
        box = Box(
                [loc[index][2],-loc[index][0],-loc[index][1]],
          
                [dim[index][0],dim[index][2],dim[index][1]],
          
                Quaternion(scalar=np.cos(yaw[index] / 2), vector=[0, 0, np.sin(yaw[index] / 2)]).inverse,
          
                name=name_map_reverse[case['name']],
                token="token",
            )
        boxes.append(box)


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

    points_v = np.fromfile('/content/data/'+data[index]['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 5])

    case = res[index]

    score = case['score']
    number = len(np.where(score>0.4)[0])
      
    ax.scatter(points_v[:, 0], points_v[:, 1], s=0.01, c="white", cmap='grey')

    #points_v = np.fromfile('/content/data/train_lidar/host-a004_lidar1_1232815254300468606.bin', dtype=np.float32, count=-1).reshape([-1, 5])
    #loc = res[index]['location']
    #ax.scatter(points_v[:, 0], points_v[:, 3], s=0.01, c="green", cmap='grey')



    ax.scatter(loc[:number,2], -loc[:number,0], s=10, c="red", cmap='grey')

    ax.scatter(loc[:number,2]-dim[:number,0]/2.0, -loc[:number,0]-dim[:number,2]/2.0, s=10, c="green", cmap='grey')
    ax.scatter(loc[:number,2]+dim[:number,0]/2.0, -loc[:number,0]-dim[:number,2]/2.0, s=10, c="green", cmap='grey')
    ax.scatter(loc[:number,2]-dim[:number,0]/2.0, -loc[:number,0]+dim[:number,2]/2.0, s=10, c="green", cmap='grey')
    ax.scatter(loc[:number,2]+dim[:number,0]/2.0, -loc[:number,0]+dim[:number,2]/2.0, s=10, c="green", cmap='grey')

    ax.set_xlim(-20,100)
    ax.set_ylim(-50,25)


