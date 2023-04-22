import numpy as np
import open3d as o3d

import torch
from tqdm import tqdm
import datetime

def visualize_numpy(pc_numpy, colors = None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_numpy)
    try:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    except:
        pass

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    ctr = vis.get_view_control()
    ctr.set_up((1, 0, 0))
    ctr.set_front((0, 1, 0))

    vis.run()
    
    # o3d.visualization.draw_geometries([point_cloud])
    
def visualize_part(net, testloader):
    color_choices = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #3
                     (1, 0, 0), (0, 1, 0), #5
                     (1, 0, 0), (0, 1, 0), #7
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #11
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #15
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), # 18
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #21
                     (1, 0, 0), (0, 1, 0), #23
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #27
                     (1, 0, 0), (0, 1, 0), #29
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), # 35
                     (1, 0, 0), (0, 1, 0), #37
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), #40
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), #43
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), #46
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), #49
    ]

    net.eval()

    with torch.no_grad():
        for batch_idx, data_dic in enumerate(tqdm(testloader)):
            if ((batch_idx % 10 == 0) and (batch_idx > 1260)):
                data_dic = net(data_dic)        
                points = data_dic['points'].squeeze(0).cpu().numpy()
                label = data_dic['seg_id'].squeeze(0).cpu().numpy()
                pred = torch.argmax(data_dic['pred_score_logits'], dim = -1).squeeze(0).cpu().numpy()

                color_list = np.zeros_like(points)
                for i, pred_id in enumerate(pred):
                    color_list[i] = color_choices[int(pred_id)]            
                visualize_numpy(points, colors=color_list)

                color_list = np.zeros_like(points)
                for i, label_id in enumerate(label):
                    color_list[i] = color_choices[int(label_id)]            
                visualize_numpy(points, colors=color_list)
