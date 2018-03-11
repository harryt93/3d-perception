#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):


    # Convert ROS msg to PCL data
    #############################
 
    cloud = ros_to_pcl(pcl_msg)

        
    # Statistical Outlier Filtering
    ###############################

    # Create a outlier filter object for our input point cloud.
    outlier_filter = cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(10)

    # Set threshold scale factor
    x = 0.01

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter() 


    # Output objects for visualisation
    ##################################

    # Convert objects to ros format
    no_noise_cloud_ros = pcl_to_ros(cloud_filtered)

    # Publish the objects
    no_noise_pub.publish(no_noise_cloud_ros)
 

    # Voxel Grid filter
    ###################

    # Create a VoxelGrid filter object for our input point cloud.
    vox = cloud_filtered.make_voxel_grid_filter()

    # Leaf size.
    LEAF_SIZE = 0.01

    # Set the leaf size.
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point.
    cloud_filtered = vox.filter()


    # PassThrough filter Z
    ######################

    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 0.8
    passthrough.set_filter_limits(axis_min, axis_max)

    # Use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()


    # PassThrough filter X
    ######################

    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.35
    axis_max = 0.8
    passthrough.set_filter_limits(axis_min, axis_max)

    # Use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()


    # RANSAC plane segmentation
    ###########################

    # Create the segmentation object.
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit.
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model.
    MAX_DISTANCE = 0.01
    seg.set_distance_threshold(MAX_DISTANCE)

    # Call the segment function.
    inliers, coefficients = seg.segment()


    # Extract objects
    #################

    # Extract outliers
    cloud_objects = cloud_filtered.extract(inliers, negative=True)


    # Output objects for visualisation
    ##################################

    # Convert objects to ros format
    cloud_objects_ros = pcl_to_ros(cloud_objects)

    # Publish the objects
    objects_pub.publish(cloud_objects_ros)


    # Euclidean Clustering
    ######################

    # Turn to XYZ
    white_cloud = XYZRGB_to_XYZ(cloud_objects)

    # Construct K-D Tree
    tree = white_cloud.make_kdtree()

    # Create cluster extraction object.
    ec = white_cloud.make_EuclideanClusterExtraction()

    # Set cluster parameters
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(5000)

    # Search the K-D Tree for clusters
    ec.set_SearchMethod(tree)

    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()


    # Create Cluster-Mask Point Cloud
    #################################

    # Assign a colour corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create a new cloud containing all clusters, each with a unique colour.
    cloud_cluster = pcl.PointCloud_PointXYZRGB()
    cloud_cluster.from_list(color_cluster_point_list)


    # Output objects for visualisation
    ##################################

    # Convert objects to ros format
    cluster_cloud_ros = pcl_to_ros(cloud_cluster)

    # Publish the objects
    clusters_pub.publish(cluster_cloud_ros)
   

    # Detect Objects
    ################

    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Call the PICK and PLACE
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# Utility function for finding a centroid 
def find_centroid(object_list, item):   
 
    centroid = None

    for _object in object_list:
        if _object.label == item:
            points_arr = ros_to_pcl(_object.cloud).to_array()
            centroid = np.mean(points_arr, axis=0)[:3]            


    return centroid

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Setup all of the message fields
    #################################

    TEST_SCENE_NUM = Int32()
    OBJECT_NAME = String()
    ARM_NAME = String()
    PICK_POSE = Pose()
    PLACE_POSE = Pose() 

    
    # Extract picklist
    ##################

    pick_list = []

    # Get parameters
    object_list_param = rospy.get_param('/object_list')

    # Parse parameters into individual variables
    for i in range(0, len(object_list_param)):
        name = object_list_param[i]['name']
        group = object_list_param[i]['group']
    	pick_list.append((name, group)) 


    # Extract Box Poses
    ###################          

    # Get parameters
    box_param = rospy.get_param('/dropbox')

    # Parse parameters into individual variables 
    red_box_pose = box_param[0]['position'] 
    green_box_pose = box_param[1]['position']     


    # Generate dict_list
    ####################

    dict_list = []

    # Loop through the pick list
    for item, group in pick_list:

        # Store the test scene number field
        TEST_SCENE_NUM.data = 4

        # Try to get the centroid for the item
        centroid = find_centroid(object_list, item)
       
        # Store the item name in the relevant field
        OBJECT_NAME.data = item

        # Decide on the arm, and store the result.
        if group == "green":
            ARM_NAME.data = 'right'

            PLACE_POSE.position.x = green_box_pose[0]
            PLACE_POSE.position.y = green_box_pose[1]
            PLACE_POSE.position.z = green_box_pose[2]

        else:
            ARM_NAME.data = 'left'

            PLACE_POSE.position.x = red_box_pose[0]
            PLACE_POSE.position.y = red_box_pose[1]
            PLACE_POSE.position.z = red_box_pose[2]

        # If centroid is not None, then it found an object.
        if centroid is not None:

            # Create 'place_pose' for the object
            PICK_POSE.position.x = np.asscalar(centroid[0])
            PICK_POSE.position.y = np.asscalar(centroid[1])
            PICK_POSE.position.z = np.asscalar(centroid[2])

        else:

            PICK_POSE.position.x = 0
            PICK_POSE.position.y = 0
            PICK_POSE.position.z = 0


        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(TEST_SCENE_NUM, ARM_NAME, OBJECT_NAME, PICK_POSE, PLACE_POSE)
        dict_list.append(yaml_dict)        
        

    # Output your request parameters into output yaml file
    ######################################################

    send_to_yaml("output4.yaml", dict_list)


if __name__ == '__main__':

    # ROS node initialization
    #########################

    rospy.init_node('object_recognition', anonymous=True)

    # Create Subscribers
    ####################

    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    ###################

    no_noise_pub = rospy.Publisher("/no_noise_cloud", PointCloud2, queue_size=1)
    objects_pub = rospy.Publisher("/objects_cloud", PointCloud2, queue_size=1)
    clusters_pub = rospy.Publisher("/clusters_cloud", PointCloud2, queue_size=1)    
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    ######################

    model = pickle.load(open('/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    #################################
    while not rospy.is_shutdown():
        rospy.spin()
