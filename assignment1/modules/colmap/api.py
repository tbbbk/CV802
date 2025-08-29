import os
import numpy as np
import open3d as o3d
import os.path as osp
import glob
import pycolmap

from utils.thread_utils import run_on_thread


VOCAB_PATH = 'modules/colmap/vocab_tree_flickr100K_words32K.bin'


class ColmapAPI:
    def __init__(
        self,
        gpu_index,
        camera_model,
        matcher,
    ):
        self._data_path = None
        self._pcd = None
        self._thread = None
        self._active_camera_name = None
        self._cameras = dict()
        self._vis = None

        self._gpu_index = gpu_index
        self._camera_model = camera_model
        self._matcher = matcher
        if self._matcher not in ['exhaustive_matcher', 'vocab_tree_matcher', 'sequential_matcher']:
            raise ValueError(f'Only support exhaustive_matcher and vocab_tree_matcher, got {self._matcher}')

    @property
    def data_path(self):
        if self._data_path is None:
            raise ValueError(f'Data path was not set')
        return self._data_path

    @data_path.setter
    def data_path(self, new_data_path):
        self._data_path = new_data_path

    @property
    def image_dir(self):
        return osp.join(self.data_path, 'images')

    @property
    def database_path(self):
        return osp.join(self.data_path, 'colmap/database.db')

    @property
    def sparse_dir(self):
        return osp.join(self.data_path, 'colmap/sparse')

    @property
    def num_cameras(self):
        return len(self._cameras)

    @property
    def camera_names(self):
        return list(self._cameras.keys())

    @property
    def pcd(self):
        if self._pcd is None:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        return self._pcd

    @property
    def activate_camera_name(self):
        if len(self._cameras) == 0:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        return self._active_camera_name

    @activate_camera_name.setter
    def activate_camera_name(self, new_value):
        if len(self._cameras) == 0:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        self._active_camera_name = new_value

    @property
    def camera_model(self):
        return self._camera_model

    @camera_model.setter
    def camera_model(self, new_value):
        self._camera_model = new_value 

    @property
    def matcher(self):
        return self._matcher

    @matcher.setter
    def matcher(self, new_value):
        self._matcher = new_value

    def check_colmap_folder_valid(self):
        database_path = self.database_path
        image_dir = self.image_dir
        sparse_dir = self.sparse_dir

        print('Database file:', database_path)
        print('Image path:', image_dir)
        print('Bundle adjustment path:', sparse_dir)

        is_valid = \
            osp.isfile(database_path) and \
            osp.isdir(image_dir) and \
            osp.isdir(sparse_dir)

        return is_valid

    @run_on_thread
    def _estimate_cameras(self, recompute):
        ''' Assignment 1

        In this assignment, you need to compute two things:
            pcd: A colored point cloud represented using open3d.geometry.PointCloud
            cameras: A dictionary of the following format:
                {
                    camera_name_01 [str]: {
                        'extrinsics': [rotation [Matrix 3x3], translation [Vector 3]]
                        'intrinsics': {
                            'width': int
                            'height': int
                            'fx': float
                            'fy': float
                            'cx': float
                            'cy': float
                        }
                    }
                    ...
                }

            You can check the extract_camera_parameters method to understand how the cameras are used.
        '''

        ## Insert your code below
        if recompute:
            # Compute the result once and cache it in self.data_path. This will save a lot of time on the next run
            # If you use COLMAP, save the database and bundle adjustment data in self.database_dir and
            # self.sparse_dir, respectively.
            pass

        # You can load the cached data here before adding points and cameras

        # Add points
            
        def colmap_points_to_open3d(points3D):
            xyz = []
            rgb = []

            for p3d in points3D.values():
                xyz.append(p3d.xyz)
                rgb.append(p3d.color / 255.0)

            xyz = np.array(xyz)
            rgb = np.array(rgb)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            return pcd
        
        os.makedirs(self.data_path + "/colmap", exist_ok=True)
        pycolmap.extract_features(self.database_path, self.image_dir)
        pycolmap.match_exhaustive(self.database_path)
        maps = pycolmap.incremental_mapping(self.database_path, self.image_dir, self.sparse_dir)

        pcd = colmap_points_to_open3d(maps[0].points3D)

        def colmap_cameras_to_dict(reconstruction):
            cameras_dict = {}

            for _, image in reconstruction.images.items():
                cam = reconstruction.cameras[image.camera_id]

                transform = image.cam_from_world()
                R = transform.rotation.matrix()
                t = transform.translation
                
                intrinsics = {
                    "width": cam.width,
                    "height": cam.height,
                    "fx": cam.focal_length_x,
                    "fy": cam.focal_length_y,
                    "cx": cam.params[1],
                    "cy": cam.params[2],
                }

                cameras_dict[image.name] = {
                    "extrinsic": [R, t],
                    "intrinsic": intrinsics
                }

            return cameras_dict
        
        colmap_cameras = colmap_cameras_to_dict(maps[0])

        ####### End of your code #####################

        self._pcd = pcd
        self._cameras = colmap_cameras
        self.activate_camera_name = self.camera_names[0]

    @staticmethod
    def _list_images_in_folder(directory):
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg'}
        files = sorted(glob.glob(osp.join(directory, '*')))
        files = list(filter(lambda x: osp.splitext(x)[1].lower() in image_extensions, files)) 
        return files

    def estimate_done(self):
        return not self._thread.is_alive()

    def estimate_cameras(self, recompute=False):
        self._thread = self._estimate_cameras(recompute)

    def extract_camera_parameters(self, camera_name):
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self._cameras[camera_name]['intrinsic']['width'],
            self._cameras[camera_name]['intrinsic']['height'],
            self._cameras[camera_name]['intrinsic']['fx'],
            self._cameras[camera_name]['intrinsic']['fy'],
            self._cameras[camera_name]['intrinsic']['cx'],
            self._cameras[camera_name]['intrinsic']['cy'],
        )

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = self._cameras[camera_name]['extrinsic'][0]
        extrinsics[:3, 3] = self._cameras[camera_name]['extrinsic'][1]
        extrinsics = extrinsics

        return intrinsics, extrinsics
