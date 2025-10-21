from math import perm
import os
import numpy as np
import open3d as o3d
import os.path as osp
import glob
import pycolmap
import copy  # ### NEW
from typing import Optional, Tuple  # ### NEW

from utils.thread_utils import run_on_thread
import utils.colmap_read_model as read_model
from utils.colmap2mvsnet_acm import processing_single_scene
from PIL import Image
# opencv
import cv2 as cv
import trimesh
import glob


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
        self._pcd_raw = None     # ### NEW: keep the unfiltered sparse cloud if you want to compare
        self._mesh = None        # ### NEW: last generated mesh
        self._thread = None
        self._active_camera_name = None
        self._cameras = dict()
        self._vis = None

        self._gpu_index = gpu_index
        self._camera_model = camera_model
        self._matcher = matcher
        if self._matcher not in ['exhaustive_matcher', 'vocab_tree_matcher', 'sequential_matcher']:
            raise ValueError(f'Only support exhaustive_matcher and vocab_tree_matcher, got {self._matcher}')

        # ### NEW: sensible default preprocessing / meshing params
        self._default_cleaning = dict(
            voxel_size=None,           # e.g. 0.005 (meters/scene units)
            sor_nb_neighbors=30,
            sor_std_ratio=2.0,
            radius=None,               # auto-computed if None
            min_points=16,
            dbscan_keep_largest=True,  # keep largest connected component if DBSCAN useful
            dbscan_eps=None,           # auto if None
            dbscan_min_points=30,
            densify_to=None            # e.g. 100000 to oversample via Poisson (optional)
        )
        self._default_mesh = dict(
            method='poisson',          # 'poisson' or 'bpa'
            poisson_depth=10,          # Poisson octree depth (8–11 typical)
            poisson_scale=1.1,
            poisson_linear_fit=False,
            bpa_ball_factor=1.5,       # factor * mean NN distance
            bpa_ball_radii=None,       # override radii list if desired
            crop_to_bbox_scale=1.03    # crop Poisson hull to slightly expanded bbox of points
        )

    # ----------------------------- existing properties -----------------------------
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

    # ### NEW: expose raw (uncleaned) pcd if you want to compare
    @property
    def pcd_raw(self):
        if self._pcd_raw is None:
            raise ValueError('No raw point cloud is available yet.')
        return self._pcd_raw

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

    # ================================ NEW: utility helpers ================================

    # ### NEW
    @staticmethod
    def _auto_scales_from_pcd(pcd: o3d.geometry.PointCloud, k: int = 30) -> Tuple[float, float]:
        """
        Estimate characteristic scales from NN distances.
        Returns (mean_nn, eps_dbscan_default).
        """
        if len(pcd.points) == 0:
            return 0.0, 0.0
        dists = pcd.compute_nearest_neighbor_distance()
        if len(dists) == 0:
            return 0.0, 0.0
        mean_nn = float(np.mean(dists))
        return mean_nn, 1.5 * mean_nn

    # ### NEW
    @staticmethod
    def _ensure_normals(pcd: o3d.geometry.PointCloud, radius: Optional[float] = None, k: int = 30):
        if not pcd.has_normals():
            if radius is None:
                mean_nn, _ = ColmapAPI._auto_scales_from_pcd(pcd, k=k)
                radius = max(1e-6, 2.5 * mean_nn)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max(k, 20))
            )
            pcd.orient_normals_consistent_tangent_plane(2 * k)

    # ================================ NEW: cleaning pipeline ================================
    # ### NEW
    def clean_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: Optional[float] = None,
        sor_nb_neighbors: int = 30,
        sor_std_ratio: float = 2.0,
        radius: Optional[float] = None,
        min_points: int = 16,
        dbscan_keep_largest: bool = True,
        dbscan_eps: Optional[float] = None,
        dbscan_min_points: int = 30,
        densify_to: Optional[int] = None
    ) -> o3d.geometry.PointCloud:
        """
        Clean a point cloud by:
          1) optional voxel downsample
          2) Statistical Outlier Removal
          3) Radius Outlier Removal
          4) Optional largest-cluster keep via DBSCAN
          5) Optional densify (via Poisson surface sampling)
        """
        if pcd is None or len(pcd.points) == 0:
            raise ValueError("clean_point_cloud: empty point cloud")

        p = copy.deepcopy(pcd)
        indices = np.arange(len(p.points), dtype=np.int64)

        # 1) voxel downsample (makes later steps more stable)
        if voxel_size is not None and voxel_size > 0:
            p = p.voxel_down_sample(voxel_size)

        # Compute auto scales if needed
        mean_nn, eps_auto = self._auto_scales_from_pcd(p, k=30)

        # 2) Statistical outlier removal
        if sor_nb_neighbors is not None and sor_nb_neighbors > 0 and sor_std_ratio is not None:
            p, ind = p.remove_statistical_outlier(nb_neighbors=int(sor_nb_neighbors), std_ratio=float(sor_std_ratio))
            indices = indices[np.asarray(ind, dtype=np.int64)]

        # 3) Radius outlier removal
        use_radius = radius if radius is not None else (3.0 * mean_nn if mean_nn > 0 else None)
        if use_radius is not None and min_points is not None and min_points > 0:
            p, ind = p.remove_radius_outlier(nb_points=int(min_points), radius=float(use_radius))
            indices = indices[np.asarray(ind, dtype=np.int64)]

        # 4) Largest connected component (DBSCAN) to drop floating noise clusters
        if dbscan_keep_largest and len(p.points) > 0:
            eps_use = dbscan_eps if dbscan_eps is not None else eps_auto
            if eps_use and eps_use > 0:
                labels = np.array(p.cluster_dbscan(eps=float(eps_use), min_points=int(dbscan_min_points), print_progress=False))
                if labels.size > 0:
                    mask = labels >= 0
                    if mask.any():
                        # keep largest non-negative label
                        valid_labels, counts = np.unique(labels[mask], return_counts=True)
                        largest_label = valid_labels[np.argmax(counts)]
                        keep = (labels == largest_label)
                        # p = p.select_by_index(np.where(keep)[0])
                        idx_keep = np.where(keep)[0]
                        p = p.select_by_index(idx_keep.tolist())
                        indices = indices[idx_keep]

        # 5) Optional densify
        if densify_to is not None and densify_to > len(p.points):
            p = self.densify_point_cloud_via_poisson(p, target_points=int(densify_to))

        return p, np.asarray(indices, dtype=np.int64)

    # ### NEW
    def densify_point_cloud_via_poisson(
        self,
        pcd: o3d.geometry.PointCloud,
        target_points: int = 150000,
        poisson_depth: int = 9,
        poisson_scale: float = 1.2,
        poisson_linear_fit: bool = False
    ) -> o3d.geometry.PointCloud:
        """
        Densify a sparse cloud by:
          normals → Poisson mesh → sample many points on mesh
        """
        p = copy.deepcopy(pcd)
        self._ensure_normals(p)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            p, depth=int(poisson_depth), scale=float(poisson_scale), linear_fit=bool(poisson_linear_fit)
        )
        # crop mesh to bbox of original points (slightly expanded)
        bbox = p.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.05, bbox.get_center())
        mesh = mesh.crop(bbox)

        dense = mesh.sample_points_uniformly(number_of_points=int(target_points))
        # carry color if original had it
        if p.has_colors():
            dense.colors = o3d.utility.Vector3dVector(np.tile(np.mean(np.asarray(p.colors), axis=0), (len(dense.points), 1)))
        return dense

    # ================================ NEW: meshing ================================
    # ### NEW
    def mesh_current_point_cloud(
        self,
        method: str = 'poisson',
        poisson_depth: int = 10,
        poisson_scale: float = 1.1,
        poisson_linear_fit: bool = False,
        bpa_ball_factor: float = 1.5,
        bpa_ball_radii: Optional[list] = None,
        crop_to_bbox_scale: float = 1.03
    ) -> o3d.geometry.TriangleMesh:
        """
        Mesh the current (cleaned) point cloud.
        """
        if self._pcd is None or len(self._pcd.points) == 0:
            raise ValueError("No point cloud available to mesh. Run estimate_cameras or set self._pcd.")
        p = copy.deepcopy(self._pcd)
        self._ensure_normals(p)

        if method.lower() == 'poisson':
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                p, depth=int(poisson_depth), scale=float(poisson_scale), linear_fit=bool(poisson_linear_fit)
            )
            # crop to bounding box
            bbox = p.get_axis_aligned_bounding_box()
            bbox = bbox.scale(float(crop_to_bbox_scale), bbox.get_center())
            mesh = mesh.crop(bbox)
        elif method.lower() in ('bpa', 'ball_pivoting', 'ballpivoting'):
            mean_nn, _ = self._auto_scales_from_pcd(p, k=30)
            if bpa_ball_radii is None or len(bpa_ball_radii) == 0:
                r = max(1e-6, float(bpa_ball_factor) * mean_nn) if mean_nn > 0 else 0.01
                bpa_ball_radii = [r, 2.0 * r]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                p, o3d.utility.DoubleVector([float(x) for x in bpa_ball_radii])
            )
        else:
            raise ValueError(f"Unknown meshing method: {method}")

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        self._mesh = mesh
        return mesh

    # ================================ NEW: baseline sphere test ================================
    # ### NEW
    @staticmethod
    def _make_noisy_sphere_point_cloud(
        radius: float = 1.0,
        n_points: int = 8000,
        noise_std: float = 0.01,
        outlier_ratio: float = 0.05,
        color=(0.8, 0.2, 0.2)
    ) -> o3d.geometry.PointCloud:
        """
        Generate a noisy sphere surface point cloud with optional outliers.
        Uniform on sphere via normalized Gaussian sampling.
        """
        n_core = int(n_points * (1.0 - outlier_ratio))
        n_out = n_points - n_core

        # Uniform directions on S^2
        dirs = np.random.normal(size=(n_core, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

        # radius with small Gaussian noise
        r = radius + np.random.normal(scale=noise_std, size=(n_core, 1))
        xyz_core = dirs * r

        # outliers in a cube around the sphere
        out_box = 1.6 * radius
        xyz_out = np.random.uniform(low=-out_box, high=out_box, size=(n_out, 3)) if n_out > 0 else np.empty((0, 3))

        xyz = np.vstack([xyz_core, xyz_out])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=float), (xyz.shape[0], 1)))
        return pcd

    # ### NEW
    def run_sphere_baseline(
        self,
        radius: float = 1.0,
        n_points: int = 20000,
        noise_std: float = 0.01,
        outlier_ratio: float = 0.05,
        cleaning_params: Optional[dict] = None,
        meshing_params: Optional[dict] = None,
        save_dir: Optional[str] = None
    ):
        """
        1) Generate noisy sphere cloud
        2) Clean using the same pipeline
        3) Mesh it
        4) Optionally save all artifacts to disk
        Returns: (pcd_raw, pcd_clean, mesh)
        """
        pcd_raw = self._make_noisy_sphere_point_cloud(radius, n_points, noise_std, outlier_ratio)
        cp = dict(self._default_cleaning)
        if cleaning_params: cp.update(cleaning_params)
        mp = dict(self._default_mesh)
        if meshing_params: mp.update(meshing_params)

        pcd_clean, _ = self.clean_point_cloud(
            pcd_raw,
            voxel_size=cp['voxel_size'],
            sor_nb_neighbors=cp['sor_nb_neighbors'],
            sor_std_ratio=cp['sor_std_ratio'],
            radius=cp['radius'],
            min_points=cp['min_points'],
            dbscan_keep_largest=cp['dbscan_keep_largest'],
            dbscan_eps=cp['dbscan_eps'],
            dbscan_min_points=cp['dbscan_min_points'],
            densify_to=cp['densify_to']
        )

        # Temporarily set _pcd to call the mesher
        _old = self._pcd
        self._pcd = pcd_clean
        mesh = self.mesh_current_point_cloud(
            method=mp['method'],
            poisson_depth=mp['poisson_depth'],
            poisson_scale=mp['poisson_scale'],
            poisson_linear_fit=mp['poisson_linear_fit'],
            bpa_ball_factor=mp['bpa_ball_factor'],
            bpa_ball_radii=mp['bpa_ball_radii'],
            crop_to_bbox_scale=mp['crop_to_bbox_scale']
        )
        self._pcd = _old

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            o3d.io.write_point_cloud(osp.join(save_dir, "sphere_raw.ply"), pcd_raw)
            o3d.io.write_point_cloud(osp.join(save_dir, "sphere_clean.ply"), pcd_clean)
            o3d.io.write_triangle_mesh(osp.join(save_dir, "sphere_mesh.ply"), mesh)
            print(f"[Baseline] Saved to {save_dir}")

        return pcd_raw, pcd_clean, mesh

    # ================================ existing pipeline with light edits ================================
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
        def _ensure_dirs():
            os.makedirs(osp.join(self.data_path, "colmap"), exist_ok=True)
            os.makedirs(self.sparse_dir, exist_ok=True)

        def _list_model_subdirs(base):
            if not osp.isdir(base):
                return []
            subs = []
            for d in os.listdir(base):
                full = osp.join(base, d)
                if osp.isdir(full) and d.isdigit():
                    subs.append((int(d), full))
            subs.sort(key=lambda x: x[0])
            return subs

        def _load_cached_reconstructions():
            maps = {}
            for mid, mdir in _list_model_subdirs(self.sparse_dir):
                try:
                    maps[mid] = pycolmap.Reconstruction(mdir)
                except Exception as e:
                    print(f"Warning: failed to load reconstruction {mdir}: {e}")
            return maps

        def _run_matching(db_path):
            # Choose matching strategy
            if self._matcher == 'exhaustive_matcher':
                pycolmap.match_exhaustive(db_path, sift_options=sift_opts)
            elif self._matcher == 'vocab_tree_matcher':
                if not osp.isfile(VOCAB_PATH):
                    raise FileNotFoundError(
                        f"Vocabulary tree not found at {VOCAB_PATH}. "
                        f"Either place the file there or change VOCAB_PATH."
                    )
                pycolmap.match_vocabulary_tree(db_path, VOCAB_PATH, sift_options=sift_opts)
            elif self._matcher == 'sequential_matcher':
                # Uses temporal proximity (good for videos)
                pycolmap.match_sequential(db_path, sift_options=sift_opts)
            else:
                raise ValueError(f"Unknown matcher: {self._matcher}")

        def _run_pipeline():
            # Extract SIFT, match per chosen matcher, run mapper; write models under sparse_dir
            if osp.isfile(self.database_path):
                os.remove(self.database_path)

            pycolmap.extract_features(
                self.database_path,
                self.image_dir,
                camera_model=self.camera_model,
                sift_options=sift_opts
            )
            _run_matching(self.database_path)
            maps_local = pycolmap.incremental_mapping(
                self.database_path,
                self.image_dir,
                self.sparse_dir
            )
            return maps_local

        if not osp.isdir(self.image_dir):
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")

        image_files = self._list_images_in_folder(self.image_dir)
        if len(image_files) == 0:
            raise RuntimeError(f"No images found under {self.image_dir}")
        
        sift_opts = {
            "use_gpu": False,
            "gpu_index": self._gpu_index
        }

        _ensure_dirs()

        if recompute:
            os.makedirs(self.data_path + "/colmap", exist_ok=True)
            pycolmap.extract_features(self.database_path, self.image_dir)
            pycolmap.match_exhaustive(self.database_path)
            maps = pycolmap.incremental_mapping(self.database_path, self.image_dir, self.sparse_dir)
        else: 
            maps = _load_cached_reconstructions()
            if not maps:
                print("No cached data found. Running COLMAP...")
                os.makedirs(self.data_path + "/colmap", exist_ok=True)
                pycolmap.extract_features(self.database_path, self.image_dir)
                pycolmap.match_exhaustive(self.database_path)
                maps = pycolmap.incremental_mapping(self.database_path, self.image_dir, self.sparse_dir)
            else:
                print("Cached data found. Loading...")

        # -------------------- Convert COLMAP → Open3D and camera dict --------------------
        def colmap_points_to_open3d(points3D):
            xyz = []
            rgb = []
            point_ids = []
            image_ids = []

            for pid in sorted(points3D.keys()):
                p3d = points3D[pid]
                point_ids.append(pid)
                xyz.append(p3d.xyz)
                rgb.append(p3d.color / 255.0)
                elems = getattr(p3d.track, "elements", p3d.track)
                ids = [int(e.image_id) for e in elems]
                image_ids.append([int(i) for i in ids])

            xyz = np.array(xyz)
            rgb = np.array(rgb)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            if rgb.shape[0] == xyz.shape[0]:
                pcd.colors = o3d.utility.Vector3dVector(rgb)

            return pcd, np.asarray(point_ids, dtype=np.int64), image_ids

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

        # Pick the first available reconstruction id robustly
        if not maps:
            raise RuntimeError("COLMAP mapping produced no models.")
        chosen_id = sorted(maps.keys())[0]
        reconstruction = maps[chosen_id]

        pcd_raw, point_ids, point_image_ids = colmap_points_to_open3d(reconstruction.points3D)

        # ### NEW: run cleaning on the sparse point cloud (toggle/adjust params via self._default_cleaning)
        pcd_clean, keep_indices = self.clean_point_cloud(
            pcd_raw,
            voxel_size=self._default_cleaning['voxel_size'],
            sor_nb_neighbors=self._default_cleaning['sor_nb_neighbors'],
            sor_std_ratio=self._default_cleaning['sor_std_ratio'],
            radius=self._default_cleaning['radius'],
            min_points=self._default_cleaning['min_points'],
            dbscan_keep_largest=self._default_cleaning['dbscan_keep_largest'],
            dbscan_eps=self._default_cleaning['dbscan_eps'],
            dbscan_min_points=self._default_cleaning['dbscan_min_points'],
            densify_to=self._default_cleaning['densify_to']
        )

        colmap_cameras = colmap_cameras_to_dict(reconstruction)

        ####### End of your code #####################

        self._pcd_raw = pcd_raw       # ### NEW
        self._pcd = pcd_clean         # cleaned becomes the default
        self._cameras = colmap_cameras
        self.activate_camera_name = self.camera_names[0]
        # save clean pcd 
        # o3d.io.write_point_cloud(osp.join(self.data_path, "colmap/cleaned.ply"), self._pcd)
        # save clean pcd to point3D.bin
        self._colmap_point_ids = point_ids
        self._colmap_image_ids = point_image_ids 
        self._clean_indices = keep_indices
        self._reconstruction = reconstruction  # keep the full reconstruction object
        self.save_for_recon()
        print('COLMAP camera estimation done.')

    def save_for_recon(self):
        realdir = osp.join(self.data_path, 'colmap')
        camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
        camdata = read_model.read_cameras_binary(camerasfile)
        
        # cam = camdata[camdata.keys()[0]]
        list_of_keys = list(camdata.keys())
        cam = camdata[list_of_keys[0]]
        print( 'Cameras', len(cam))

        h, w, f = cam.height, cam.width, cam.params[0]
        # w, h, f = factor * w, factor * h, factor * f
        hwf = np.array([h,w,f]).reshape([3,1])
        
        imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
        imdata = read_model.read_images_binary(imagesfile)
        
        w2c_mats = []
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        
        names = [imdata[k].name for k in imdata]
        print( 'Images #', len(names))
        perm = np.argsort(names)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape([3,1])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            w2c_mats.append(m)
        
        w2c_mats = np.stack(w2c_mats, 0)
        c2w_mats = np.linalg.inv(w2c_mats)
        
        poses = c2w_mats[:, :3, :4].transpose([1,2,0])
        poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
        
        # points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
        # pts3d = read_model.read_points3d_binary(points3dfile)
        
        # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
        poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
        # added
        poses = np.moveaxis(poses, -1, 0)
        poses = poses[perm]

        images = [self._reconstruction.images[k] for k in sorted(self._reconstruction.images.keys())]
        ordered_images = [images[i] for i in perm]
        image_id_to_index = {im.image_id: idx for idx, im in enumerate(ordered_images)}
        clean_to_raw = self._clean_indices
        # if clean_to_raw is None or len(clean_to_raw) != len(self._pcd.points):
        #     clean_to_raw = self._map_clean_points_to_raw()
        clean_to_raw = np.asarray(clean_to_raw, dtype=np.int64)
        points = np.asarray(self._pcd.points, dtype=np.float32)
        view_id = np.zeros((len(ordered_images), len(points)), dtype=bool)
        for clean_idx, raw_idx in enumerate(clean_to_raw):
            if raw_idx < 0 or raw_idx >= len(self._colmap_image_ids):
                continue
            for img_id in self._colmap_image_ids[raw_idx]:
                mapped = image_id_to_index.get(int(img_id))
                if mapped is not None:
                    view_id[mapped, clean_idx] = True
        
        # np.save(osp.join(self.data_path, "poses.npy"), poses) # NO need to save anymore
        out_dir = os.path.join(self.data_path, 'preprocessed')
        os.makedirs(out_dir, exist_ok=True)
        sfm_dir = osp.join(out_dir, "sfm_pts")
        os.makedirs(sfm_dir, exist_ok=True)
        np.save(osp.join(sfm_dir, "points.npy"), points)
        np.save(osp.join(sfm_dir, "view_id.npy"), view_id)
        o3d.io.write_point_cloud(osp.join(self.data_path, "sparse_points_interest.ply"), self._pcd)

        # gen_cameras.py
        poses_hwf = poses
        poses_raw = poses_hwf[:, :, :4]
        hwf = poses_hwf[:, :, 4]
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose[:3, :4] = poses_raw[0]
        cam_dict = dict()
        n_images = len(poses_raw)

        # Convert space
        convert_mat = np.zeros([4, 4], dtype=np.float32)
        convert_mat[0, 1] = 1.0
        convert_mat[1, 0] = 1.0
        convert_mat[2, 2] =-1.0
        convert_mat[3, 3] = 1.0

        for i in range(n_images):
            pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            pose[:3, :4] = poses_raw[i]
            pose = pose @ convert_mat
            h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
            intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
            intrinsic[0, 2] = (w - 1) * 0.5
            intrinsic[1, 2] = (h - 1) * 0.5
            w2c = np.linalg.inv(pose)
            world_mat = intrinsic @ w2c
            world_mat = world_mat.astype(np.float32)
            cam_dict['camera_mat_{}'.format(i)] = intrinsic
            cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
            cam_dict['world_mat_{}'.format(i)] = world_mat
            cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)


        pcd = trimesh.load(os.path.join(self.data_path, 'sparse_points_interest.ply'))
        vertices = pcd.vertices
        bbox_max = np.max(vertices, axis=0)
        bbox_min = np.min(vertices, axis=0)
        center = (bbox_max + bbox_min) * 0.5
        radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
        scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
        scale_mat[:3, 3] = center

        for i in range(n_images):
            cam_dict['scale_mat_{}'.format(i)] = scale_mat
            cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

        
        os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

        image_list = glob.glob(os.path.join(self.data_path, 'images/*.png')) + glob.glob(os.path.join(self.data_path, 'images/*.jpg'))
        image_list.sort()

        for i, image_path in enumerate(image_list):
            img = cv.imread(image_path)
            cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
            cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)

        np.savez(os.path.join(out_dir, 'cameras.npz'), **cam_dict)
        processing_single_scene(in_folder=self.data_path, out_folder=out_dir)


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
