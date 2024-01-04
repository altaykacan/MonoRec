import json
import glob
import os
import datetime as dt
from collections import namedtuple
from pathlib import Path

import numpy as np
import pykitti
import torch
import torchvision
from PIL import Image
from scipy import sparse
from skimage.transform import resize
from torch.utils.data import Dataset
import pykitti.utils as utils

from utils import map_fn

class DeepScenarioOdometry:
    """Class for parsing Deep Scenario (or any custom data) based on the pykitti
    implementation"""

    def __init__(self, base_path, sequence, scale_factor=1, **kwargs):
        """Set the path."""
        self.sequence = sequence
        self.sequence_path = os.path.join(base_path, 'sequences', sequence)
        self.pose_path = os.path.join(base_path, 'poses')
        self.scale_factor = scale_factor
        self.frames = kwargs.get('frames', None)

        # Default image file extension is 'png'
        self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        self._load_calib()
        self._load_timestamps()
        self._load_poses()

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.timestamps)

    @property
    def cam0(self):
        """Generator to read image files for cam0 (monochrome left)."""
        return utils.yield_images(self.cam0_files, mode='L')

    def get_cam0(self, idx):
        """Read image file for cam0 (monochrome left) at the specified index."""
        return utils.load_image(self.cam0_files[idx], mode='L')

    @property
    def cam1(self):
        """Generator to read image files for cam1 (monochrome right)."""
        return utils.yield_images(self.cam1_files, mode='L')

    def get_cam1(self, idx):
        """Read image file for cam1 (monochrome right) at the specified index."""
        return utils.load_image(self.cam1_files[idx], mode='L')

    @property
    def cam2(self):
        """Generator to read image files for cam2 (RGB left)."""
        return utils.yield_images(self.cam2_files, mode='RGB')

    def get_cam2(self, idx):
        """Read image file for cam2 (RGB left) at the specified index."""
        return utils.load_image(self.cam2_files[idx], mode='RGB')

    @property
    def cam3(self):
        """Generator to read image files for cam0 (RGB right)."""
        return utils.yield_images(self.cam3_files, mode='RGB')

    def get_cam3(self, idx):
        """Read image file for cam3 (RGB right) at the specified index."""
        return utils.load_image(self.cam3_files[idx], mode='RGB')

    @property
    def gray(self):
        """Generator to read monochrome stereo pairs from file.
        """
        return zip(self.cam0, self.cam1)

    def get_gray(self, idx):
        """Read monochrome stereo pair at the specified index."""
        return (self.get_cam0(idx), self.get_cam1(idx))

    @property
    def rgb(self):
        """Generator to read RGB stereo pairs from file.
        """
        return zip(self.cam2, self.cam3)

    def get_rgb(self, idx):
        """Read RGB stereo pair at the specified index."""
        return (self.get_cam2(idx), self.get_cam3(idx))


    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.cam0_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_0',
                         '*.{}'.format(self.imtype))))
        self.cam1_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_1',
                         '*.{}'.format(self.imtype))))
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_2',
                         '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_3',
                         '*.{}'.format(self.imtype))))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.cam0_files = utils.subselect_files(
                self.cam0_files, self.frames)
            self.cam1_files = utils.subselect_files(
                self.cam1_files, self.frames)
            self.cam2_files = utils.subselect_files(
                self.cam2_files, self.frames)
            self.cam3_files = utils.subselect_files(
                self.cam3_files, self.frames)
            self.velo_files = utils.subselect_files(
                self.velo_files, self.frames)

    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.sequence_path, 'calib.txt')
        filedata = utils.read_calib_file(calib_filepath)

        # Create 3x4 projection matrices, assuming all cameras are the same
        # You would just need to provide `P0: ... ... ...` in calib.txt
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = P_rect_00
        P_rect_20 = P_rect_00
        P_rect_30 = P_rect_00

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Assuming there is no difference between the cameras (monocular video)
        # # Compute the rectified extrinsics from cam0 to camN
        # T1 = np.eye(4)
        # T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        # T2 = np.eye(4)
        # T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        # T3 = np.eye(4)
        # T2[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Ignoring velodyne data, as we have monocular video
        # # Compute the velodyne to rectified camera coordinate transforms
        # data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        # data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        # data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        # data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        # data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics which are all the same
        # Hacky solution to not change getter methods from pykitti.odometry
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # No stereo baseline
        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # # between them
        # p_cam = np.array([0, 0, 0, 1])
        # p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        # p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        # p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        # p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        # data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        # data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def _load_timestamps(self):
        """Load timestamps from file."""
        timestamp_file = os.path.join(self.sequence_path, 'times.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                t = dt.timedelta(seconds=float(line))
                self.timestamps.append(t)

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.timestamps = [self.timestamps[i] for i in self.frames]

    def _load_poses(self):
        """Load ground truth poses (T_w_cam0) from file."""
        pose_file = os.path.join(self.pose_path, self.sequence + '.txt')

        # Read and parse the poses
        poses = []
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
                if self.frames is not None:
                    lines = [lines[i] for i in self.frames]

                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    # Multiply the translation components (last column) with the scale factor
                    T_w_cam0[:,-1] = T_w_cam0[:,-1] * self.scale_factor
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)

        except FileNotFoundError:
            print('Ground truth poses are not avaialble for sequence ' +
                  self.sequence + '.')

        self.poses = poses


class DeepScenarioDataset(Dataset):

    def __init__(self, dataset_dir, frame_count=2, sequences=None, depth_folder="image_depth",
                 target_image_size=(256, 512), max_length=None, dilation=1, offset_d=0, use_color=True, use_dso_poses=False,
                 use_color_augmentation=False, lidar_depth=False, dso_depth=True, annotated_lidar=False, return_stereo=False,
                 return_mvobj_mask=False, use_index_mask=(),
                 scale_factor=1):
        """
        Dataset implementation for Deep Scenario data, heavily inspired by the KITTI Odometry implementation.
        :param dataset_dir: Top level folder for KITTI Odometry (should contain folders sequences, poses, poses_dvso (if available)
        :param frame_count: Number of frames used per sample (excluding the keyframe). By default, the keyframe is in the middle of those frames. (Default=2)
        :param sequences: Which sequences to use. Should be tuple of strings, e.g. ("00", "01", ...)
        :param depth_folder: The folder within the sequence folder that contains the depth information (e.g. sequences/00/{depth_folder})
        :param target_image_size: Desired image size (correct processing of depths is only guaranteed for default value). (Default=(256, 512))
        :param max_length: Maximum length per sequence. Useful for splitting up sequences and testing. (Default=None)
        :param dilation: Spacing between the frames (Default 1)
        :param offset_d: Index offset for frames (offset_d=0 means keyframe is centered). (Default=0)
        :param use_color: Use color (camera 2) or greyscale (camera 0) images (default=True)
        :param use_dso_poses: Use poses provided by d(v)so instead of KITTI poses. Requires poses_dvso folder. (Default=True)
        :param use_color_augmentation: Use color jitter augmentation. The same transformation is applied to all frames in a sample. (Default=False)
        :param lidar_depth: Use depth information from (annotated) velodyne data. (Default=False)
        :param dso_depth: Use depth information from d(v)so. (Default=True)
        :param annotated_lidar: If lidar_depth=True, then this determines whether to use annotated or non-annotated depth maps. (Default=True)
        :param return_stereo: Return additional stereo frame. Only used during training. (Default=False)
        :param return_mvobj_mask: Return additional moving object mask. Only used during training. If return_mvobj_mask=2, then the mask is returned as target instead of the depthmap. (Default=False)
        :param use_index_mask: Use the listed index masks (if a sample is listed in one of the masks, it is not used). (Default=())
        :param scale_factor: Similar to the implementation of TUMMonoVODataset, this is useful when using monocular odometry as the trajectories might not have the true scale
        """
        self.dataset_dir = Path(dataset_dir)
        self.frame_count = frame_count
        self.sequences = sequences
        self.depth_folder = depth_folder
        self.lidar_depth = lidar_depth
        self.annotated_lidar = annotated_lidar
        self.dso_depth = dso_depth
        self.target_image_size = target_image_size
        self.use_index_mask = use_index_mask
        self.offset_d = offset_d
        self.scale_factor = scale_factor

        # Using custom Odometry class
        self._datasets = [DeepScenarioOdometry(dataset_dir, sequence, scale_factor) for sequence in self.sequences]
        # self._datasets = [pykitti.odometry(dataset_dir, sequence) for sequence in self.sequences]

        self._offset = (frame_count // 2) * dilation
        extra_frames = frame_count * dilation
        if self.annotated_lidar and self.lidar_depth:
            extra_frames = max(extra_frames, 10)
            self._offset = max(self._offset, 5)
        self._dataset_sizes = [
            len((dataset.cam0_files if not use_color else dataset.cam2_files)) - (extra_frames if self.use_index_mask is None else 0) for dataset in
            self._datasets]
        if self.use_index_mask is not None:
            index_masks = []
            for sequence_length, sequence in zip(self._dataset_sizes, self.sequences):
                index_mask = {i:True for i in range(sequence_length)}
                for index_mask_name in self.use_index_mask:
                    with open(self.dataset_dir / "sequences" / sequence / (index_mask_name + ".json")) as f:
                        m = json.load(f)
                        for k in list(index_mask.keys()):
                            if not str(k) in m or not m[str(k)]:
                                del index_mask[k]
                index_masks.append(index_mask)
            self._indices = [
                list(sorted([int(k) for k in sorted(index_mask.keys()) if index_mask[k] and int(k) >= self._offset and int(k) < dataset_size + self._offset - extra_frames]))
                for index_mask, dataset_size in zip(index_masks, self._dataset_sizes)
            ]
            self._dataset_sizes = [len(indices) for indices in self._indices]
        if max_length is not None:
            self._dataset_sizes = [min(s, max_length) for s in self._dataset_sizes]
        self.length = sum(self._dataset_sizes)

        intrinsics_box = [self.compute_target_intrinsics(dataset, target_image_size, use_color) for dataset in
                          self._datasets]
        self._crop_boxes = [b for _, b in intrinsics_box]
        # Removing ground truth depth information
        # if self.dso_depth:
        #     self.dso_depth_parameters = [self.get_dso_depth_parameters(dataset) for dataset in self._datasets]
        # elif not self.lidar_depth:
        #     self._depth_crop_boxes = [
        #         self.compute_depth_crop(self.dataset_dir / "sequences" / s / depth_folder) for s in
        #         self.sequences]
        self._intrinsics = [format_intrinsics(i, self.target_image_size) for i, _ in intrinsics_box]
        self.dilation = dilation
        self.use_color = use_color
        self.use_dso_poses = use_dso_poses
        self.use_color_augmentation = use_color_augmentation
        if self.use_dso_poses:
            for dataset in self._datasets:
                dataset.pose_path = self.dataset_dir / "poses_dvso"
                dataset._load_poses()
        if self.use_color_augmentation:
            self.color_transform = ColorJitterMulti(brightness=.2, contrast=.2, saturation=.2, hue=.1)
        self.return_stereo = return_stereo
        if self.return_stereo:
            self._stereo_transform = []
            for d in self._datasets:
                st = torch.eye(4, dtype=torch.float32)
                st[0, 3] = d.calib.b_rgb if self.use_color else d.calib.b_gray
                self._stereo_transform.append(st)

        self.return_mvobj_mask = return_mvobj_mask

    def get_dataset_index(self, index: int):
        for dataset_index, dataset_size in enumerate(self._dataset_sizes):
            if index >= dataset_size:
                index = index - dataset_size
            else:
                return dataset_index, index
        return None, None

    def preprocess_image(self, img: Image.Image, crop_box=None):
        if crop_box:
            img = img.crop(crop_box)
        if self.target_image_size:
            img = img.resize((self.target_image_size[1], self.target_image_size[0]), resample=Image.BILINEAR)
        if self.use_color_augmentation:
            img = self.color_transform(img)
        image_tensor = torch.tensor(np.array(img).astype(np.float32))
        image_tensor = image_tensor / 255 - .5
        if not self.use_color:
            image_tensor = torch.stack((image_tensor, image_tensor, image_tensor))
        else:
            image_tensor = image_tensor.permute(2, 0, 1)
        del img
        return image_tensor

    def preprocess_depth(self, depth: np.ndarray, crop_box=None):
        """
        This function does cropping and resizing if it's specified by the
        config and returns the inverse depth.
        """
        if crop_box:
            if crop_box[1] >= 0 and crop_box[3] <= depth.shape[0]:
                depth = depth[int(crop_box[1]):int(crop_box[3]), :]
            else:
                depth_ = np.ones((crop_box[3] - crop_box[1], depth.shape[1]))
                depth_[-crop_box[1]:-crop_box[1]+depth.shape[0], :] = depth
                depth = depth_
            if crop_box[0] >= 0 and crop_box[2] <= depth.shape[1]:
                depth = depth[:, int(crop_box[0]):int(crop_box[2])]
            else:
                depth_ = np.ones((depth.shape[0], crop_box[2] - crop_box[0]))
                depth_[:, -crop_box[0]:-crop_box[0]+depth.shape[1]] = depth
                depth = depth_
        if self.target_image_size:
            depth = resize(depth, self.target_image_size, order=0)
        return torch.tensor(1 / depth)

    def preprocess_depth_dso(self, depth: Image.Image, dso_depth_parameters, crop_box=None):
        h, w, f_x = dso_depth_parameters
        depth = np.array(depth, dtype=float)
        indices = np.array(np.nonzero(depth), dtype=float)
        indices[0] = np.clip(indices[0] / depth.shape[0] * h, 0, h-1)
        indices[1] = np.clip(indices[1] / depth.shape[1] * w, 0, w-1)

        depth = depth[depth > 0]
        depth = (w * depth / (0.54 * f_x * 65535))

        data = np.concatenate([indices, np.expand_dims(depth, axis=0)], axis=0)

        if crop_box:
            data = data[:, (crop_box[1] <= data[0, :]) & (data[0, :] < crop_box[3]) & (crop_box[0] <= data[1, :]) & (data[1, :] < crop_box[2])]
            data[0, :] -= crop_box[1]
            data[1, :] -= crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            crop_width = crop_box[2] - crop_box[0]
        else:
            crop_height = h
            crop_width = w

        data[0] = np.clip(data[0] / crop_height * self.target_image_size[0], 0, self.target_image_size[0]-1)
        data[1] = np.clip(data[1] / crop_width * self.target_image_size[1], 0, self.target_image_size[1]-1)

        depth = np.zeros(self.target_image_size)
        depth[np.around(data[0]).astype(int), np.around(data[1]).astype(int)] = data[2]

        return torch.tensor(depth, dtype=torch.float32)

    def preprocess_depth_annotated_lidar(self, depth: Image.Image, crop_box=None):
        depth = np.array(depth, dtype=float) # had to change from np.float to work with current environment -altay
        h, w = depth.shape
        indices = np.array(np.nonzero(depth), dtype=float)

        depth = depth[depth > 0]
        depth = 256.0 / depth

        data = np.concatenate([indices, np.expand_dims(depth, axis=0)], axis=0)

        if crop_box:
            data = data[:, (crop_box[1] <= data[0, :]) & (data[0, :] < crop_box[3]) & (crop_box[0] <= data[1, :]) & (
                        data[1, :] < crop_box[2])]
            data[0, :] -= crop_box[1]
            data[1, :] -= crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            crop_width = crop_box[2] - crop_box[0]
        else:
            crop_height = h
            crop_width = w

        data[0] = np.clip(data[0] / crop_height * self.target_image_size[0], 0, self.target_image_size[0] - 1)
        data[1] = np.clip(data[1] / crop_width * self.target_image_size[1], 0, self.target_image_size[1] - 1)

        depth = np.zeros(self.target_image_size)
        depth[np.around(data[0]).astype(int), np.around(data[1]).astype(int)] = data[2]

        return torch.tensor(depth, dtype=torch.float32)

    def __getitem__(self, index: int):
        dataset_index, index = self.get_dataset_index(index)
        if dataset_index is None:
            raise IndexError()

        if self.use_index_mask is not None:
            index = self._indices[dataset_index][index] - self._offset

        sequence_folder = self.dataset_dir / "sequences" / self.sequences[dataset_index]
        # Assuming we have no ground truth depth
        # depth_folder = sequence_folder / self.depth_folder

        if self.use_color_augmentation:
            self.color_transform.fix_transform()

        dataset = self._datasets[dataset_index]
        keyframe_intrinsics = self._intrinsics[dataset_index]
        # Removing functionality related to ground truth depth
        # if not (self.lidar_depth or self.dso_depth):
        #     keyframe_depth = self.preprocess_depth(np.load(depth_folder / f"{(index + self._offset):06d}.npy"), self._depth_crop_boxes[dataset_index]).type(torch.float32).unsqueeze(0)
        # else:
        #     if self.lidar_depth:
        #         if not self.annotated_lidar:
        #             lidar_depth = 1 / torch.tensor(sparse.load_npz(depth_folder / f"{(index + self._offset):06d}.npz").todense()).type(torch.float32).unsqueeze(0)
        #             lidar_depth[torch.isinf(lidar_depth)] = 0
        #             keyframe_depth = lidar_depth
        #         else:
        #             keyframe_depth = self.preprocess_depth_annotated_lidar(Image.open(depth_folder / f"{(index + self._offset):06d}.png"), self._crop_boxes[dataset_index]).unsqueeze(0)
        #     else:
        #         keyframe_depth = torch.zeros(1, self.target_image_size[0], self.target_image_size[1], dtype=torch.float32)

        #     if self.dso_depth:
        #         dso_depth = self.preprocess_depth_dso(Image.open(depth_folder / f"{(index + self._offset):06d}.png"), self.dso_depth_parameters[dataset_index], self._crop_boxes[dataset_index]).unsqueeze(0)
        #         mask = dso_depth == 0
        #         dso_depth[mask] = keyframe_depth[mask]
        #         keyframe_depth = dso_depth

        keyframe = self.preprocess_image(
            (dataset.get_cam0 if not self.use_color else dataset.get_cam2)(index + self._offset),
            self._crop_boxes[dataset_index])
        keyframe_pose = torch.tensor(dataset.poses[index + self._offset], dtype=torch.float32)

        frames = [self.preprocess_image((dataset.get_cam0 if not self.use_color else dataset.get_cam2)(index + self._offset + i + self.offset_d),
                                        self._crop_boxes[dataset_index]) for i in
                  range(-(self.frame_count // 2) * self.dilation, ((self.frame_count + 1) // 2) * self.dilation + 1, self.dilation) if i != 0]
        intrinsics = [self._intrinsics[dataset_index] for _ in range(self.frame_count)]
        try:
            poses = [torch.tensor(dataset.poses[index + self._offset + i + self.offset_d], dtype=torch.float32) for i in
                 range(-(self.frame_count // 2) * self.dilation, ((self.frame_count + 1) // 2) * self.dilation + 1, self.dilation) if i != 0]
        except Exception as e:
            print("Uh oh, something went wrong!")

        data = {
            "keyframe": keyframe,
            "keyframe_pose": keyframe_pose,
            "keyframe_intrinsics": keyframe_intrinsics,
            "frames": frames,
            "poses": poses,
            "intrinsics": intrinsics,
            "sequence": torch.tensor([int(self.sequences[dataset_index])], dtype=torch.int32),
            "image_id": torch.tensor([int(index + self._offset)], dtype=torch.int32)
        }

        if self.return_stereo:
            stereoframe = self.preprocess_image(
                (dataset.get_cam1 if not self.use_color else dataset.get_cam3)(index + self._offset),
                self._crop_boxes[dataset_index])
            stereoframe_pose = torch.tensor(dataset.poses[index + self._offset], dtype=torch.float32) @ self._stereo_transform[dataset_index]
            data["stereoframe"] = stereoframe
            data["stereoframe_pose"] = stereoframe_pose
            data["stereoframe_intrinsics"] = keyframe_intrinsics

        if self.return_mvobj_mask > 0:
            mask = torch.tensor(np.load(sequence_folder / "mvobj_mask" / f"{index + self._offset:06d}.npy"), dtype=torch.float32).unsqueeze(0)
            data["mvobj_mask"] = mask
            if self.return_mvobj_mask == 2:
                return data, mask
        # Removing keyframe_depth as a return value, we are assuming that we do
        # not have ground truth depth values now.
        return data, 0

    def __len__(self) -> int:
        return self.length


    def compute_target_intrinsics(self, dataset, target_image_size, use_color):
        # Because of cropping and resizing of the frames, we need to recompute the intrinsics
        P_cam = dataset.calib.P_rect_00 if not use_color else dataset.calib.P_rect_20
        orig_size = tuple(reversed((dataset.cam0 if not use_color else dataset.cam2).__next__().size))

        # r is the ratio of Height / Width
        r_orig = orig_size[0] / orig_size[1]
        r_target = target_image_size[0] / target_image_size[1]

        if r_orig >= r_target:
            new_height = r_target * orig_size[1] # width stays the same, new height is computed
            box = (0, (orig_size[0] - new_height) // 2, orig_size[1], orig_size[0] - (orig_size[0] - new_height) // 2)

            c_x = P_cam[0, 2] / orig_size[1] # width doesn't change so c_x stays the same, it is only normalized
            c_y = (P_cam[1, 2] - (orig_size[0] - new_height) / 2) / new_height

            rescale = orig_size[1] / target_image_size[1]

        else:
            new_width = orig_size[0] / r_target # height stays the same, new width is computed
            box = ((orig_size[1] - new_width) // 2, 0, orig_size[1] - (orig_size[1] - new_width) // 2, orig_size[0])

            c_x = (P_cam[0, 2] - (orig_size[1] - new_width) / 2) / new_width
            c_y = P_cam[1, 2] / orig_size[0]

            rescale = orig_size[0] / target_image_size[0]

        f_x = P_cam[0, 0] / target_image_size[1] / rescale # we normalize the focal lenghts and divide by the rescaling factor
        f_y = P_cam[1, 1] / target_image_size[0] / rescale

        intrinsics = (f_x, f_y, c_x, c_y)

        return intrinsics, box

    def get_dso_depth_parameters(self, dataset):
        # Info required to process d(v)so depths
        P_cam =  dataset.calib.P_rect_20
        orig_size = tuple(reversed(dataset.cam2.__next__().size))
        return orig_size[0], orig_size[1], P_cam[0, 0]

    def get_index(self, sequence, index):
        for i in range(len(self.sequences)):
            if int(self.sequences[i]) != sequence:
                index += self._dataset_sizes[i]
            else:
                break
        return index


def format_intrinsics(intrinsics, target_image_size):
    intrinsics_mat = torch.zeros(4, 4)
    intrinsics_mat[0, 0] = intrinsics[0] * target_image_size[1] #f_x
    intrinsics_mat[1, 1] = intrinsics[1] * target_image_size[0] #f_y
    intrinsics_mat[0, 2] = intrinsics[2] * target_image_size[1] #c_x
    intrinsics_mat[1, 2] = intrinsics[3] * target_image_size[0] #c_y
    intrinsics_mat[2, 2] = 1
    intrinsics_mat[3, 3] = 1
    return intrinsics_mat


class ColorJitterMulti(torchvision.transforms.ColorJitter):
    def fix_transform(self):
        self.transform = self.get_params(self.brightness, self.contrast,
                                         self.saturation, self.hue)

    def __call__(self, x):
        return map_fn(x, self.transform)
