# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from PIL import Image
import torch.utils.data


def load_xbox_poses(path):
    """Load KRT file containing intrinsic and extrinsic parameters."""
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                "intrin": np.array(intrin),
                "extrin": np.array(extrin)
            }

    return cameras


# image size, change if needed
img_h, img_w = 256, 256


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        camerafilter, framelist, keyfilter,
        fixedcameras=[],
        imagemean=0., imagestd=1.,
        subsampletype=None, subsamplesize=0
    ):
        '''
            framelist:                      int list of frame ids (same for each pose).
            camerafilter:                   lambda function that select a list from all camera ids list.
            keyfilter:                      list of string that on/off some data loading.
            fixedcameras:                   list of string that corrsponded to certain fixed pose. Used in training.
            imagemean / imagestd:           a float, mean/std applied on image values before input network.
            worldscale:                     a float scale applied on object transformation matrix.
            subsampletype:                  a string, pixel sampling mode (patch sample / random sample)
            subsamplesize:                  a int, patchsize in patch sample mode / number of sample (before square) in random sample mode.
        '''

        # load camera intrinsic and extrinsic for each pose
        posepath = "experiments/xbox_avatar/data/KRT"
        xbox_poses = load_xbox_poses(posepath)

        # load camera lists and filter w.r.t user requirements
        self.allcameras = sorted(list(xbox_poses.keys()))
        self.cameras = list(filter(camerafilter, self.allcameras))
        self.framelist = framelist
        self.framecamlist = [
            (x, cam)
            for x in self.framelist
            for cam in (self.cameras if len(self.cameras) > 0 else [None])
        ]

        # get options
        self.keyfilter = keyfilter
        self.fixedcameras = fixedcameras
        self.imagemean = imagemean
        self.imagestd = imagestd
        self.subsampletype = subsampletype
        self.subsamplesize = subsamplesize

        # Preprocess cam pose matrices
        # campos: (3, )
        # camrot: (3, 3)
        # focal: (2, )
        # princpt: (2, )
        self.campos, self.camrot, self.focal, self.princpt = {}, {}, {}, {}
        for cam in self.cameras:
            self.campos[cam] = (-np.dot(xbox_poses[cam]['extrin'][:3, :3].T, xbox_poses[cam]['extrin'][:3, 3])).astype(np.float32)
            self.camrot[cam] = (xbox_poses[cam]['extrin'][:3, :3]).astype(np.float32)
            self.focal[cam] = (np.diag(xbox_poses[cam]['intrin'][:2, :2])).astype(np.float32)
            self.princpt[cam] = (xbox_poses[cam]['intrin'][:2, 2]).astype(np.float32)

        # transformation of world space - object space
        # transf: (3, 4)
        self.transf = np.eye(4)[0:3, :]

        # load background images for each camera
        # for xbox avatar it is simply a black background
        # fix image size here, change if needed.
        if "bg" in self.keyfilter:
            self.bg = {}
            for i, cam in enumerate(self.cameras):
                self.bg[cam] = np.zeros([3, img_h, img_w], dtype=np.uint8)

    def get_allcameras(self):
        return self.allcameras

    def get_krt(self):
        return {k: {
                "pos": self.campos[k],
                "rot": self.camrot[k],
                "focal": self.focal[k],
                "princpt": self.princpt[k],
                "size": np.array([img_w, img_h])}
                for k in self.cameras}

    def known_background(self):
        return "bg" in self.keyfilter

    def get_background(self, bg):
        if "bg" in self.keyfilter:
            for i, cam in enumerate(self.cameras):
                if cam in self.bg:
                    bg[cam].data[:] = torch.from_numpy(self.bg[cam]).to("cuda")

    def __len__(self):
        return len(self.framecamlist)

    def __getitem__(self, idx):
        # frame and cam id
        frame_id, cam_id = self.framecamlist[idx]

        result = {}

        # tag: whether this sample is valid?
        validinput = True

        # fixed camera images (used in training)
        if "fixedcamimage" in self.keyfilter:
            ninput = len(self.fixedcameras)
            fixedcamimage = np.zeros((3 * ninput, img_h, img_w), dtype=np.float32)
            for i in range(ninput):
                imagepath = "experiments/dryice1/data/cam{}/image{:04}.jpg".format(self.fixedcameras[i], int(frame_id))
                image = np.asarray(Image.open(imagepath), dtype=np.uint8)[::2, ::2, :].transpose((2, 0, 1)).astype(np.float32)
                if np.sum(image) == 0:
                    validinput = False
                fixedcamimage[i * 3: (i + 1) * 3, :, :] = image

            # apply mean / std
            fixedcamimage[:] -= self.imagemean
            fixedcamimage[:] /= self.imagestd

            result["fixedcamimage"] = fixedcamimage

        result["validinput"] = np.float32(1.0 if validinput else 0.0)

        # current frame/camera view
        if cam_id is not None:
            # camera poses
            if "camera" in self.keyfilter:
                # camera data
                result["camrot"] = np.dot(self.transf[:3, :3].T, self.camrot[cam_id].T).T
                result["campos"] = np.dot(self.transf[:3, :3].T, self.campos[cam_id] - self.transf[:3, 3])
                result["focal"] = self.focal[cam_id]
                result["princpt"] = self.princpt[cam_id]
                result["camindex"] = self.allcameras.index(cam_id)

            # image
            if "image" in self.keyfilter:
                # image
                imagepath = "experiments/dryice1/data/cam{}/image{:04}.jpg".format(cam_id, int(frame_id))
                image = np.asarray(Image.open(imagepath), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)
                height, width = image.shape[1:3]
                valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)
                result["image"] = image
                result["imagevalid"] = valid

            # Pixel coords for shooting rays
            if "pixelcoords" in self.keyfilter:
                if self.subsampletype == "patch":
                    indx = np.random.randint(0, width - self.subsamplesize + 1)
                    indy = np.random.randint(0, height - self.subsamplesize + 1)

                    px, py = np.meshgrid(
                        np.arange(indx, indx + self.subsamplesize).astype(np.float32),
                        np.arange(indy, indy + self.subsamplesize).astype(np.float32)
                    )
                elif self.subsampletype == "random":
                    px = np.random.randint(0, width, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                    py = np.random.randint(0, height, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                elif self.subsampletype == "random2":
                    px = np.random.uniform(0, width - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                    py = np.random.uniform(0, height - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                else:
                    px, py = np.meshgrid(np.arange(width).astype(np.float32), np.arange(height).astype(np.float32))

                # (H, W, 2)
                result["pixelcoords"] = np.stack((px, py), axis=-1)

        return result



# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, camerafilter, framelist, keyfilter,
#             fixedcameras=[], fixedcammean=0., fixedcamstd=1.,
#             imagemean=0., imagestd=1.,
#             worldscale=1., subsampletype=None, subsamplesize=0):
#         krtpath = "experiments/xbox_avatar/data/KRT"
#         krt = load_krt(krtpath)

#         # get options
#         self.allcameras = sorted(list(krt.keys()))
#         self.cameras = list(filter(camerafilter, self.allcameras))
#         self.framelist = framelist
#         self.framecamlist = [(x, cam)
#                 for x in self.framelist
#                 for cam in (self.cameras if len(self.cameras) > 0 else [None])]

#         self.keyfilter = keyfilter
#         self.fixedcameras = fixedcameras
#         self.fixedcammean = fixedcammean
#         self.fixedcamstd = fixedcamstd
#         self.imagemean = imagemean
#         self.imagestd = imagestd
#         self.subsampletype = subsampletype
#         self.subsamplesize = subsamplesize

#         # compute camera positions
#         self.campos, self.camrot, self.focal, self.princpt = {}, {}, {}, {}
#         for cam in self.cameras:
#             self.campos[cam] = (-np.dot(krt[cam]['extrin'][:3, :3].T, krt[cam]['extrin'][:3, 3])).astype(np.float32)
#             self.camrot[cam] = (krt[cam]['extrin'][:3, :3]).astype(np.float32)
#             self.focal[cam] = (np.diag(krt[cam]['intrin'][:2, :2]) / 4.).astype(np.float32)
#             self.princpt[cam] = (krt[cam]['intrin'][:2, 2] / 4.).astype(np.float32)

#         # transformation that places the center of the object at the origin
#         transfpath = "experiments/dryice1/data/pose.txt"
#         self.transf = np.genfromtxt(transfpath, dtype=np.float32, skip_footer=2)
#         self.transf[:3, :3] *= worldscale

#         # load background images for each camera
#         if "bg" in self.keyfilter:
#             self.bg = {}
#             for i, cam in enumerate(self.cameras):
#                 try:
#                     imagepath = "experiments/dryice1/data/cam{}/bg.jpg".format(cam)
#                     image = np.asarray(Image.open(imagepath), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)
#                     self.bg[cam] = image
#                 except:
#                     pass

#     def get_allcameras(self):
#         return self.allcameras

#     def get_krt(self):
#         return {k: {
#                 "pos": self.campos[k],
#                 "rot": self.camrot[k],
#                 "focal": self.focal[k],
#                 "princpt": self.princpt[k],
#                 "size": np.array([667, 1024])}
#                 for k in self.cameras}

#     def known_background(self):
#         return "bg" in self.keyfilter

#     def get_background(self, bg):
#         if "bg" in self.keyfilter:
#             for i, cam in enumerate(self.cameras):
#                 if cam in self.bg:
#                     bg[cam].data[:] = torch.from_numpy(self.bg[cam]).to("cuda")

#     def __len__(self):
#         return len(self.framecamlist)

#     def __getitem__(self, idx):
#         frame, cam = self.framecamlist[idx]

#         result = {}

#         validinput = True

#         # fixed camera images
#         if "fixedcamimage" in self.keyfilter:
#             ninput = len(self.fixedcameras)

#             fixedcamimage = np.zeros((3 * ninput, 512, 334), dtype=np.float32)
#             for i in range(ninput):
#                 imagepath = (
#                         "experiments/dryice1/data/cam{}/image{:04}.jpg"
#                         .format(self.fixedcameras[i], int(frame)))
#                 image = np.asarray(Image.open(imagepath), dtype=np.uint8)[::2, ::2, :].transpose((2, 0, 1)).astype(np.float32)
#                 if np.sum(image) == 0:
#                     validinput = False
#                 fixedcamimage[i*3:(i+1)*3, :, :] = image
#             fixedcamimage[:] -= self.imagemean
#             fixedcamimage[:] /= self.imagestd
#             result["fixedcamimage"] = fixedcamimage

#         result["validinput"] = np.float32(1.0 if validinput else 0.0)

#         # image data
#         if cam is not None:
#             if "camera" in self.keyfilter:
#                 # camera data
#                 result["camrot"] = np.dot(self.transf[:3, :3].T, self.camrot[cam].T).T
#                 result["campos"] = np.dot(self.transf[:3, :3].T, self.campos[cam] - self.transf[:3, 3])
#                 result["focal"] = self.focal[cam]
#                 result["princpt"] = self.princpt[cam]
#                 result["camindex"] = self.allcameras.index(cam)

#             if "image" in self.keyfilter:
#                 # image
#                 imagepath = (
#                         "experiments/dryice1/data/cam{}/image{:04}.jpg"
#                         .format(cam, int(frame)))
#                 image = np.asarray(Image.open(imagepath), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)
#                 height, width = image.shape[1:3]
#                 valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)
#                 result["image"] = image
#                 result["imagevalid"] = valid

#             if "pixelcoords" in self.keyfilter:
#                 if self.subsampletype == "patch":
#                     indx = np.random.randint(0, width - self.subsamplesize + 1)
#                     indy = np.random.randint(0, height - self.subsamplesize + 1)

#                     px, py = np.meshgrid(
#                             np.arange(indx, indx + self.subsamplesize).astype(np.float32),
#                             np.arange(indy, indy + self.subsamplesize).astype(np.float32))
#                 elif self.subsampletype == "random":
#                     px = np.random.randint(0, width, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
#                     py = np.random.randint(0, height, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
#                 elif self.subsampletype == "random2":
#                     px = np.random.uniform(0, width - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
#                     py = np.random.uniform(0, height - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
#                 else:
#                     px, py = np.meshgrid(np.arange(width).astype(np.float32), np.arange(height).astype(np.float32))

#                 result["pixelcoords"] = np.stack((px, py), axis=-1)

#         return result
