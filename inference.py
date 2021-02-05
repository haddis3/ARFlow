import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms

from utils.flow_utils import flow_to_image, resize_flow
from utils.warp_utils import flow_warp, get_occu_mask_backward
from utils.torch_utils import restore_model
from models.pwclite import PWCLite
from skimage import img_as_ubyte


def get_occu_mask_bidirection(flow12, flow21, scale=0.1, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/Sintel/pwclite_ar.tar')
    # parser.add_argument('-m', '--model', default='checkpoints/Sintel/111320_ckpt.pth.tar')
    parser.add_argument('-s', '--test_shape', default=[512, 512], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    parser.add_argument('-list', '--test_list', nargs='+',
                        default='data/flow_dataset/ceshi/left')

    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    ts = TestHelper(cfg)

    for name in os.listdir(args.test_list):

        print('processing {}'.format(name))
        path = []
        right = os.path.join('data/flow_dataset/ceshi/right', name)
        path.append(os.path.join(args.test_list, name))
        path.append(right)
        imgs = [imageio.imread(img).astype(np.float32) for img in path]
        h, w = imgs[0].shape[:2]

        res_dict = ts.run(imgs)
        flow_12 = res_dict['flows_fw'][0]
        flow_21 = res_dict['flows_bw'][0]

        flow_12 = resize_flow(flow_12, (h, w))  # [1, 2, H, W]
        flow_21 = resize_flow(flow_21, (h, w))  # [1, 2, H, W]
        occu_mask1 = 1 - get_occu_mask_bidirection(flow_12, flow_21)  # [1, 1, H, W]
        occu_mask2 = 1 - get_occu_mask_bidirection(flow_21, flow_12)
        back_occu_mask1 = get_occu_mask_backward(flow_21)
        back_occu_mask2 = get_occu_mask_backward(flow_21)

        warped_image_12 = flow_warp(torch.from_numpy(np.transpose(imgs[1], [2, 0, 1])).unsqueeze(0).cuda(), flow_12, pad='border')
        warped_image_21 = flow_warp(torch.from_numpy(np.transpose(imgs[0], [2, 0, 1])).unsqueeze(0).cuda(), flow_21, pad='border')
        np_warped_image12 = warped_image_12[0].detach().cpu().numpy().transpose([1, 2, 0])
        np_warped_image21 = warped_image_21[0].detach().cpu().numpy().transpose([1, 2, 0])

        np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
        np_flow_21 = flow_21[0].detach().cpu().numpy().transpose([1, 2, 0])
        # vx = np_flow_12[:, :, 0]
        # vy = np_flow_12[:, :, 1]
        # f = open(os.path.join(r'G:\ARFlow-master\data\flow_dataset\ceshi_tmp', name + '_vx.bin'), 'wb')
        # vx.astype(np.float32).tofile(f)
        # f.close()
        #
        # f = open(os.path.join(r'G:\ARFlow-master\data\flow_dataset\ceshi_tmp', name + '_vy.bin'), 'wb')
        # vy.astype(np.float32).tofile(f)
        # f.close()
        occu_mask_12 = occu_mask1[0].detach().cpu().numpy().transpose([1, 2, 0])
        occu_mask_21 = occu_mask2[0].detach().cpu().numpy().transpose([1, 2, 0])
        back_occu_mask_1 = back_occu_mask1[0].detach().cpu().numpy().transpose([1, 2, 0])
        back_occu_mask_2 = back_occu_mask2[0].detach().cpu().numpy().transpose([1, 2, 0])

        vis_flow_12 = flow_to_image(np_flow_12)
        vis_flow_21 = flow_to_image(np_flow_21)

        row1 = np.concatenate([np.uint8(imgs[0]), np.uint8(imgs[1]), np.uint8(vis_flow_12), np.uint8(vis_flow_21)], axis=1)
        # row2 = np.concatenate([np.uint8(np_warped_image12), np.uint8(np_warped_image21), 255*np.tile(occu_mask_12, (1, 1, 3)),
        #                        255*np.tile(occu_mask_21, (1, 1, 3))], axis=1)
        row2 = np.concatenate([np.uint8(np_warped_image12), np.uint8(np_warped_image21), 255 * np.tile(back_occu_mask_1, (1, 1, 3)),
             255 * np.tile(back_occu_mask_2, (1, 1, 3))], axis=1)
        image_matrix = np.concatenate([row1, row2], axis=0)

        imageio.imsave(name, image_matrix)
        # imageio.imsave('warped_image.jpg', warped_image)

        # fig = plt.figure()
        # plt.imshow(image_matrix)
        # plt.show()
