#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :model.py
#@Date        :2022/04/09 16:48:16
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import torch.nn as nn
import numpy as np
import time
from torch.nn import functional as F
from config import cfg
from networks.backbones.resnet import ResNetBackbone
from networks.backbones.transformer import ILAVideoTransformer, FactorizedVideoTransformer, VideoTransformer
from networks.necks.unet import UNet
from networks.heads.sdf_head import SDFHead
from networks.heads.mano_head import ManoHead
from networks.heads.fc_head import FCHead
from mano.mano_preds import get_mano_preds
from mano.manolayer import ManoLayer
from mano.inverse_kinematics import ik_solver_mano
from utils.pose_utils import soft_argmax, decode_volume
from utils.sdf_utils import kinematic_embedding, pixel_align


class model(nn.Module):
    def __init__(self, cfg, backbone, neck, hand_sdf_head, obj_sdf_head, mano_head, volume_head, rot_head, feat_transformer):
        super(model, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.dim_backbone_feat = 2048 if self.cfg.backbone == 'resnet_50' else 512
        self.neck = neck
        self.hand_sdf_head = hand_sdf_head
        self.obj_sdf_head = obj_sdf_head
        self.mano_head = mano_head
        self.volume_head = volume_head
        self.rot_head = rot_head
        self.feat_transformer = feat_transformer
        self.backbone_2_sdf = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        if cfg.with_add_feats:
            self.sdf_encoder = nn.Linear(512 + 4, self.cfg.sdf_latent)
        else:
            self.sdf_encoder = nn.Linear(512, self.cfg.sdf_latent)

        if self.mano_head is not None:
            self.backbone_2_mano = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        
        if self.rot_head is not None:
            self.backbone_2_rot = nn.Sequential(
                nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True))
        
        self.loss_l1 = torch.nn.L1Loss(reduction='sum')
        self.loss_l2 = torch.nn.MSELoss()
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    #cond_input may include camera intrinsics or hand wrist position
    def forward(self, inputs, targets=None, metas=None, mode='train'):
        if mode == 'train':
            input_img = inputs['img']
            num_frames = len(input_img)
            
            if self.cfg.hand_branch and self.cfg.obj_branch:
                sdf_data_frames = []
                cls_data_frames = []
                for i in range(num_frames):
                    sdf_data_frames.append(torch.cat([targets['hand_sdf'][i], targets['obj_sdf'][i]], 1))
                    cls_data_frames.append(torch.cat([targets['hand_labels'][i], targets['obj_labels'][i]], 1))
                if metas['epoch'] < self.cfg.sdf_add_epoch:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'][0].size()[:2]), torch.zeros(targets['obj_sdf'][0].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.zeros(targets['hand_sdf'][0].size()[:2]), torch.ones(targets['obj_sdf'][0].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                else:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'][0].size()[:2]), torch.ones(targets['obj_sdf'][0].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.ones(targets['hand_sdf'][0].size()[:2]), torch.ones(targets['obj_sdf'][0].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
            elif self.cfg.hand_branch:
                sdf_data_frames = targets['hand_sdf']
                cls_data_frames = targets['hand_labels']
                mask_hand = torch.ones(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1).cuda()
            elif self.cfg.obj_branch:
                sdf_data_frames = targets['obj_sdf']
                cls_data_frames = targets['obj_labels']
                mask_obj = torch.ones(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1).cuda()

            xyz_points_frames, sdf_gt_hand_frames, sdf_gt_obj_frames = [], [], []
            for i in range(num_frames):
                sdf_data_frames[i] = sdf_data_frames[i].reshape(self.cfg.train_batch_size * self.cfg.num_sample_points, -1)
                xyz_points_frames.append(sdf_data_frames[i][:, 0:-2])
                sdf_gt_hand = sdf_data_frames[i][:, -2].unsqueeze(1)
                sdf_gt_obj = sdf_data_frames[i][:, -1].unsqueeze(1)
                cls_data_frames[i] = cls_data_frames[i].to(torch.long).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points)

                if self.cfg.hand_branch:
                    sdf_gt_hand = torch.clamp(sdf_gt_hand, -self.cfg.clamp_dist, self.cfg.clamp_dist)
                    sdf_gt_hand_frames.append(sdf_gt_hand)
                if self.cfg.obj_branch:
                    sdf_gt_obj = torch.clamp(sdf_gt_obj, -self.cfg.clamp_dist, self.cfg.clamp_dist)
                    sdf_gt_obj_frames.append(sdf_gt_obj)

            sdf_feat_frames, obj_pose_results_frames, hand_pose_results_frames, volume_joint_preds_frames = [], [], [], []
            for i in range(num_frames):
                # go through backbone
                backbone_feat = self.backbone(input_img[i])

                volume_results = {}
                if self.volume_head is not None:
                    hm_feat = self.neck(backbone_feat)
                    hm_pred = self.volume_head(hm_feat)
                    hm_pred, hm_conf = soft_argmax(cfg, hm_pred, num_joints=1)
                    volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'][i], metas['cam_intr'][i])
                    volume_joint_preds_frames.append(volume_joint_preds)
                    volume_results['joints'] = volume_joint_preds
                else:
                    volume_results = None

                # estimate the hand pose
                if self.mano_head is not None:
                    mano_feat = self.backbone_2_mano(backbone_feat)
                    mano_feat = mano_feat.mean(3).mean(2)
                    hand_pose_results = self.mano_head(mano_feat)
                    hand_pose_results = get_mano_preds(hand_pose_results, self.cfg, metas['cam_intr'][i], metas['hand_center_3d'][i])
                    hand_pose_results_frames.append(hand_pose_results)
                else:
                    hand_pose_results = None
                
                # estimate the object rotation
                if self.rot_head is not None:
                    rot_feat = self.backbone_2_rot(backbone_feat)
                    rot_feat = rot_feat.mean(3).mean(2)
                    obj_rot = self.rot_head(rot_feat)
                else:
                    obj_rot = None
        
                # convert the object pose to the hand-relative coordinate system
                if cfg.obj_trans or cfg.obj_rot:
                    obj_pose_results = {}
                    obj_transform = torch.zeros(self.cfg.train_batch_size, 4, 4).to(input_img[i].device)
                    obj_transform[:, :3, 3] = volume_joint_preds[:, 0, :] - metas['hand_center_3d'][i]
                    obj_transform[:, 3, 3] = 1
                    if cfg.obj_rot:
                        obj_transform[:, :3, :3] = obj_rot
                        obj_corners = torch.matmul(obj_rot, metas['obj_rest_corners_3d'][i]).transpose(2, 1).transpose(2, 1) + volume_joint_preds
                    else:
                        obj_transform[:, :3, :3] = torch.eye(3).to(input_img[i].device)
                        obj_corners = metas['obj_rest_corners_3d'][i] + volume_joint_preds
                    obj_pose_results['global_trans'] = obj_transform
                    obj_pose_results['center'] = volume_joint_preds
                    obj_pose_results['corners'] = obj_corners
                    if hand_pose_results is not None:
                        obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
                else:
                    obj_pose_results = None
                obj_pose_results_frames.append(obj_pose_results)

                # generate features for the sdf head
                sdf_feat = self.backbone_2_sdf(backbone_feat)
                sdf_feat_frames.append(sdf_feat)
        
            sdf_refined_feat_frames = [sdf_feat_frames[i].unsqueeze(1) for i in range(num_frames)]
            sdf_refined_feat_frames = torch.cat(sdf_refined_feat_frames, axis=1)
            sdf_refined_feat_frames = self.feat_transformer(sdf_refined_feat_frames)
            
            sdf_point_feat_frames = []
            for i in range(num_frames):
                sdf_point_feat, _ = pixel_align(
                    self.cfg, 
                    xyz_points_frames[i], 
                    self.cfg.num_sample_points, 
                    sdf_refined_feat_frames[i], 
                    metas['hand_center_3d'][i], 
                    metas['cam_intr'][i]
                )
                sdf_point_feat = sdf_point_feat.reshape(self.cfg.train_batch_size, self.cfg.num_sample_points, -1)
                sdf_point_feat = sdf_point_feat.mean(1)
                sdf_point_feat = self.sdf_encoder(sdf_point_feat)
                sdf_point_feat = sdf_point_feat.repeat_interleave(self.cfg.num_sample_points, dim=0)
                sdf_point_feat_frames.append(sdf_point_feat)

            if self.hand_sdf_head is not None:
                sdf_hand_frames, cls_hand_frames = [], []
                for i in range(num_frames):
                    if self.cfg.hand_encode_style == 'kine':
                        if metas['epoch'] >= self.cfg.pose_epoch:
                            hand_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, hand_pose_results_frames[i], 'hand')
                            hand_points = hand_points.reshape((-1, self.cfg.hand_point_latent))
                        else:
                            mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=True).cuda()
                            _, _, _, gt_global_trans, gt_rot_center = mano_layer(metas['hand_poses'][i], th_betas=metas['hand_shapes'][i], root_palm=False)
                            gt_hand_pose_results = {}
                            gt_hand_pose_results['global_trans'] = gt_global_trans
                            hand_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, gt_hand_pose_results, 'hand')
                            hand_points = hand_points.reshape((-1, self.cfg.hand_point_latent))
                    elif self.cfg.hand_encode_style == 'gt_kine':
                            mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=True).cuda()
                            _, _, _, gt_global_trans, gt_rot_center = mano_layer(metas['hand_poses'][i], th_betas=metas['hand_shapes'][i], root_palm=False)
                            gt_hand_pose_results = {}
                            gt_hand_pose_results['global_trans'] = gt_global_trans
                            hand_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, gt_hand_pose_results, 'hand')
                            hand_points = hand_points.reshape((-1, self.cfg.hand_point_latent))
                    else:
                        hand_points = xyz_points_frames[i].reshape((-1, self.cfg.hand_point_latent))
                    hand_sdf_decoder_inputs = torch.cat([sdf_point_feat_frames[i], hand_points], dim=1)
                    sdf_hand, cls_hand = self.hand_sdf_head(hand_sdf_decoder_inputs)
                    sdf_hand = torch.clamp(sdf_hand, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
                    sdf_hand_frames.append(sdf_hand)
                    cls_hand_frames.append(cls_hand)
            else:
                sdf_hand_frames = None
                cls_hand_frames = None
        
            if self.obj_sdf_head is not None:
                sdf_obj_frames = []
                for i in range(num_frames):
                    if self.cfg.obj_encode_style == 'kine':
                        if metas['epoch'] >= self.cfg.pose_epoch: 
                            obj_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, obj_pose_results_frames[i], 'obj')
                            obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                        else:
                            gt_obj_pose_results = {}
                            gt_obj_pose = metas['obj_transform'][i]
                            if self.rot_head is None:
                                gt_obj_pose[:, :3, :3] = torch.eye(3)
                            gt_obj_pose_results['global_trans'] = gt_obj_pose
                            obj_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, gt_obj_pose_results, 'obj')
                            obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                    elif self.cfg.obj_encode_style == 'gt_trans':
                            gt_obj_pose_results = {}
                            gt_obj_pose = metas['obj_transform'][i]
                            gt_obj_pose[:, :3, :3] = torch.eye(3)
                            gt_obj_pose_results['global_trans'] = gt_obj_pose
                            obj_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, gt_obj_pose_results, 'obj')
                            obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                    elif self.cfg.obj_encode_style == 'gt_transrot':
                            gt_obj_pose_results = {}
                            gt_obj_pose = metas['obj_transform'][i]
                            gt_obj_pose_results['global_trans'] = gt_obj_pose
                            obj_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, gt_obj_pose_results, 'obj')
                            obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                    else:
                        obj_points = xyz_points_frames[i].reshape((-1, self.cfg.obj_point_latent))
                    obj_sdf_decoder_inputs = torch.cat([sdf_point_feat_frames[i], obj_points], dim=1)
                    sdf_obj, _ = self.obj_sdf_head(obj_sdf_decoder_inputs)
                    sdf_obj = torch.clamp(sdf_obj, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
                    sdf_obj_frames.append(sdf_obj)
            else:
                sdf_obj_frames = None

            sdf_results = {}
            sdf_results['hand'] = sdf_hand_frames
            sdf_results['obj'] = sdf_obj_frames
            sdf_results['cls'] = cls_hand_frames

            loss = {}
            if self.hand_sdf_head is not None:
                loss_hand_frames = []
                for i in range(num_frames):
                    loss_hand_frames.append(self.cfg.hand_sdf_weight * self.loss_l1(sdf_hand_frames[i] * mask_hand, sdf_gt_hand_frames[i] * mask_hand) / mask_hand.sum())
                loss['hand_sdf'] = sum(loss_hand_frames) / num_frames

            if self.obj_sdf_head is not None:
                loss_obj_frames = []
                for i in range(num_frames):
                    loss_obj_frames.append(self.cfg.obj_sdf_weight * self.loss_l1(sdf_obj_frames[i] * mask_obj, sdf_gt_obj_frames[i] * mask_obj) / mask_obj.sum())
                loss['obj_sdf'] = sum(loss_obj_frames) / num_frames

            if cls_hand_frames is not None:
                loss_hand_cls_frames = []
                for i in range(num_frames):
                    if cls_hand_frames[i] is None:
                        continue
                    if metas['epoch'] >= self.cfg.sdf_add_epoch:
                        loss_hand_cls_frames.append(self.cfg.hand_cls_weight * self.loss_ce(cls_hand_frames[i], cls_data_frames[i]))
                    else:
                        loss_hand_cls_frames.append(0. * self.loss_ce(cls_hand_frames[i], cls_data_frames[i]))
                if len(loss_hand_cls_frames) > 0:
                    loss['hand_cls'] = sum(loss_hand_cls_frames) / len(loss_hand_cls_frames)
                

            if self.mano_head is not None:
                loss_mano_joints_frames = []
                loss_mano_shape_frames = []
                loss_mano_pose_frames = []
                for i in range(num_frames):
                    valid_idx = hand_pose_results_frames[i]['vis']
                    loss_mano_joints_frames.append(self.cfg.mano_joints_weight * self.loss_l2(valid_idx.unsqueeze(-1) * hand_pose_results_frames[i]['joints'], valid_idx.unsqueeze(-1) * targets['hand_joints_3d'][i]))
                    loss_mano_shape_frames.append(self.cfg.mano_shape_weight * self.loss_l2(hand_pose_results_frames[i]['shape'], torch.zeros_like(hand_pose_results_frames[i]['shape'])))
                    loss_mano_pose_frames.append(self.cfg.mano_pose_weight * self.loss_l2(valid_idx * (hand_pose_results_frames[i]['pose'][:, 3:] - hand_pose_results_frames[i]['mean_pose']), valid_idx * torch.zeros_like(hand_pose_results_frames[i]['pose'][:, 3:])))
                loss['mano_joints'] = sum(loss_mano_joints_frames) / num_frames
                loss['mano_shape'] = sum(loss_mano_shape_frames) / num_frames
                loss['mano_pose'] = sum(loss_mano_pose_frames) / num_frames

            if self.volume_head is not None:
                loss_volume_joints_frames = []
                for i in range(num_frames):
                    volume_joint_targets = targets['obj_center_3d'][i].unsqueeze(1)
                    loss_volume_joints_frames.append(self.cfg.volume_weight * self.loss_l2(volume_joint_preds_frames[i], volume_joint_targets))
                loss['volume_joint'] = sum(loss_volume_joints_frames) / num_frames
            
            if self.rot_head is not None:
                loss_obj_corner_frames = []
                for i in range(num_frames):
                    loss_obj_corner_frames.append(self.cfg.corner_weight * self.loss_l2(obj_pose_results_frames[i]['corners'], targets['obj_corners_3d'][i]))
                loss['obj_corner'] = sum(loss_obj_corner_frames) / num_frames

            return loss, sdf_results, hand_pose_results_frames, obj_pose_results_frames
        else:
            with torch.no_grad():
                input_img = inputs['img']
                num_frames = len(input_img)
                
                sdf_feat_frames, obj_pose_results_frames, hand_pose_results_frames, volume_joint_preds_frames = [], [], [], []
                for i in range(num_frames):
                    # go through backbone
                    backbone_feat = self.backbone(input_img[i])

                    # generate features for the sdf head
                    sdf_feat = self.backbone_2_sdf(backbone_feat)
                    # sdf_feat = sdf_feat.mean(3).mean(2)  # [B, 512]
                    # sdf_feat = self.sdf_encoder(sdf_feat)
                    sdf_feat_frames.append(sdf_feat)

                    volume_results = {}
                    if self.volume_head is not None:
                        hm_feat = self.neck(backbone_feat)
                        hm_pred = self.volume_head(hm_feat)
                        hm_pred, hm_conf = soft_argmax(cfg, hm_pred, num_joints=1)
                        volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'][i], metas['cam_intr'][i])
                        volume_joint_preds_frames.append(volume_joint_preds)
                        volume_results['joints'] = volume_joint_preds
                    else:
                        volume_results = None

                    if self.mano_head is not None:
                        mano_feat = self.backbone_2_mano(backbone_feat)
                        mano_feat = mano_feat.mean(3).mean(2)
                        hand_pose_results = self.mano_head(mano_feat)
                        hand_pose_results = get_mano_preds(hand_pose_results, self.cfg, metas['cam_intr'][i], metas['hand_center_3d'][i])
                    else:
                        hand_pose_results = None
                    hand_pose_results_frames.append(hand_pose_results)
        
                    if self.rot_head is not None:
                        rot_feat = self.backbone_2_rot(backbone_feat)
                        rot_feat = rot_feat.mean(3).mean(2)
                        obj_rot = self.rot_head(rot_feat)
                    else:
                        obj_rot = None

                    # convert the object pose to the hand-relative coordinate system
                    if cfg.obj_trans or cfg.obj_rot:
                        obj_pose_results = {}
                        obj_transform = torch.zeros(self.cfg.test_batch_size, 4, 4).to(input_img.device)
                        obj_transform[:, :3, 3] = volume_joint_preds[:, 0, :] - metas['hand_center_3d'][i]
                        obj_transform[:, 3, 3] = 1
                        if cfg.obj_rot:
                            obj_transform[:, :3, :3] = obj_rot
                            obj_corners = torch.matmul(obj_rot, metas['obj_rest_corners_3d'][i]).transpose(2, 1).transpose(2, 1) + volume_joint_preds
                        else:
                            obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
                            obj_corners = metas['obj_rest_corners_3d'][i] + volume_joint_preds
                        obj_pose_results['global_trans'] = obj_transform
                        obj_pose_results['center'] = volume_joint_preds
                        obj_pose_results['corners'] = obj_corners
                        if hand_pose_results is not None:
                            obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
                    else:
                        obj_pose_results = None
                    obj_pose_results_frames.append(obj_pose_results)
            return sdf_feat_frames, hand_pose_results_frames, obj_pose_results_frames


def get_model(cfg, is_train):
    num_resnet_layers = int(cfg.backbone.split('_')[-1])
    backbone = ResNetBackbone(num_resnet_layers)
    if is_train:
        backbone.init_weights()

    neck_inplanes = 2048 if num_resnet_layers == 50 else 512
    if cfg.obj_trans:
        neck = UNet(neck_inplanes, 256, 3)
    else:
        neck = None

    if cfg.fa_trans:
        feat_transformer = FactorizedVideoTransformer(image_size=16, num_frames=cfg.num_frames, depth=8, dim_head=512)
    else:
        feat_transformer = ILAVideoTransformer(
            image_size=8, 
            num_frames=cfg.num_frames,
            dim=512,
            depth=8,
            heads=8,
            dim_head=64,
            in_channels=512,
        )
        
    if cfg.hand_branch:
        hand_sdf_head = SDFHead(cfg.sdf_latent, cfg.hand_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], cfg.hand_cls, cfg.sdf_head['num_class'])
    else:
        hand_sdf_head = None
    
    if cfg.obj_branch:
        obj_sdf_head = SDFHead(cfg.sdf_latent, cfg.obj_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], False, cfg.sdf_head['num_class'])
    else:
        obj_sdf_head = None
    
    if cfg.mano_branch:
        mano_head = ManoHead(depth=cfg.mano_depth)
    else:
        mano_head = None
    
    if cfg.obj_trans:
        volume_head = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
    else:
        volume_head = None
    
    if cfg.obj_rot:
        rot_head = RotHead(rot_format=cfg.rot_format)
    else:
        rot_head = None

    ho_model = model(cfg, backbone, neck, hand_sdf_head, obj_sdf_head, mano_head, volume_head, rot_head, feat_transformer)

    return ho_model


if __name__ == '__main__':
    model = get_model(cfg, True)
    input_size = (2, 3, 256, 256)
    input_img = torch.randn(input_size)
    input_point_size = (2, 2000, 3)
    input_points = torch.randn(input_point_size)
    sdf_results, hand_pose_results, obj_pose_results = model(input_img, input_points)