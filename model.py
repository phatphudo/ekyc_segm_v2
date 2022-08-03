import os

import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.resnet import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.ops import misc as misc_nn_ops
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps


class CardDetector(MaskRCNN):
    def __init__(self, model_type, device, pretrained=False, ft_ext=False):
        
        trainable_backbone_layers = None
        pretrained_backbone = False if pretrained else True
        
        trainable_backbone_layers = _validate_trainable_layers(pretrained or pretrained_backbone, 
                                                               trainable_backbone_layers, 5, 3)
        backbone = resnet50(pretrained=pretrained_backbone, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
        
        super().__init__(backbone, num_classes=91)
        
        if pretrained:
            url = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
            state_dict = load_state_dict_from_url(url)
            self.load_state_dict(state_dict)
            overwrite_eps(self, 0.0)
        
        assert model_type in ['single', 'multi']
        if model_type == 'single':
            num_classes = 1 + 1
        else:
            num_classes = 1 + 9
        
        if ft_ext:
            for param in self.parameters():
                param.requires_grad = False
        
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_channels = self.roi_heads.mask_predictor.conv5_mask.in_channels
        self.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 512, num_classes)
                
        self.model_type = model_type
        self.device = device
        
        self.to(device)
    
    def load_weights(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = checkpoint_path.split('/')[-1]
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(state_dict)
            print(f"Loaded checkpoint [{checkpoint}].")
