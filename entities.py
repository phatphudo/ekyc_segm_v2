import os
import time

import torch
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import cv2
import numpy as np
from tqdm import tqdm
from imantics import Mask

import coco.utils as utils
from model import CardDetector
from dataset import SingleDataset, MultiDataset
from coco.engine import evaluate_


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.model_type = cfg['model']['model_type']
        self.run_no = cfg['model']['run_no']
        self.run_name = f"{self.model_type}_{self.run_no}"

        self.data_dicts = cfg['data']
        self.split_ratio = cfg['dataloader']['split_ratio']
        self.batch_size = cfg['dataloader']['batch_size']
        self.num_workers = cfg['dataloader']['num_workers']
        
        tfms = get_tfms(train=True)
        self.train_dl, self.val_dl = self.get_dl(tfms)
    
        self.model = get_model(cfg)
        
        self.epochs = cfg['trainer']['epochs']
        self.optim_fn = _get_optim(cfg['trainer']['optim_fn'])
        self.optim_hp = cfg['trainer']['optim_hp']
        if cfg['trainer']['lr_scheduler']:
            self.lr_scheduler = _get_sched(cfg['trainer']['lr_scheduler'])
        else:
            self.lr_scheduler = None
        self.writer = SummaryWriter(log_dir=f"runs/{self.run_name}/")
        self.model_dir = cfg['model_dir']
        
        self.test_dir = cfg['test_dir']
        
    def get_dl(self, tfms=None):
        datasets = []
        data_dicts = self.data_dicts
        split_ratio = self.split_ratio
        
        for data_dict in data_dicts:
            data_root = data_dict['data_root']
            img_folder = data_dict['img_folder']
            anno_file = data_dict['anno_file']
            if self.model_type == 'single':
                ds = SingleDataset(
                    os.path.join(data_root, img_folder),
                    os.path.join(data_root, anno_file),
                    tfms
                )
            else:
                ds = MultiDataset(
                    os.path.join(data_root, img_folder),
                    os.path.join(data_root, anno_file),
                    tfms
                )
            datasets.append(ds)
        dataset= ConcatDataset(datasets)
        val_size = int(split_ratio * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, 
                              num_workers=self.num_workers, collate_fn=utils.collate_fn)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, collate_fn=utils.collate_fn)
        
        return train_dl, val_dl
        
    def fit(self):
        torch.cuda.empty_cache()
        
        print("Training...")
        train_time = time.time()
        
        model = self.model
        epochs = self.epochs
        train_dl, val_dl = self.train_dl, self.val_dl
        optimizer = self.optim_fn(self.model.parameters(), **self.optim_hp)
        if self.lr_scheduler is not None:
            lr_sched = self.lr_scheduler(optimizer, self.optim_hp['lr'], epochs=self.epochs, 
                                         steps_per_epoch=len(self.train_dl))
        writer = self.writer
        
        for epoch in range(epochs):
            epoch_time = time.time()

            # Training phase:
            model.train()
            train_losses = []

            for i, batch in enumerate(tqdm(train_dl)):
                loss_dict = self.step(batch)
                loss = sum(loss for loss in loss_dict.values())
                train_losses.append(loss)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if lr_sched is not None:
                    curr_lr = lr_sched.get_last_lr()[0]
                    lr_sched.step()
                else:
                    curr_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Iter/learning_rate", curr_lr, epoch*len(train_dl) + i)
                
                for k in sorted(loss_dict.keys()):
                    writer.add_scalar(f"Iter/{k}", loss_dict[k].item(), epoch*len(train_dl) + i)

            if lr_sched is not None:
                last_lr = lr_sched.get_last_lr()[0]
            else:
                last_lr = optimizer.param_groups[0]['lr']

            # Validation phase:
            val_losses = []
            with torch.no_grad():
                for batch in val_dl:
                    loss_dict = self.step(batch)
                    loss = sum(loss for loss in loss_dict.values())
                    val_losses.append(loss)

            train_loss = torch.stack(train_losses).mean().item()
            val_loss = torch.stack(val_losses).mean().item()
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))
            print(f"Epoch [{epoch:>2d}] : time: {epoch_time} | last_lr: {last_lr:.6f} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                print("-"*20)
            
            writer.add_scalars('Epoch/train_loss vs. val_loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # writer.add_scalar('Epoch/last_lr', last_lr, epoch)

            # Checkpoints
            if epoch == 0:
                best = val_loss
            checkpoint_path = self.model_dir + f'segm_{self.run_name}_last.pth'
            torch.save(model.state_dict(), checkpoint_path)
            if val_loss < best:
                checkpoint_path = self.model_dir + f'segm_{self.run_name}_best.pth'
                torch.save(model.state_dict(), checkpoint_path)
                best = val_loss
        
        train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_time))
        print(f"Total training time: {train_time}")
        writer.close()
        
    def step(self, batch):
        images, targets = batch
        device = self.device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        return loss_dict

    def eval(self):
        print("Evaluating...")
        eval_time = time.time()
        
        tfms = get_tfms(train=False)
        if self.model_type == 'single':
            ds = SingleDataset(
                os.path.join(self.test_dir, 'images/'),
                os.path.join(self.test_dir, 'annotations.json'),
                tfms
            )
        else:
            ds = MultiDataset(
                os.path.join(self.test_dir, 'images/'),
                os.path.join(self.test_dir, 'annotations.json'),
                tfms
            )
        dl = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=utils.collate_fn)
        
        evaluate_(self.model, dl, self.device)
        
        eval_time = time.time() - eval_time
        eval_time = time.strftime('%H:%M:%S', time.gmtime(eval_time))
        print(f"Total evaluating time: {eval_time}")

def get_model(cfg):
    device = torch.device(cfg['device'])
    model = CardDetector(
        model_type=cfg['model']['model_type'],
        device=device,
        pretrained=cfg['model']['pretrained'],
        ft_ext=cfg['model']['ft_ext']
    )
    return model
    
def get_tfms(train=False):
    stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tfms = [T.ToTensor(), T.Normalize(*stats)]
    if train:
        tfms.extend([T.RandomAutocontrast(),
                     T.ColorJitter(brightness=.5, hue=.2)])
    return T.Compose(tfms)

def _get_optim(optim_name):
    if optim_name == 'adam':
        return torch.optim.Adam
    elif optim_name == 'sgd':
        return torch.optim.SGD
    else:
        raise NotImplementedError

def _get_sched(sched_name):
    if sched_name == 'one_cycle':
        return torch.optim.lr_scheduler.OneCycleLR
    elif sched_name == 'linear':
        return torch.optim.lr_scheduler.LinearLR
    else:
        raise NotImplementedError


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_type = cfg['model']['model_type']
        self.run_no = cfg['model']['run_no']
        self.run_name = f"{self.model_type}_{self.run_no}"    
        
        checkpoint = f"segm_{self.run_name}_best.pth"
        checkpoint_path = os.path.join(cfg['model_dir'], checkpoint)
        model = get_model(cfg)
        model.load_weights(checkpoint_path)
        model.eval()
        
        if self.model_type == 'single':
            classes = ['__background__', 'IDCard']
        elif self.model_type == 'multi':
            classes = ['__background__', 'CitizenCardV1_back', 'CitizenCardV1_front', 
                       'CitizenCardV2_back', 'CitizenCardV2_front', 'IdentificationCard_back',
                       'IdentificationCard_front', 'LicenseCard', 'Other', 'Passport']
        
        self.model = model
        self.classes = classes
        self.device = torch.device(cfg['device'])
        
        self.conf = cfg['confidence']
        self.out_dir = os.path.join(cfg['out_dir'], self.run_name)
        os.makedirs(self.out_dir, exist_ok=True)
        
    def inf_single_image(self, img_path, inf_mode='get_segm'):
        img_file = img_path.split('/')[-1]
        if img_file.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # img = Image.open(img_path).convert('RGB')
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tfms = get_tfms(train=False)
            img = tfms(img)
            img = img.to(self.device)
        else:
            print(f"Image file must be in ['jpg', 'png', 'jpeg']!")
        
        model_time = time.time()
        pred_dict = self.model([img])[0]
        print(f"Model time: {time.time() - model_time} sec")
        print(f"Memory used: {torch.cuda.max_memory_allocated(self.device) / 1024.**2} MB")
        
        if len(pred_dict['labels']) == 0:
            print(f"Model has no predictions for [{img_file}]")
            if inf_mode == 'get_segm':
                return None
            elif inf_mode == 'eval':
                return {}, None
        else:
            assert inf_mode in ['save_pred', 'save_segm', 'get_segm', 'eval']
            
            preds = get_pred(pred_dict, self.conf)

            if len(preds.items()) == 0:
                print(f"No predictions satisfying [conf={self.conf}] for [{img_file}]")
                if inf_mode == 'get_segm':
                    return None
                elif inf_mode == 'eval':
                    return preds, None
            else:
                preds['clss'] = [self.classes[i] for i in preds['clss']]
                
                detected, warp = segment_img(img_path, preds)

                if inf_mode == 'save_segm':
                    cv2.imwrite(os.path.join(self.out_dir, img_file), warp)
                    print(f"Output saved @ [{self.out_dir}]!")
                elif inf_mode == 'save_pred':
                    cv2.imwrite(os.path.join(self.out_dir, img_file), combine_imgs(detected, warp))
                    print(f"Output saved @ [{self.out_dir}]!")
                elif inf_mode == 'get_segm':
                    if self.model_type == 'single':
                        return warp
                    else:
                        return preds['clss'][0], warp
                elif inf_mode == 'eval':
                    return preds, combine_imgs(detected, warp)

def get_pred(pred, confidence=0.5):
    pred_score = list(pred['scores'].detach().cpu().numpy())
    t = []
    for i, x in enumerate(pred_score):
        if x > confidence: 
            t.append(i)
    
    if len(t) == 0:
        return {}
    else:
        pred_t = t[-1]
        # pred_t = np.argmax(np.array(pred_score)) # Take pred
        pred_masks = (pred['masks'] > 0.5).squeeze(0).detach().cpu().numpy()
        pred_class = list(pred['labels'].cpu().numpy())
        # pred_boxes = [[tuple(map(int, (i[0], i[1]))), tuple(map(int, (i[2], i[3])))] for i in list(pred['boxes'].detach().cpu().numpy())]
        pred_boxes = [list(map(int, (i[0], i[1], i[2], i[3]))) for i in list(pred['boxes'].detach().cpu().numpy())]

        return {
            'masks': pred_masks[:pred_t+1],
            'boxes': pred_boxes[:pred_t+1],
            'clss': pred_class[:pred_t+1],
            'scores':pred_score[:pred_t+1]
        }

def segment_img(img_path, preds):
    img = cv2.imread(img_path)
    img_org = img.copy()
    
    for i in range(len(preds['masks'])):
        mask = preds['masks'][i]
        if len(mask.shape) > 2:
            temp = mask.squeeze(0)
            rgb_mask = get_coloured_mask(temp)
        else:
            rgb_mask = get_coloured_mask(mask)
        
        img = cv2.addWeighted(img, 1, rgb_mask, 0.25, 0)
        x1, y1, x2, y2 = preds['boxes'][i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.putText(img, f"{preds['clss'][i]} {preds['scores'][i]:.2f}", (x1-3, y2-3), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    mask = preds['masks'][0]
    if len(mask.shape) > 2:
        mask = mask.squeeze(0)
    polygons = Mask(mask).polygons()
    points = polygons.points
    pts = np.array(points[0])
    warp = perspective(img_org, pts)

    return img, warp

def get_coloured_mask(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = [0, 255, 255]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def perspective(img, pts):
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    corners = [tl, tr, br, bl]
    src = np.array(corners, np.float32)

    # Get the shape of new image
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    x_min, y_min, x_max, y_max = map(int, (mins[0], mins[1], maxs[0], maxs[1]))
    new_w = x_max - x_min
    new_h = y_max - y_min

    # Define destination points on new image
    dst = np.array([[0, 0], [new_w - 1, 0],
                    [new_w - 1, new_h - 1], [0, new_h]], np.float32)

    # Perform 'reversed' perspective transform
    trans_mat = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, trans_mat, (new_w, new_h))

    if new_w < new_h:
        warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return warp

def combine_imgs(img1, img2, space=30):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    vis = np.zeros((max(h1, h2), w1+w2+space, 3), np.uint8)
    
    vis[:h1, :w1, :3] = img1
    vis[:h2, w1+space:w1+w2+space, :3] = img2
    
    return vis

