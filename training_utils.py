import torch
import torch.nn as nn
import numpy as np
import os
import sys
from utils import untiling_torch, tiling_torch, calcMetricsPerPixels, calcAllMetrics
import torchvision.transforms as T
#import cv2
from scipy.ndimage import convolve


def cycle(img_X, msk_X, img_Y, gen_X, gen_Y, disc_X, disc_Y, mse, opt_disc, opt_gen, L1, phase, fcn_X,
          fcn_Y, IoU, opt_fcn, losses, lambda_gen, lambda_cycle, lambda_sem, lambda_phase, device):
    disc_X.train()
    disc_Y.train()
    gen_X.train()
    gen_Y.train()
    fcn_Y.train()

    ### X discriminator
    fake_img_X = gen_X(img_Y)
    D_X_real = disc_X(img_X)
    D_X_fake = disc_X(fake_img_X.detach())

    D_X_real_loss = mse(D_X_real, torch.ones_like(D_X_real))
    D_X_fake_loss = mse(D_X_fake, torch.zeros_like(D_X_fake))

    # X_reals += D_X_real.mean().item()
    # X_fakes += D_X_fake.mean().item()

    D_X_loss = D_X_real_loss + D_X_fake_loss

    ### Y discriminator
    fake_img_Y = gen_Y(img_X)
    D_Y_real = disc_Y(img_Y)
    D_Y_fake = disc_Y(fake_img_Y.detach())

    D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
    D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))

    # Y_reals += D_Y_real.mean().item()
    # Y_fakes += D_Y_fake.mean().item()

    D_Y_loss = D_Y_real_loss + D_Y_fake_loss

    ### Put it together
    D_loss = (D_X_loss + D_Y_loss) / 2
    losses[0] += D_loss.item()

    opt_disc.zero_grad()
    D_loss.backward()
    opt_disc.step()

    # adversarial loss for both generators
    D_X_fake = disc_X(fake_img_X)
    D_Y_fake = disc_Y(fake_img_Y)
    loss_G_X = mse(D_X_fake, torch.ones_like(D_X_fake))
    loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))

    # cycle loss
    cycle_X = gen_X(fake_img_Y)
    cycle_Y = gen_Y(fake_img_X)

    # phase loss
    if(lambda_phase > 0):
        loss_phase_X = phase(fake_img_Y, img_X)
        loss_phase_Y = phase(fake_img_X, img_Y)
    else:
        loss_phase_X = torch.zeros(1, dtype=torch.float, device=device)
        loss_phase_Y = torch.zeros(1, dtype=torch.float, device=device)

    loss_phase = 0.5 * (loss_phase_X + loss_phase_Y)

    cycle_X_loss = L1(img_X, cycle_X)
    cycle_Y_loss = L1(img_Y, cycle_Y)

    # semantic losses
    msk_Gx = fcn_Y(fake_img_Y)
    msk_FGx = fcn_X(cycle_X)
    loss_S_Gx = IoU(msk_Gx, msk_X)
    loss_S_FGx = IoU(msk_FGx, msk_X)

    msk_Fy = torch.argmax(fcn_X(fake_img_X), 1, True)
    msk_Y = fcn_Y(img_Y)
    msk_GFy = fcn_Y(cycle_Y)
    loss_S_y = IoU(msk_Y, msk_Fy)
    loss_S_GFy = IoU(msk_GFy, msk_Fy)

    # sum all components of the generator loss
    G_loss = (
            loss_G_Y * lambda_gen
            + loss_G_X * lambda_gen
            + cycle_Y_loss * lambda_cycle
            + cycle_X_loss * lambda_cycle
    )

    # sum all components of the semantic loss
    S_loss = (
        (
            loss_S_Gx
            + loss_S_FGx
            + loss_S_y
            + loss_S_GFy

        ) * lambda_sem
    )

    # phase loss
    P_loss = loss_phase * lambda_phase

    # full loss
    #loss = (
            #G_loss +
            #S_loss +
           # P_loss 
   # )
    
    # full loss
    loss = (
            G_loss +
            S_loss  
    )

    losses[1] += G_loss.item()
    losses[2] += S_loss.item()
    #losses[3] += P_loss.item()
    #losses[4] += loss.item()
    losses[3] += loss.item()

    print(
        f"[D_X: {D_X_loss.item():>5f}; D_Y: {D_Y_loss.item():>5f}; G_X: {loss_G_X.item():>5f}; G_Y: {loss_G_Y.item():>5f}; cycle_X: {cycle_X_loss.item():>5f}; cycle_Y: {cycle_Y_loss.item():>5f}; S_Gx: {loss_S_Gx.item():>5f}; S_FGx: {loss_S_FGx.item():>5f}; S_y: {loss_S_y.item():>5f}; S_GFy: {loss_S_GFy.item():>5f}]")

    opt_fcn.zero_grad()
    opt_gen.zero_grad()

    loss.backward()

    opt_fcn.step()
    opt_gen.step()


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

  
# custom loss function
class mIoULoss(nn.Module):
    def __init__(self, weights=None, n_classes=3):
        super(mIoULoss, self).__init__()
        self.classes = range(n_classes)
        self.weights=weights

    def forward(self, inputs, targets):

        n_classes = len(self.classes)
        
        #print("the target size in the loss forward function is:")
        #print(targets.size())

        #one_hot = torch.nn.functional.one_hot(targets.to(torch.int64), n_classes)
        one_hot = torch.nn.functional.one_hot(targets.to(torch.int64), 4)
        #print(one_hot.size())
        one_hot_1 = torch.zeros([one_hot.size()[0],one_hot.size()[1],one_hot.size()[2],one_hot.size()[3],n_classes],dtype=torch.int64,device=inputs.device)
        #print(one_hot_1.size())
        #for i in range(one_hot.size()[2]):
            #for j in range(one_hot.size()[3]):
                #if (one_hot[0,0,i,j,:] == [0,0,0,1]):
                    #one_hot_1[0,0,i,j,:] == [0,0,0]
                #else:
                    #one_hot_1[0,0,i,j,:] == [one_hot_1[0,0,i,j,:][0],one_hot_1[0,0,i,j,:][1],one_hot_1[0,0,i,j,:][2]]
            

        target_oneHot = torch.squeeze(one_hot[:,:,:,:,:3], 1).permute(0, 3, 1, 2)
        #print(target_oneHot[0,:,0,0])
        #target_oneHot = one_hot.permute(0, 3, 1, 2)
        
        loss = torch.zeros(n_classes, dtype=torch.float, device=inputs.device)

        if self.weights is None:
            weights = [1] * n_classes
        else:
            weights = self.weights

        for class_index, weight in zip(self.classes, weights):

            if len(inputs.size()) < 4:
                input_c = inputs[None, class_index, ...]
                N = 1
            else:
                input_c = inputs[:, class_index, ...]
                N = inputs.size()[0]
            target_oneHot_c = target_oneHot[:, class_index, ...]
            #print(target_oneHot_c.size())
            
            valid_pixels = torch.sum(target_oneHot, dim=1)
            #print(valid_pixels.size())

            inter = input_c * valid_pixels * target_oneHot_c * valid_pixels
            inter = inter.view(N, 1, -1).sum(2)

            union = input_c * valid_pixels + target_oneHot_c * valid_pixels- (input_c * target_oneHot_c * valid_pixels)
            union = union.view(N, 1, -1).sum(2)

            iou = (inter + 1e-7) / (union + 1e-7)
            
            loss[class_index] = (1.0 - iou.mean()) * weight
        
        return loss.mean()      # alternatively it is possible to use the sum instead of the mean

class mIoULoss_N(nn.Module):
    def __init__(self, weights=None, n_classes=3):
        super(mIoULoss, self).__init__()
        self.classes = range(n_classes)
        self.weights=weights

    def forward(self, inputs, targets):
	# the targets for the target masks is [1,1,512,768] with probabilities
        n_classes = len(self.classes)
        kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9  # 3x3 averaging kernel
                       
        pixel_weights = convolve(targets[0,0,:,:].cpu(), kernel, mode='constant', cval=0)
        
        min_val = pixel_weights.min(axis=(0, 1), keepdims=True)
        max_val = pixel_weights.max(axis=(0, 1), keepdims=True)
        pixel_weights = (pixel_weights - min_val) / (max_val - min_val)
        
        pixel_weights = np.ones((targets.size()[2], targets.size()[3]))
    
        print("the target size in the loss forward function is:")
        print(targets.size())

        #one_hot = torch.nn.functional.one_hot(targets.to(torch.int64), n_classes)
        #targets_argmax = torch.argmax(targets, 2, True)
        one_hot = torch.nn.functional.one_hot(targets.to(torch.int64), 4)

        #print(one_hot.size())
        one_hot_1 = torch.zeros([one_hot.size()[0],one_hot.size()[1],one_hot.size()[2],one_hot.size()[3],n_classes],dtype=torch.int64,device=inputs.device)
        #print(one_hot_1.size())
        #for i in range(one_hot.size()[2]):
            #for j in range(one_hot.size()[3]):
                #if (one_hot[0,0,i,j,:] == [0,0,0,1]):
                    #one_hot_1[0,0,i,j,:] == [0,0,0]
                #else:
                    #one_hot_1[0,0,i,j,:] == [one_hot_1[0,0,i,j,:][0],one_hot_1[0,0,i,j,:][1],one_hot_1[0,0,i,j,:][2]]
            

        target_oneHot = torch.squeeze(one_hot[:,:,:,:,:3], 1).permute(0, 3, 1, 2)
        #print(target_oneHot[0,:,0,0])
        #target_oneHot = one_hot.permute(0, 3, 1, 2)
        
        loss = torch.zeros(n_classes, dtype=torch.float, device=inputs.device)

        if self.weights is None:
            weights = [1] * n_classes
        else:
            weights = self.weights

        for class_index, weight in zip(self.classes, weights):

            if len(inputs.size()) < 4:
                input_c = inputs[None, class_index, ...]
                N = 1
            else:
                input_c = inputs[:, class_index, ...]
                N = inputs.size()[0]
            target_oneHot_c = target_oneHot[:, class_index, ...]
            #print(target_oneHot_c.size())
            
            pixel_weights_1d = torch.from_numpy(pixel_weights)
            pixel_weights_1d = pixel_weights_1d.cuda()
            pixel_weights_1d = torch.unsqueeze(pixel_weights_1d, 0)
            
            #print(target_oneHot_c.size())

            inter = input_c*pixel_weights_1d * target_oneHot_c*pixel_weights_1d
            inter = inter.view(N, 1, -1).sum(2)

            union = input_c*pixel_weights_1d + target_oneHot_c*pixel_weights_1d - (input_c*pixel_weights_1d * target_oneHot_c*pixel_weights_1d)
            
            #valid_pixels = torch.sum(target_oneHot, dim=1)
            #print(valid_pixels.size())

            #inter = input_c * valid_pixels * target_oneHot_c * valid_pixels
            #inter = inter.view(N, 1, -1).sum(2)

            #union = input_c * valid_pixels + target_oneHot_c * valid_pixels- (input_c * target_oneHot_c * valid_pixels)
            union = union.view(N, 1, -1).sum(2)

            iou = (inter + 1e-7) / (union + 1e-7)
            
            loss[class_index] = (1.0 - iou.mean()) * weight
        
        return loss.mean()      # alternatively it is possible to use the sum instead of the mean

def model_save_checkpoint(model, optimizer, loss, path, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


# using lr of old checkpoint
def load_checkpoint(path, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2
    fft_amp = torch.sqrt(fft_amp + 1e-20)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


class phase_loss(nn.Module):
    def __init__(self):
        super(phase_loss, self).__init__()

    def forward(self, img1, img2):
        # note that img2 has to be real image
        fft1 = torch.fft.fft2(img1)  # fft tranform
        fft2 = torch.fft.fft2(img2)  # fft tranform

        fft1 = torch.view_as_real(fft1)
        fft2 = torch.view_as_real(fft2)

        fft1 = torch.nn.functional.normalize(fft1)
        fft2 = torch.nn.functional.normalize(fft2)

        # amp1, pha1 = self.extract_ampl_phase(fft1)

        amp2, _ = extract_ampl_phase(fft2)

        amp2max, _ = torch.max(amp2, dim=2, keepdim=True)
        amp2max, _ = torch.max(amp2max, dim=3, keepdim=True)
        w2 = amp2 / (amp2max + 1e-20)

        inner_product = (fft1 * fft2).sum(dim=-1)
        norm1 = (fft1.pow(2).sum(dim=-1) + 1e-20).pow(0.5)
        norm2 = (fft2.pow(2).sum(dim=-1) + 1e-20).pow(0.5)
        cos = inner_product / (norm1 * norm2 + 1e-20)
        cos = cos * w2

        return -1.0 * cos.mean()

# validation loop by using image tiles previously produced and read in a directory
def val_loop(loader, model, model_name, pad, device, epoch, save_model, val_loss, IoULoss, main_dir, optimizer, scheduler,
             writer=None, file_path=None):
    size = len(loader.dataset)
    model.eval()
    val_loss[0] = 0

    class_metrics = np.zeros((2, 4))
    keys = ['IoU', 'Acc', 'Pre', 'Rec', 'F-s']
    # metrics = [meanIoU,crop_IoU,weed_IoU,meanAcc,crop_Acc,weed_Acc,meanPre,crop_Pre,weed_Pre,meanRec,crop_Rec,weed_Rec]

    with torch.no_grad():
        number_of_batch = 0
        for x, y in loader:
            number_of_batch = number_of_batch + 1
            x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.int)
            pred = model(x)
            val_loss[0] += IoULoss(pred, y).item()

            pred = untiling_torch(pred, pad)
            true = untiling_torch(y, pad)

            pred = torch.argmax(pred, 0, True)
            metrics = calcMetricsPerPixels(pred, true)

            for i in range(len(metrics)):
                for j in range(4):
                    class_metrics[i][j] += metrics[i][j]

    val_loss[0] = val_loss[0] / number_of_batch

    if scheduler != None:
        scheduler.step(val_loss[0])

    metrics = calcAllMetrics(class_metrics)

    ######### Write in a file
    print(f"Avg loss: {val_loss[0]:>8f}")
    path_write = os.path.join(file_path, 'filename.txt')
    with open(path_write, 'a') as f:

        original_stdout = sys.stdout  # Save a reference to the original standard outpu
        print(f"Epoch {epoch}\n-------------------------------")

        sys.stdout = f  # Change the standard output to the file we created.
        print(f"Epoch {epoch}\n-------------------------------")

        for i in range(len(keys)):
            sys.stdout = f  # Change the standard output to the file we created.
            print(
                "mean" + keys[i] + f": {metrics[i * 3]:>8f},  crop" + keys[i] + f" : {metrics[i * 3 + 1]:>8f},  weed" +
                keys[i] + f" : {metrics[i * 3 + 2]:>8f}")
            sys.stdout = original_stdout  # Reset the standard output to its original value
            print(
                "mean" + keys[i] + f": {metrics[i * 3]:>8f},  crop" + keys[i] + f" : {metrics[i * 3 + 1]:>8f},  weed" +
                keys[i] + f" : {metrics[i * 3 + 2]:>8f}")

    if save_model:
        path_write = os.path.join(main_dir, 'pySave', model_name, 'epoch' + str(epoch))
        if not os.path.exists(path_write):
            os.makedirs(path_write)
        model_save_checkpoint(model, optimizer, val_loss[0], os.path.join(path_write, 'model.pt'), epoch)

    if writer is not None:
        writer.add_scalar('val loss',
                          val_loss[0],
                          epoch)

        for m in range(len(keys)):
            writer.add_scalar('mean' + keys[m], metrics[m * 3], epoch)

            writer.add_scalar('crop' + keys[m], metrics[m * 3 + 1], epoch)

            writer.add_scalar('weed' + keys[m], metrics[m * 3 + 2], epoch)

        writer.flush()


def train_loop(loader, device, model, optimizer, loss_fn):
    model.train()
    # loop = tqdm(loader)
    size = len(loader.dataset)
    loss = 0
    class_metrics = np.zeros((3, 4))
    keys = ['IoU', 'Acc', 'Pre', 'Rec', 'F-s']
    for batch_idx, (data, targets) in enumerate(loader):

        data = data.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        # forward
        predictions = model(data)
        # loss = loss_fn(predictions, targets[:,0,:,:])
        loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch_idx * len(data)
        # loss_temp = loss.item()*batch_idx + loss
        # loss += loss_temp
        # loss = loss/(batch_idx+1)

        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        for x, y in zip(predictions, targets):
            x = torch.argmax(x, 0, True)
            metrics = calcMetricsPerPixels(x, y)

            for i in range(len(metrics)):
                for j in range(4):
                    class_metrics[i][j] += metrics[i][j]

    print('---------------------------Training---------------------------')

    metrics = calcAllMetrics(class_metrics)
    for i in range(len(keys)):
        print(
            "mean" + keys[i] + f": {metrics[i * 4]:>8f},  bg" + keys[i] + f" : {metrics[i * 4 + 1]:>8f},  crop" + keys[
                i] + f" : {metrics[i * 4 + 2]:>8f},  weed" +
            keys[i] + f" : {metrics[i * 4 + 3]:>8f}")
    #return model

# validation loop with dynamic patching of tiles
def val_loop_patch(loader, model, pad, range_h, range_w ,device, epoch, val_loss, IoULoss, scheduler, writer=None, rotate=False):

    model.eval()
    val_loss[0] = 0

    class_metrics = np.zeros((3, 4))
    keys = ['IoU', 'Acc', 'Pre', 'Rec', 'F-s']
    # the metrics are: meanIoU, crop_IoU, weed_IoU, meanAcc, crop_Acc, weed_Acc, meanPre, crop_Pre, weed_Pre, meanRec, crop_Rec, weed_Rec

    with torch.no_grad():
        number_of_batch = 0
        for x, y in loader:
            number_of_batch = number_of_batch + 1

            tiled_x = tiling_torch(x,pad,range_h,range_w)
            tiled_x, true = tiled_x.to(device, dtype=torch.float), y.to(device, dtype=torch.int)
            if rotate:
                tiled_x = T.functional.rotate(tiled_x, -90)

            preds = model(tiled_x)
            #print(preds.size())
            if (number_of_batch == 1):
            	print(preds.size())
           #preds shape is [6,3,320,320]
            if rotate:
                preds = T.functional.rotate(preds, 90)
            

            pred = untiling_torch(preds,pad,range_h,range_w)
            pred = pred.to(device, dtype=torch.float)
           

            val_loss[0] += IoULoss(pred, true).item()

            pred = torch.argmax(pred, 0, True)
            # pred shape is [1,512,768]
            metrics = calcMetricsPerPixels(pred, true)

            for i in range(len(metrics)):
                for j in range(4):
                    class_metrics[i][j] += metrics[i][j]

    val_loss[0] = val_loss[0] / number_of_batch

    if scheduler != None:
        scheduler.step(val_loss[0])

    metrics = calcAllMetrics(class_metrics)
    for i in range(len(keys)):
        print(
            "mean" + keys[i] + f": {metrics[i * 4]:>8f},  bg" + keys[i] + f" : {metrics[i * 4 + 1]:>8f},  crop" + keys[i] + f" : {metrics[i * 4 + 2]:>8f},  weed" + keys[i] + f" : {metrics[i * 4 + 3]:>8f}")

    if writer is not None:
        writer.add_scalar('val loss',
                          val_loss[0],
                          epoch)

        for m in range(len(keys)):
            writer.add_scalar('mean' + keys[m], metrics[m * 4], epoch)
            writer.add_scalar('bg' + keys[m], metrics[m * 4 + 1], epoch)
            writer.add_scalar('crop' + keys[m], metrics[m * 4 + 2], epoch)
            writer.add_scalar('weed' + keys[m], metrics[m * 4 + 3], epoch)

        writer.flush()

# validation loop with dynamic patching of tiles
def val_loop_patch_cbst(loader, model, pad, range_h, range_w ,device, epoch, val_loss, IoULoss, scheduler, writer=None, rotate=False, write_to_file=False):

    model.eval()
    val_loss[0] = 0

    class_metrics = np.zeros((3, 4))
    keys = ['IoU', 'Acc', 'Pre', 'Rec', 'F-s']
    # the metrics are: meanIoU, crop_IoU, weed_IoU, meanAcc, crop_Acc, weed_Acc, meanPre, crop_Pre, weed_Pre, meanRec, crop_Rec, weed_Rec

    preds_list = []
    preds_prob_list = []
    with torch.no_grad():
        number_of_batch = 0
        for x, y in loader:
            number_of_batch = number_of_batch + 1

            tiled_x = tiling_torch(x,pad,range_h,range_w)
            tiled_x, true = tiled_x.to(device, dtype=torch.float), y.to(device, dtype=torch.int)
            if rotate:
                tiled_x = T.functional.rotate(tiled_x, -90)

            preds = model(tiled_x)
            if rotate:
                preds = T.functional.rotate(preds, 90)

            pred = untiling_torch(preds,pad,range_h,range_w)
            pred = pred.to(device, dtype=torch.float)

            preds_prob_list.append(pred)

            val_loss[0] += IoULoss(pred, true).item()

            pred = torch.argmax(pred, 0, True)
            metrics = calcMetricsPerPixels(pred, true)
            
            preds_list.append(pred)

            for i in range(len(metrics)):
                for j in range(4):
                    class_metrics[i][j] += metrics[i][j]

        preds_array = np.array(preds_list)
        preds_prob_array = np.array(preds_prob_list)

    val_loss[0] = val_loss[0] / number_of_batch

    if scheduler != None:
        scheduler.step(val_loss[0])

    metrics = calcAllMetrics(class_metrics)
    #with open('metrics_CBST.txt', 'a') as f:
        #f.write("***********************************************************")
        #f.write('\n')
    for i in range(len(keys)):
    	#print("from the training util val_loop_patch_cbst")
    	if write_to_file:
    	    with open('metrics_CBST.txt', 'a') as f:
                a = "mean" + keys[i] + f": {metrics[i * 4]:>8f},  bg" + keys[i] + f" : {metrics[i * 4 + 1]:>8f},  crop" + keys[i] + f" : {metrics[i * 4 + 2]:>8f},  weed" + keys[i] + f" : {metrics[i * 4 + 3]:>8f}"
                f.write(a)
                f.write('\n')
    	print(
            "mean" + keys[i] + f": {metrics[i * 4]:>8f},  bg" + keys[i] + f" : {metrics[i * 4 + 1]:>8f},  crop" + keys[i] + f" : {metrics[i * 4 + 2]:>8f},  weed" + keys[i] + f" : {metrics[i * 4 + 3]:>8f}")
        
    
          
    if writer is not None:
        writer.add_scalar('val loss',
                          val_loss[0],
                          epoch)

        for m in range(len(keys)):
            writer.add_scalar('mean' + keys[m], metrics[m * 4], epoch)
            writer.add_scalar('bg' + keys[m], metrics[m * 4 + 1], epoch)
            writer.add_scalar('crop' + keys[m], metrics[m * 4 + 2], epoch)
            writer.add_scalar('weed' + keys[m], metrics[m * 4 + 3], epoch)

        writer.flush()

    return preds_array, preds_prob_array
