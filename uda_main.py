import torch
import os
import config.source_on_target_cfg as config
import config.predict_cfg as pred_config
import numpy as np
from dataset import crop_fullimg, create_indices_cbst, crop_filtered_patches_cbst, create_indices
from torch.utils.data import DataLoader
from training_utils import mIoULoss, val_loop_patch, val_loop_patch_cbst, train_loop, model_save_checkpoint
from uNet import UNET
from utils import mask_to_rgb, tiling_torch, untiling_torch, read_rgb_mask
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import itertools

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU')

    print(torch.__version__)

    ################# Target dataset preparation
    target_team = config.target_team
    target_plant = config.target_plant
    t_range_h = config.t_range_h
    t_range_w = config.t_range_w

    rotate = config.rotate  # rotate images of 90 degrees

    model_name = config.model_name
    model_path = config.model_path
    model_path_1 = config.model_path_1

    target_dir = os.path.join(config.dataset_dir, target_team, target_plant)

    folder_images_target = os.path.join(target_dir, 'Images')
    folder_mask_target = os.path.join(target_dir, 'Masks')

    indices = np.arange(len(os.listdir(folder_images_target)))

    val_dataset_target = crop_fullimg(folder_images_target, folder_mask_target, indices, t_range_w,
                                             t_range_h, config.image_w, config.image_h, transform=False, rotate=False)

    val_dl = DataLoader(val_dataset_target, batch_size=1, shuffle=False)
    
    ################ Model preparation
    model = UNET(in_channels=3, out_channels=3, dataset_dir=config.dataset_dir).to(device)
    IoU = mIoULoss(weights=None)


    ################# LOADING the trained intial model

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    write_name = config.write_name
    write_path = config.write_path
    writer = SummaryWriter(write_path)

    val_loss = [0]
    preds_array, preds_prob_array = val_loop_patch_cbst(val_dl, model, config.pad, t_range_h, t_range_w, device, 0, val_loss, IoU, None,
                   writer, rotate, write_to_file=True)

    ################ Combined DATASET PREPARATION


    def get_thresholds(preds_prob_array, preds_one_hot, prop):
        # assume the preds_array is an array with [no.of masks, height, width, one code]
        print("in the get_thresholds function preds_prob_array")
        print(preds_prob_array.shape)
        print(preds_prob_array[0].size())
        thresholds = []
        crop_prob_list = []
        weed_prob_list = []
        bg_prob_list = []
        for mask, mask_prob in zip(preds_one_hot, preds_prob_array):
            flatten_mask = torch.flatten(mask)
            flatten_mask = flatten_mask.detach().cpu().numpy()
            flatten_mask_prob_bg = torch.flatten(mask_prob[0,:,:])
            flatten_mask_prob_crop = torch.flatten(mask_prob[1,:,:])
            flatten_mask_prob_weed = torch.flatten(mask_prob[2,:,:])
            flatten_mask_prob_bg = flatten_mask_prob_bg.detach().cpu().numpy()
            flatten_mask_prob_crop = flatten_mask_prob_crop.detach().cpu().numpy()
            #print(flatten_mask_prob_crop.shape)
            flatten_mask_prob_weed = flatten_mask_prob_weed.detach().cpu().numpy()
            index_bg = [index for index,value in enumerate(flatten_mask) if value == 0]
            index_crop = [index for index,value in enumerate(flatten_mask) if value == 1]
            index_weed = [index for index,value in enumerate(flatten_mask) if value == 2]
            #index_bg = [i for i in index_bg if (i not in index_crop) and (i not in index_weed)]
            bg_prob_list.append(flatten_mask_prob_bg[index_bg])
           
            if (len(index_crop) > 0):
                #print(flatten_mask_prob_crop[index_crop])
                crop_prob_list.append(flatten_mask_prob_crop[index_crop])
            if (len(index_weed) > 0):
                weed_prob_list.append(flatten_mask_prob_weed[index_weed])
        crop_prob_list = np.array(crop_prob_list)
        weed_prob_list = np.array(weed_prob_list)
        bg_prob_list = np.array(bg_prob_list)
        bg_prob_list = np.hstack(bg_prob_list)
        crop_prob_list = np.hstack(crop_prob_list)
        weed_prob_list = np.hstack(weed_prob_list)
        print(crop_prob_list)
        prop_sort = np.argsort(crop_prob_list)
        weed_sort = np.argsort(weed_prob_list)
        bg_sort = np.argsort(bg_prob_list)
        #with open('metrics_CBST.txt', 'a') as f:
            #f.write("***********************************************************")
            #f.write('\n')
            #f.writelines([str(bg_sort.shape[0]), str(" "), str(prop_sort.shape[0]), str(" "), str(weed_sort.shape[0])])
            #f.write('\n')
        print("the number of the background pixels are:")
        print(bg_sort.shape[0])
        print("the number of the crop pixels are:")
        print(prop_sort.shape[0])
        print("the number of the weed pixels are:")
        print(weed_sort.shape[0])
        print("the calculated thresholds are")
        #print(prop_sort)
        thresholds = [bg_prob_list[bg_sort[-int(prop*bg_sort.shape[0])]], crop_prob_list[prop_sort[-int(prop*prop_sort.shape[0])]], weed_prob_list[weed_sort[-int(prop*weed_sort.shape[0])]]]
        #thresholds = [weed_prob_list[weed_sort[-int(prop*weed_sort.shape[0])]], crop_prob_list[prop_sort[-int(prop*prop_sort.shape[0])]], weed_prob_list[weed_sort[-int(prop*weed_sort.shape[0])]]]
        # thresholds = [np.partition(preds_prob_array[index_crop], prop*len(index_crop)), 
        # np.partition(preds_prob_array[index_weed], prop*len(index_weed))]
        print(thresholds)
        return thresholds
    def filter_target_masks(mask_array_prob, mask_array, thresholds, mask_target_array):
        # assume the mask array is for each image, it's [ one-hot, height, width]
        for i in range(mask_array.shape[0]):
            for j in range(mask_array.shape[1]):
                #if (mask_array[i,j,0] == 0):
                if (mask_array[i,j,0] == 0 and mask_array_prob[i,j,0] < thresholds[0]):
                    mask_array[i,j,0] = 255
                    mask_target_array[i,j,0] = 255
                if (mask_array[i,j,0] == 1 and mask_array_prob[i,j,1] < thresholds[1]):
                    mask_array[i,j,0] = 255
                    mask_target_array[i,j,0] = 255
                if (mask_array[i,j,0] == 2 and mask_array_prob[i,j,2] < thresholds[2]):
                    mask_array[i,j,0] = 255
                    mask_target_array[i,j,0] = 255
        return mask_array
        
    def prepare_data(preds_array, preds_prob_array, prop):

        source_dir = os.path.join(config.dataset_dir, config.team, config.plant)
        folder_images_source = os.path.join(source_dir, 'Images')
        print(folder_images_source)
        folder_mask_source = os.path.join(source_dir, 'Masks')

        step = 1
        real_len = ((len(os.listdir(folder_images_source)) + preds_array.shape[0]) / step)
        #real_len = len(os.listdir(folder_images_source))
        train_indices_temp, val_indices, test_indices = create_indices_cbst(real_len,step)
        train_indices = train_indices_temp
        for i in range(6):
            train_indices = np.concatenate([train_indices, train_indices_temp])


        ima_list = []
        mask_list = []
        #print("the indices are")
        #print(train_indices)
        i = 0
        for img in sorted(os.listdir(folder_images_source)):
            i = i + 1
            img = Image.open(os.path.join(folder_images_source,img))
            ima_list.append(img)
            if (i == 1):
                print("the source image size is:")
                print(img.size)
        count = 0
        for mask in sorted(os.listdir(folder_mask_source)):
            #mask_list.append(Image.open(os.path.join(folder_mask_source, mask)))
            image = Image.open(str(os.path.join(folder_mask_source, mask)))
            #out_shape = [pred_config.image_w * pred_config.range_w, pred_config.image_h * pred_config.range_h]
            out_shape = [config.image_w * config.range_w, config.image_h * config.range_h] 
            img = image.resize(out_shape, Image.NEAREST)
            transform = transforms.Compose([transforms.PILToTensor()])
            img_tensor = transform(img)
            a = torch.reshape(img_tensor, (img_tensor.size()[1],img_tensor.size()[2],img_tensor.size()[0]))
            mask_list.append(img)
            if (count == 0):
                print("this is in the mask for source resized masrk size")
                print(img.size)
            count = count + 1
        i = 0
        for img in sorted(os.listdir(folder_images_target)):
            img = Image.open(os.path.join(folder_images_target,img))
            ima_list.append(img)
            if (i == 0):
                print("the target image size is:")
                print(img.size)
            i = i + 1

        # filtered_pred_masks = filter_target_masks(preds_array, prop)
        # for mask in filtered_pred_masks:
        count = 0
        target_array_list = []
        for mask in sorted(os.listdir(folder_mask_target)):
        #     #mask_list.append(Image.open(os.path.join(folder_mask_source, mask)))
            image = Image.open(str(os.path.join(folder_mask_target, mask)))
            out_shape = [config.image_w * t_range_w, config.image_h * t_range_h] 
            img = image.resize(out_shape, Image.NEAREST)
            img_array = np.array(img)
            target_array_list.append(read_rgb_mask(img_array))
            transform = transforms.Compose([transforms.PILToTensor()])
            img_tensor = transform(img)
            a = torch.reshape(img_tensor, (img_tensor.size()[1],img_tensor.size()[2],img_tensor.size()[0]))
            #mask_list.append(img)
            if (count == 0):
                print("the target mask shape is:")
                print(img.size)
                #print(np.array(a).shape)
            count = count + 1
        threshold_list = get_thresholds(preds_prob_array, preds_array, prop)
        for mask, mask_prob, mask_target_array in zip(preds_array, preds_prob_array, target_array_list) :
            #a = mask.detach().cpu().numpy()
            #a = a.reshape(a.shape[1],a.shape[2],a.shape[0])
            #b = mask_prob.detach().cpu().numpy()
            #b = b.reshape(b.shape[1],b.shape[2],b.shape[0])
            #filtered_mask = filter_target_masks(b, a, threshold_list)
            pred1 = mask.permute(1, 2, 0).detach().cpu().numpy()
            pred2 = mask_prob.permute(1, 2, 0).detach().cpu().numpy()
            filtered_mask = filter_target_masks(pred2, pred1, threshold_list, mask_target_array)
            pred = Image.fromarray(np.uint8(mask_to_rgb(filtered_mask)), 'RGB')
            #print("the pred image size is:")
            #print(pred1.shape)
            #this is when you use the pseudo labels for the target masks
            mask_list.append(pred)
            
            #mask_list.append(filtered_mask)
            #if (count == 0):
                #print("the shape of the target mask array for each image is:")
                #print(a.shape)
                # print(mask.size())
            #count = count + 1

        if config.tile_filtering:
            combined_dataset_source = crop_filtered_patches_cbst(ima_list, mask_list, train_indices, config.range_w, config.range_h, t_range_w, t_range_h,
                                                        config.image_w, config.image_h, config.pad, transform=True)
        else:
            combined_dataset_source = crop_patches_cbst(ima_list, mask_list, train_indices, config.range_w, config.range_h, t_range_w, t_range_h,
                                                    config.image_w, config.image_h, config.pad, transform=True)

        combine_dl = DataLoader(combined_dataset_source, batch_size=12, shuffle=True)

        return combine_dl

    def train_model(combine_dl, iteration):
        unet_model = UNET(in_channels=3, out_channels=3, dataset_dir=config.dataset_dir).to(device)
        train_IoULoss = mIoULoss(weights = pred_config.weights)
        val_IoULoss = mIoULoss(weights = None)
        optimizer = torch.optim.Adam(unet_model.parameters(), pred_config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=pred_config.patience,
                                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                            eps=1e-08, verbose=True)

            ############### TRAINING AND VALIDATION
        val_loss = [1000]

        if not os.path.exists(pred_config.model_path):
            os.makedirs(pred_config.model_path)

        save_path = os.path.join(pred_config.model_path,'unet_model_cbst.pt')
        paper_dir = os.path.join(config.dataset_dir,'PaperWork')
        val_model_name = config.model_name
        write_path = os.path.join(paper_dir, 'runs', val_model_name,'Val')
        val_writer = SummaryWriter(write_path)
        epochs = 50
        for epoch in range(0, epochs):

            lr = optimizer.param_groups[0]['lr']
            if lr < 1e-6:
                break

            print("-------------------------------")
            print(f"Epoch {epoch}")
            print(f"Learning rate: {lr}\n-------------------------------")
            train_loop(combine_dl, device, unet_model, optimizer, train_IoULoss)
            
            # if (epoch == epochs - 1):
            print("---------------------------Validation---------------------------")
            #with open('metrics_CBST.txt', 'a') as f:
                #f.write("*******************Epoch: {}***********************".format(epoch))
                #f.write('\n')
            preds_hot, preds_prob = val_loop_patch_cbst(val_dl, unet_model, config.pad, t_range_h, t_range_w, device, epoch, val_loss, val_IoULoss, scheduler, val_writer, write_to_file=False)
        
            model_save_checkpoint(unet_model, optimizer, val_loss[0], save_path, epoch)
            get_thresholds(preds_prob, preds_hot, 0.2)
        print("---------------------------Test---------------------------")
        with open('metrics_CBST.txt', 'a') as f:
                f.write("*******************Iteration: {}***********************".format(iteration))
                f.write('\n')
        preds_hot, preds_prob = val_loop_patch_cbst(val_dl, unet_model, config.pad, t_range_h, t_range_w, device, 0, val_loss, val_IoULoss, scheduler, val_writer, write_to_file=True)
        return preds_hot, preds_prob


    ################ CBST loop
    results_list = []
    count_list = []
    #init_train_dl = train_dl
    model1 = model
    proportion_list = [0.01,0.2,0.2]
    proportion = 0.2
    with open('metrics_CBST.txt', 'a') as f:
        f.write("**Adaptation from Robot {} to Robot {}, plant from {} to {}**".format(config.team, config.target_team, config.plant, config.target_plant))
        f.write('\n')
    combine_dl = prepare_data(preds_array, preds_prob_array, 0.1 )
            
    preds_array, preds_prob_array = train_model(combine_dl, -1)
    
    for i in range(10): 
        with open('metrics_CBST.txt', 'a') as f:
            f.write("**********Selection Proportion: {}**************".format(proportion))
            f.write('\n')
        combine_dl = prepare_data(preds_array, preds_prob_array, proportion )
        
        #if (i > 2):
        proportion = proportion + 0.05
            
        preds_array, preds_prob_array = train_model(combine_dl, i)
        
        #print("test in the cbst loop")
        #print(preds_array.shape)
        #checkpoint = torch.load(model_path_1)
        #model1.load_state_dict(checkpoint['model_state_dict'])
        #write_name = config.write_name
        #write_path = config.write_path
        #writer = SummaryWriter(write_path)
        #val_loss = [0]
        #preds_array = val_loop_patch_cbst(val_dl, model1, config.pad, t_range_h, t_range_w, device, 0, val_loss, IoU, None,writer, rotate)
        #results_list.append(print_accuracy(preds_array))
        

    ################ save images
    image_path = config.image_path

    # Printing of images
    model.eval()
    i = 0
    with torch.no_grad():
        for x, y in val_dl:

            y = y.to(device, dtype=torch.long)

            tiled_x = tiling_torch(x, config.pad, t_range_h, t_range_w).to(device, dtype=torch.float)
            if rotate:
                tiled_x = T.functional.rotate(tiled_x, -90)

            preds = model(tiled_x)
            if rotate:
                preds = T.functional.rotate(preds, 90)

            pred = untiling_torch(preds, config.pad, t_range_h, t_range_w)
            pred = torch.argmax(pred, 0, True)

            pred = pred.permute(1, 2, 0).detach().cpu().numpy()
            img = x[0].permute(1, 2, 0).detach().cpu().numpy()
            true = y[0].permute(1, 2, 0).detach().cpu().numpy()

            img = Image.fromarray(np.uint8(img * 255), 'RGB')
            true = Image.fromarray(np.uint8(mask_to_rgb(true)), 'RGB')
            pred = Image.fromarray(np.uint8(mask_to_rgb(pred)), 'RGB')

            f_path = os.path.join(image_path)
            if not os.path.exists(f_path):
                os.makedirs(f_path)

            img.save(os.path.join(f_path, str(i) + 'img.png'))
            true.save(os.path.join(f_path, str(i) + 'true.png'))
            pred.save(os.path.join(f_path, str(i) + 'pred.png'))

            i = i + 1







    


