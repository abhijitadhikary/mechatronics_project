from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
from model import *
from metrics import *

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)



load_checkpoint = True
learning_rate = 0.0001
num_epochs = 500
image_size = 224
num_channels_input = 3
batch_size = 3
num_keypoints_pose = 16

variant = 'fcn'
checkpoint_path = 'checkpoints'
checkpoint_name = f'checkpoint_{variant}'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

custom_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
])

dataset = MPII_dataset(custom_transforms=custom_transforms)
dataloader_train = DataLoader(dataset, shuffle=True, batch_size=batch_size)

# model = KeypointNet(image_size=image_size, num_channels=num_channels_input, num_keypoints=num_keypoints_pose).to(device)
model = FCN_Resnet101(image_size=image_size, num_channels=num_channels_input, num_keypoints=num_keypoints_pose).to(device)

criterion_heatmap = nn.BCELoss()
criterion_keypoints = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)






epoch_start = 0
epoch_end = num_epochs
loss_train_best = np.Inf

model, optimizer, epoch_start, loss_train_best = load_pretrained_model(load_checkpoint, checkpoint_name, checkpoint_path, model, optimizer, epoch_start, loss_train_best)


for epoch_index in range(epoch_start, epoch_end, 1):

    loss_train_epoch = 0
    num_batch_train = len(dataloader_train)

    for batch_index, current_batch in enumerate(dataloader_train):
        if batch_index == 7:
            pass
        else:
            # print()
            pass
        image, keypoints_real, visibility_array_batch, height_batch, width_batch, center_batch = current_batch

        model.train()
        image = image.to(device)

        heatmap_real = generate_keypoint_heatmap(image, height_batch, width_batch, visibility_array_batch, keypoints_real).to(device)

        keypoints_real = keypoints_real.to(device)

        heatmap_coarse, heatmap_fine = model(image)

        keypoints_coarse = get_max_value_heatmap(heatmap_coarse).to(device) / height_batch[0].item()
        keypoints_fine = get_max_value_heatmap(heatmap_fine).to(device) / height_batch[0].item()

        optimizer.zero_grad()

        loss_heatmap_coarse = criterion_heatmap(heatmap_coarse, heatmap_real)
        loss_heatmap_fine = criterion_heatmap(heatmap_fine, heatmap_real)

        loss_regression_coarse = criterion_keypoints(keypoints_coarse, keypoints_real)
        loss_regression_fine = criterion_keypoints(keypoints_fine, keypoints_real)

        # loss = loss_heatmap + loss_joints * 10
        loss = loss_heatmap_coarse + loss_heatmap_fine + loss_regression_coarse + loss_regression_fine

        loss.backward()
        optimizer.step()

        loss_train_batch = loss.item()
        loss_train_epoch += loss_train_batch

        print(f'Epoch: [{epoch_index}/{num_epochs}]\tBatch: [{batch_index}/{num_batch_train}]\tLoss: {loss_train_batch:.3f}')

        if batch_index == 0:

            with torch.no_grad():
                model.eval()
                # visualize
                eval_image_index = 0
                current_image = get_np_image_from_tensor(image[eval_image_index])

                joints_X_real, joints_Y_real = separate_X_Y(keypoints_real[eval_image_index])
                joints_X_pred, joints_Y_pred = separate_X_Y(keypoints_fine[eval_image_index]) #########

                current_visibility_array = get_np_from_tensor(visibility_array_batch[eval_image_index])
                current_height = get_np_from_tensor(height_batch[eval_image_index])
                current_width = get_np_from_tensor(width_batch[eval_image_index])
                current_center = get_np_from_tensor(center_batch[eval_image_index])

                current_heatmap_stack_real = get_np_from_tensor(heatmap_real[eval_image_index])
                current_heatmap_stack_coarse = get_np_from_tensor(heatmap_coarse[eval_image_index])
                current_heatmap_stack_fine = get_np_from_tensor(heatmap_fine[eval_image_index])

                current_keypoints_pred = keypoints_fine[eval_image_index]
                current_heatmap_stack_r = heatmap_real[eval_image_index].detach().cpu().numpy()
                current_heatmap_stack_c = heatmap_coarse[eval_image_index].detach().cpu().numpy()
                current_heatmap_stack_f = heatmap_fine[eval_image_index].detach().cpu().numpy()

                image_tag = f'{epoch_index}_{batch_index}_{eval_image_index}'
                display_overlaid_image_dual(current_image, joints_X_real, joints_Y_real, joints_X_pred, joints_Y_pred, current_center, current_height, current_width, image_tag, current_visibility_array, current_heatmap_stack_real, current_heatmap_stack_fine)
                visualize_keypoint_heatmaps(current_heatmap_stack_r, current_heatmap_stack_c, current_heatmap_stack_f, current_image, image_tag=f'heatmap_{image_tag}')

                # calculate PCKH
                calculate_pckh(joints_X_real, joints_Y_real, joints_X_pred, joints_Y_pred, current_visibility_array)

    if loss_train_epoch < loss_train_best:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint = {
                        'epoch_index': epoch_index,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': loss_train_best,
                    }
        torch.save(checkpoint, os.path.join(checkpoint_path, f'{checkpoint_name}.pth'))

        print(f'new best model saved at epoch: {epoch_index}')
        loss_train_best = loss_train_epoch


    loss_train_epoch /= num_batch_train
    scheduler.step(loss_train_epoch)

    print(f'-------------> Epoch: [{epoch_index}/{num_epochs}]\tLoss: {loss_train_epoch:.3f} <---------------')
























# image, joints_real, visibility_array_batch, height_batch, width_batch, center_batch = next(iter(dataloader_train))
#
# eval_image_index = 0
# current_image = image[eval_image_index].permute(1, 2, 0).detach().cpu().numpy()
# coordinates_real_combined = joints_real[eval_image_index].detach().cpu().numpy()
# joints_X_real = coordinates_real_combined[::2]
# joints_Y_real = coordinates_real_combined[1::2]
#
#
# current_visibility_array = visibility_array_batch[eval_image_index].detach().cpu().numpy()
# current_height = height_batch[eval_image_index].detach().cpu().numpy()
# current_width = width_batch[eval_image_index].detach().cpu().numpy()
# current_center = center_batch[eval_image_index].detach().cpu().numpy()
# display_overlaid_image(current_image, joints_X_real, joints_Y_real, current_center)