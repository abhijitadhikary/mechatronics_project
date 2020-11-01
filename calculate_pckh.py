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
batch_size = 32
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
model.eval()

# average_pckh = np.zeros(10).astype(np.float32)
total = 0
pckh_list = []
for epoch_index in range(0, 1, 1):

    loss_train_epoch = 0
    num_batch_train = len(dataloader_train)
    with torch.no_grad():
        for batch_index, current_batch in enumerate(dataloader_train):
            if batch_index == 7:
                pass
            else:
                # print()
                pass
            image, keypoints_real, visibility_array_batch, height_batch, width_batch, center_batch = current_batch

            image = image.to(device)

            heatmap_real = generate_keypoint_heatmap(image, height_batch, width_batch, visibility_array_batch, keypoints_real).to(device)

            keypoints_real = keypoints_real.to(device)

            heatmap_coarse, heatmap_fine = model(image)

            keypoints_coarse = get_max_value_heatmap(heatmap_coarse).to(device) / height_batch[0].item()
            keypoints_fine = get_max_value_heatmap(heatmap_fine).to(device) / height_batch[0].item()


            loss_heatmap_coarse = criterion_heatmap(heatmap_coarse, heatmap_real)
            loss_heatmap_fine = criterion_heatmap(heatmap_fine, heatmap_real)

            loss_regression_coarse = criterion_keypoints(keypoints_coarse, keypoints_real)
            loss_regression_fine = criterion_keypoints(keypoints_fine, keypoints_real)

            # loss = loss_heatmap + loss_joints * 10
            loss = loss_heatmap_coarse + loss_heatmap_fine + loss_regression_coarse + loss_regression_fine

            loss_train_batch = loss.item()
            loss_train_epoch += loss_train_batch

            print(f'Batch: [{batch_index}/{num_batch_train}]\tLoss: {loss_train_batch:.3f}')

            # visualize
            x_real = []
            y_real = []
            x_pred = []
            y_pred = []
            vis = []

            for eval_image_index in range(image.shape[0]):
                joints_X_real, joints_Y_real = separate_X_Y(keypoints_real[eval_image_index])
                joints_X_pred, joints_Y_pred = separate_X_Y(keypoints_fine[eval_image_index]) #########
                current_visibility_array = get_np_from_tensor(visibility_array_batch[eval_image_index])

                x_real.append(joints_X_real)
                y_real.append(joints_Y_real)
                x_pred.append(joints_X_pred)
                y_pred.append(joints_X_pred)
                vis.append(current_visibility_array)

            x_real = np.array(x_real)
            y_real = np.array(y_real)
            x_pred = np.array(x_pred)
            y_pred = np.array(y_pred)
            vis = np.array(vis)
            # calculate PCKH
            current_pckh, class_names = calculate_pckh(x_real, y_real, x_pred, y_pred, vis)

            # average_pckh += current_pckh
            total += 1
            pckh_list.append(current_pckh)
                # print(pckh_list)
                # print(len(pckh_list))

    loss_train_epoch /= len(dataloader_train)

    print(f'-------------> \tLoss: {loss_train_epoch:.3f} <---------------')

    pickle.dump(pckh_list, open( "pckh_batch.pkl", "wb" ) )
    average_pckh = np.array(pckh_list)
    average_pckh = np.mean(average_pckh, axis=0)
    print((average_pckh, class_names))

    import pickle


