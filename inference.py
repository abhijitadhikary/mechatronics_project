import cv2
import numpy as np
from model import *
from utils import *
import torch.optim as optim
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:\\Users\\abhi\\anaconda3\\pkgs\\libopencv-4.4.0-py37_2\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
height = 224
width = 224
num_channels_input = 3
batch_size = 4
num_keypoints_pose = 16

variant = 'fcn'
checkpoint_path = 'checkpoints'
checkpoint_name = f'checkpoint_{variant}'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 224
load_checkpoint = True
learning_rate = 0.0001

model = DeepLabV3(image_size=image_size, num_channels=num_channels_input, num_keypoints=num_keypoints_pose).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
model, optimizer, epoch_start, loss_train_best = load_pretrained_model(load_checkpoint, checkpoint_name, checkpoint_path, model, optimizer, epoch_start=0, loss_train_best=0)
model.eval()
# To capture video from webcam.
cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(5, 60)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, input_image = cap.read()
    height_input, width_input, num_channels = input_image.shape
    input_image = input_image[:, width_input//2-height_input//2:width_input//2-height_input//2+height_input]

    resized_image = cv2.resize(input_image, (height, width), interpolation=cv2.INTER_AREA)
    resized_image_tensor = transforms.ToTensor()(resized_image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, heatmap_fine = model(resized_image_tensor)
    keypoints_fine = (get_max_value_heatmap(heatmap_fine).to(device))
    joints_X, joints_Y = separate_X_Y(keypoints_fine[0])





    # draw image on canvas
    # fig = Figure()
    # canvas = FigureCanvas(fig)
    # ax = fig.gca()
    # ax.axis('off')
    # num_joints = len(joints_X)
    #
    # for index in range(num_joints - 1):
    #     if index == 9 or index == 5:
    #         continue
    #     start_point = (joints_X[index], joints_X[index + 1])
    #     end_point = (joints_Y[index], joints_Y[index + 1])
    #
    #     if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
    #         ax.plot(start_point, end_point, linewidth=5)
    # ax.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    # # ax.scatter(center[0], center[1], marker='o', color='y', s=200)  # plot center of the human
    # ax.scatter(joints_X, joints_Y, color='r', marker='o', s=100)
    # plt.xticks([])
    # plt.yticks([])
    # # plt.savefig('inference_test.png')
    #
    #
    #
    #
    #
    # canvas.draw()  # draw the canvas, cache the renderer
    #
    # # image_overlaid = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    # s, (width, height) = canvas.print_to_buffer()
    #
    # # Option 2a: Convert to a NumPy array.
    # image_overlaid = np.fromstring(s, np.uint8).reshape((height, width, 4))
    # image_overlaid = cv2.cvtColor(image_overlaid, cv2.COLOR_RGBA2BGR)
    # # Display
    # cv2.imshow('img', image_overlaid)
    # cv2.imshow('img', resized_image)
    # Stop if escape key is pressed
    joints_X *= (height_input / image_size)
    joints_Y *= (height_input / image_size)
    for i in range((len(joints_X) - 1)):
        if i == 9 or i == 5:
            continue
        cv2.putText(input_image, "{}".format(i), (joints_X[i], joints_Y[i]), cv2.FONT_HERSHEY_PLAIN, fontScale=0.7, color=(255, 255, 255))
        start_point = (joints_X[i], joints_Y[i])
        end_point = (joints_X[i + 1], joints_Y[i + 1])
        cv2.line(input_image, start_point, end_point, color=(0, 0, 255), thickness=5)

    # resized_image = cv2.resize(resized_image, (1080, 1080), interpolation=cv2.INTER_AREA)
    cv2.imshow('img', input_image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# Release the VideoCapture object
cap.release()