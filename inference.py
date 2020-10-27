import cv2
import numpy as np
from model import *
from utils import *
import torch.optim as optim
from time import time
from emotion_recognition_model_custom_def import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def load_model_and_checkpoints():
    model = FacEmoteModel()
    model.to('cuda:0')

    checkpoint_saved = torch.load('checkpoints/ER_Model_Custom_checkpoint_5.pth')
    #model.model incase of pretrained
    model.load_state_dict(checkpoint_saved['model_state_dict'])
    return model

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

pose_model = FCN_Resnet101(image_size=image_size, num_channels=num_channels_input, num_keypoints=num_keypoints_pose).to(device)
optimizer = optim.Adam(pose_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
pose_model, optimizer, epoch_start, loss_train_best = load_pretrained_model(load_checkpoint, checkpoint_name, checkpoint_path, pose_model, optimizer, epoch_start=0, loss_train_best=0)
pose_model.eval()

face_model = load_model_and_checkpoints()
# face_model.eval()

# To capture video from webcam.
cap = cv2.VideoCapture(0)
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

    start_time = time()
    with torch.no_grad():
        _, heatmap_fine = pose_model(resized_image_tensor)
    fps = 1.0 / (time() - start_time)
    # print(f'FPS: {fps}')
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


    # convert to absolute coordinates
    joints_X *= (height_input / image_size)
    joints_Y *= (height_input / image_size)

    # extract face
    height_face = np.abs(int((joints_Y[9] - joints_Y[8])*1.1))
    face_midpoint_X = int((joints_X[8] + joints_X[9]) / 2)
    face_midpoint_Y = int((joints_Y[8] + joints_Y[9]) / 2)

    top_left_X = face_midpoint_X-(height_face//2)
    top_left_Y = face_midpoint_Y-(height_face//2)

    cv2.rectangle(input_image, (top_left_X, top_left_Y), (top_left_X + height_face, top_left_Y + height_face), (0, 255, 0), 5)

    # input_image = cv2.circle(input_image, (face_midpoint_X, face_midpoint_Y), 5, (255, 128, 128), -1)
    face_image = np.copy(input_image[top_left_Y:top_left_Y+height_face, top_left_X:top_left_X+height_face, :])
    face_h, face_w, face_c = face_image.shape
    if face_h <= 0 or face_w <= 0:
        face_image = np.zeros((96, 96, 3)).astype(np.float32)

    face_image_tensor = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image_tensor = cv2.resize(face_image_tensor, (48, 48), interpolation=cv2.INTER_AREA)
    face_image_tensor = transforms.ToTensor()(face_image_tensor).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        prediction = face_model.predict_from_image(face_image_tensor)
    cv2.putText(face_image, prediction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 1)

    # draw joints
    num_joints = (len(joints_X))
    for joint_index in range(num_joints):
        input_image = cv2.circle(input_image, (joints_X[joint_index], joints_Y[joint_index]), 20, (255, 255, 0), -1)
        cv2.putText(input_image, "{}".format(joint_index), (joints_X[joint_index], joints_Y[joint_index]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=2, color=(0, 255, 0))
        if joint_index < num_joints-1:
            if joint_index == 9 or joint_index == 5:
                continue
            start_point = (joints_X[joint_index], joints_Y[joint_index])
            end_point = (joints_X[joint_index + 1], joints_Y[joint_index + 1])
            cv2.line(input_image, start_point, end_point, color=(0, 0, 255), thickness=10)

    # resized_image = cv2.resize(resized_image, (1080, 1080), interpolation=cv2.INTER_AREA)
    cv2.putText(input_image, f'{int(fps)} FPS', (10, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=2, color=(0, 0, 255))

    # cv2.imshow('img', input_image)
    face_image = cv2.resize(face_image, (height_input, height_input), interpolation=cv2.INTER_AREA)
    combined_image = np.concatenate((input_image, face_image), axis=1)
    cv2.imshow('img', combined_image)

    print()
















    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# Release the VideoCapture object
cap.release()