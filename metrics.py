import numpy as np

"""
The canonical part stick order:
0 Head
1 Torso
2 Right Upper Arm
3 Right Lower Arm
4 Right Upper Leg
5 Right Lower Leg
6 Left Upper Arm
7 Left Lower Arm
8 Left Upper Leg
9 Left Lower Leg
"""

CANONICAL_STICK_NAMES = ['Head', 'Torso', 'RU Arm', 'RL Arm', 'RU Leg',
                         'RL Leg', 'LU Arm', 'LL Arm', 'LU Leg', 'LL Leg']


def eval_relaxed_pcp(gt_joints, predicted_joints, thresh=0.5):
    """
    Relaxed PCP as in DeepPose paper.
    Compute average relaxed pcp per stick.
    Args:
      gt_joints, predicted_joints: arrays of gt and predicted joints in the canonical order
      thresh: fraction of the gt stick length. This is the maximal average deviation of the
        predicted joints of the stick from the gt joints position of the stick.
    Returns:
        pcp_per_stick: array of pcp scores. i-th element is the pcp score for the i-th stick
    """
    if len(gt_joints) != len(predicted_joints):
        raise ValueError('Len of gt must be equal to len of predicted')
    if len(gt_joints) == 0:
        raise ValueError('Empty array')

    num_examples = len(gt_joints)

    # the number of sticks for a pose
    num_sticks = gt_joints[0]['sticks'].shape[0]
    if num_sticks != 10:
        raise ValueError('PCP requires 10 sticks. Provided: {}'.format(num_sticks))
    is_matched = np.zeros((num_examples, num_sticks), dtype=int)

    for i in range(num_examples):
        for stick_id in range(num_sticks):
            gt_stick_len = np.linalg.norm(gt_joints[i]['sticks'][stick_id, :2] -
                                          gt_joints[i]['sticks'][stick_id, 2:])
            delta_a = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, :2] -
                                     gt_joints[i]['sticks'][stick_id, :2]) / gt_stick_len
            delta_b = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, 2:] -
                                     gt_joints[i]['sticks'][stick_id, 2:]) / gt_stick_len
            delta = (delta_a + delta_b) / 2.0

            is_matched[i, stick_id] = delta <= thresh
    pcp_per_stick = np.mean(is_matched, 0)
    return pcp_per_stick


def eval_strict_pcp(gt_joints, predicted_joints, thresh=0.5):
    """
    Compute average pcp per stick
    Args:
      gt_joints, predicted_joints: arrays of gt and predicted joints in the canonical order
      thresh: fraction of the gt stick length. This is the maximal deviation of the
        predicted joint from the gt joint position.
    Returns:
        pcp_per_stick: array of pcp scores. i-th element is the pcp score for the i-th stick
    """
    if len(gt_joints) != len(predicted_joints):
        raise ValueError('Len of gt must be equal to len of predicted')
    if len(gt_joints) == 0:
        raise ValueError('Empty array')

    num_examples = len(gt_joints)
    # the number of sticks for a pose
    num_sticks = gt_joints[0]['sticks'].shape[0]
    if num_sticks != 10:
        raise ValueError('PCP requires 10 sticks. Provided: {}'.format(num_sticks))
    is_matched = np.zeros((num_examples, num_sticks), dtype=int)

    for i in range(num_examples):
        for stick_id in range(num_sticks):
            gt_stick_len = np.linalg.norm(gt_joints[i]['sticks'][stick_id, :2] -
                                          gt_joints[i]['sticks'][stick_id, 2:])
            delta_a = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, :2] -
                                     gt_joints[i]['sticks'][stick_id, :2]) / gt_stick_len
            delta_b = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, 2:] -
                                     gt_joints[i]['sticks'][stick_id, 2:]) / gt_stick_len

            is_matched[i, stick_id] = (delta_a <= thresh and delta_b <= thresh)
    pcp_per_stick = np.mean(is_matched, 0)
    return pcp_per_stick


def average_pcp_left_right_limbs(pcp_per_stick):
    part_names = ['Head', 'Torso', 'U Arm', 'L Arm', 'U Leg', 'L Leg', 'mean']
    pcp_per_part = pcp_per_stick[:2].tolist() + \
                   [(pcp_per_stick[i] + pcp_per_stick[i + 4]) / 2 for i in range(2, 6)]
    pcp_per_part.append(np.mean(pcp_per_part))
    return pcp_per_part, part_names


def eval_pckh(gt_joints, predicted_joints, thresh=0.5):
    """
    Compute average PCKh per joint.
    Matching threshold is 50% (thresh) of the head segment box size by default
    Args:
      gt_joints, predicted_joints: arrays of gt and predicted joints in the canonical order
      thresh: fraction of the head segment length. This is the maximal deviation of the
        predicted joint from the gt joint position.
    Returns:
        pckh_per_joint: array of PCKh scores. i-th element is the PCKh score for the i-th joint
    """
    if len(gt_joints) != len(predicted_joints):
        raise ValueError('Len of gt must be equal to len of predicted')
    if len(gt_joints) == 0:
        raise ValueError('Empty array')
    num_joints = 16
    num_examples = len(gt_joints)

    is_matched = np.zeros((num_examples, num_joints), dtype=int)

    for i in range(num_examples):
        if gt_joints[i]['joints'].shape != (num_joints, 2):
            raise ValueError('MPII::PCKh requires 16 joints with 2D coordinates for each.'
                             ' Person {}: provided joints shape: {}'.format(i, gt_joints[0]['joints'].shape))
        head_id = 0
        gt_head_len = np.linalg.norm(gt_joints[i]['sticks'][head_id, :2] -
                                     gt_joints[i]['sticks'][head_id, 2:])
        for joint_id in range(num_joints):
            delta = np.linalg.norm(predicted_joints[i]['joints'][joint_id] -
                                   gt_joints[i]['joints'][joint_id]) / gt_head_len

            is_matched[i, joint_id] = delta <= thresh
    pckh_per_joint = np.mean(is_matched, 0)
    return pckh_per_joint


def average_pckh_symmetric_joints(pckh_per_joint, dataset_name=None):
    # if dataset_name not in ['mpii', 'lsp']:
    #     raise ValueError('Unknown dataset {}'.format(dataset_name))

    joint_names = ['Head', 'Neck', 'Shoulder',
                   'Elbow', 'Wrist',
                   'Hip', 'Knee', 'Ankle',
                   'Thorax', 'Pelvis']
    if dataset_name == 'lsp':
        joint_names = joint_names[:-2]
    pckh_symmetric_joints = pckh_per_joint[:2].tolist()
    for i in range(2, 8):
        pckh_symmetric_joints.append((pckh_per_joint[i] + pckh_per_joint[i + 6]) / 2.0)
    pckh_symmetric_joints += pckh_per_joint[14:].tolist()
    return pckh_symmetric_joints, joint_names


def joints2sticks(joints):
    """
    Args:
        joints: array of joints in the canonical order.
      The canonical joint order:
        0 Head top
        1 Neck
        2 Right shoulder (from person's perspective)
        3 Right elbow
        4 Right wrist
        5 Right hip
        6 Right knee
        7 Right ankle
        8 Left shoulder
        9 Left elbow
        10 Left wrist
        11 Left hip
        12 Left knee
        13 Left ankle
        14 Thorax
        15 Pelvis
    Returns:
        sticks: array of sticks in the canonical order.
      The canonical part stick order:
        0 Head
        1 Torso
        2 Right Upper Arm
        3 Right Lower Arm
        4 Right Upper Leg
        5 Right Lower Leg
        6 Left Upper Arm
        7 Left Lower Arm
        8 Left Upper Leg
        9 Left Lower Leg
    """
    assert joints.shape == (16, 2)
    stick_n = 10  # number of stick
    sticks = np.zeros((stick_n, 4), dtype=np.float32)
    sticks[0, :] = np.hstack([joints[0, :], joints[1, :]])  # Head
    sticks[1, :] = np.hstack([joints[14, :], joints[15, :]])  # Torso
    sticks[2, :] = np.hstack([joints[2, :], joints[3, :]])  # Left U.arms
    sticks[3, :] = np.hstack([joints[3, :], joints[4, :]])  # Left L.arms
    sticks[4, :] = np.hstack([joints[5, :], joints[6, :]])  # Left U.legs
    sticks[5, :] = np.hstack([joints[6, :], joints[7, :]])  # Left L.legs
    sticks[6, :] = np.hstack([joints[8, :], joints[9, :]])  # Right U.arms
    sticks[7, :] = np.hstack([joints[9, :], joints[10, :]])  # Right L.arms
    sticks[8, :] = np.hstack([joints[11, :], joints[12, :]])  # Right U.legs
    sticks[9, :] = np.hstack([joints[12, :], joints[13, :]])  # Right L.legs
    return sticks


def convert2canonical(joints):
    """
    Convert joints to evaluation structure.
    Permute joints according to the canonical joint order.
    """
    assert joints.shape[1:] == (16, 2), 'MPII must contain 14 joints per person'
    # convert to the canonical joint order
    joint_order = [9,  # Head top
                   8,  # Neck
                   12,  # Right shoulder
                   11,  # Right elbow
                   10,  # Right wrist
                   2,  # Right hip
                   1,  # Right knee
                   0,  # Right ankle
                   13,  # Left shoulder
                   14,  # Left elbow
                   15,  # Left wrist
                   3,  # Left hip
                   4,  # Left knee
                   5,  # Left ankle
                   7,  # Thorax
                   6]  # Pelvis
    assert len(joint_order) == len(set(joint_order))
    canonical = [dict() for _ in range(joints.shape[0])]
    for i in range(joints.shape[0]):
        canonical[i]['joints'] = joints[i, joint_order, :]
        canonical[i]['sticks'] = joints2sticks(canonical[i]['joints'])
    return canonical


def calculate_pckh(real_X, real_Y, pred_X, pred_Y, visibility_array):
    batch_size = real_X.shape[0]

    for index in range(batch_size):
        real_X[index] = np.array(np.where(visibility_array[index] > 0, real_X[index], 0)) * 224
        real_Y[index] = np.array(np.where(visibility_array[index] > 0, real_Y[index], 0)) * 224

        pred_X[index] = np.array(np.where(visibility_array[index] > 0, pred_X[index], 0)) * 224
        pred_Y[index] = np.array(np.where(visibility_array[index] > 0, pred_Y[index], 0)) * 224

    # real.shape = bs x 16 x 2
    real = np.zeros((batch_size, 16, 2))
    pred = np.zeros((batch_size, 16, 2))

    for index in range(batch_size):
        real[index, :, 1] = real_X[index]
        real[index, :, 0] = real_Y[index]

        pred[index, :, 1] = pred_X[index]
        pred[index, :, 0] = pred_Y[index]


    # num_batches = real.shape[0]
    real_sticks = convert2canonical(real)
    pred_sticks = convert2canonical(pred)

    result = eval_pckh(real_sticks, pred_sticks)
    average_pckh = average_pckh_symmetric_joints(result)
    # print(result)
    # print(average_pckh)
    average_pckh = average_pckh[0]
    class_names = average_pckh[1]

    return average_pckh, class_names
