import numpy as np


def bbox_overlaps(boxes1, boxes2):
    """Compute the IOUs between the two sets of boxes.

    Args:
        boxes1 (numpy.ndarray): [num_boxes1, 4]-D array
        boxes2 (numpy.ndarray): [num_boxes2, 4]-D array

    Each of a box comprises 4 coordinate values in [xmin, ymin, xmax, ymax] order.

    Returns:
        overlaps (numpy.ndarray): [num_boxes1, num_boxes2]-D array, which is the distance matrix of the two sets of boxes
    """
    # Compute the areas of `boxes1` and `boxes2`.
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)  # [num_boxes1]
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)  # [num_boxes2]

    # Compute the areas of the intersections.
    intersection_h = np.maximum(
        (np.minimum(np.expand_dims(boxes1[:, 3], axis=1), boxes2[:, 3]) -
         np.maximum(np.expand_dims(boxes1[:, 1], axis=1), boxes2[:, 1]) + 1),
        0
    )  # [num_boxes1, num_boxes2]-D
    intersection_w = np.maximum(
        (np.minimum(np.expand_dims(boxes1[:, 2], axis=1), boxes2[:, 2]) -
         np.maximum(np.expand_dims(boxes1[:, 0], axis=1), boxes2[:, 0]) + 1),
        0
    )  # [num_boxes1, num_boxes2]-D
    intersection = intersection_h * intersection_w  # [num_boxes1, num_boxes2]-D

    # Compute the areas of the unions.
    union = np.maximum(
        np.expand_dims(area1, 1) + area2 - intersection,
        np.finfo(float).eps
    )

    # Compute IOU values.
    iou = intersection / union

    return iou
