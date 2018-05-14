import cv2
from operator import itemgetter


# Filters out overlapping bounding boxes with a lower score, pass an image to see the processing
# output: filtered array of bounding boxes
def non_maximum_suppression(bounding_boxes, overlap_threshold, image=None):
    bounding_boxes.sort(key=itemgetter(4), reverse=True)

    filtered_bboxes = []

    while len(bounding_boxes) != 0:
        # Show process if image is given
        if image is not None:
            show_nms_process(image, bounding_boxes, filtered_bboxes, bounding_boxes[0])
            cv2.waitKey(0)

        # Remove current bbox from the array and add it to the filtered array
        bbox = bounding_boxes[0]
        filtered_bboxes.append(bbox)
        del bounding_boxes[0]

        if image is not None:
            print("Score:", bbox[4])
            show_nms_process(image, bounding_boxes, filtered_bboxes)
            cv2.waitKey(0)

        # Delete the ones that overlap it
        delete_indices = []
        for j, other_bbox in enumerate(bounding_boxes):
            deleted=False # Used for showing process

            overlap = calc_overlap(bbox, other_bbox)
            if overlap >= overlap_threshold:
                delete_indices.append(j)
                deleted = True

            if image is not None:
                print("Overlap:", overlap)
                show_nms_process(image, bounding_boxes, filtered_bboxes, other_bbox, delete=deleted)
                cv2.waitKey(0)

        for bbox_index in sorted(delete_indices, reverse=True):
            del bounding_boxes[bbox_index]

        if image is not None:
            show_nms_process(image, bounding_boxes, filtered_bboxes)

    return filtered_bboxes


# Calculates the Intersection over Union area given two bounding boxes
# output: float between 0 and 1
def intersection_over_union(bbox, other_bbox):

    # Sum of areas
    area1 = abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])
    area2 = abs(other_bbox[0] - other_bbox[2]) * abs(other_bbox[1] - other_bbox[3])
    areas_sum = area1 + area2

    # Calculate overlap area
    diff_hor = min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0])
    diff_ver = min(bbox[3], other_bbox[3]) - max(bbox[1], other_bbox[1])

    if diff_hor <= 0 or diff_ver <= 0:
        overlap = 0
    else:
        overlap = diff_hor * diff_ver

    # Combined areas (union) is sum of areas minus the overlap
    union = areas_sum - overlap

    return overlap / union


# Calculates the overlap percentage of two bounding boxes
# output: float between 0 and 1
def calc_overlap(bbox, other_bbox):

    area1 = abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])
    area2 = abs(other_bbox[0] - other_bbox[2]) * abs(other_bbox[1] - other_bbox[3])

    # Calculate overlap area
    diff_hor = min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0])
    diff_ver = min(bbox[3], other_bbox[3]) - max(bbox[1], other_bbox[1])

    if diff_hor <= 0 or diff_ver <= 0:
        return 0
    else:
        overlap = diff_hor * diff_ver

    return overlap / min(area1, area2)


# Shows the process of Non-Maximum Suppression
# output: void
def show_nms_process(image, bounding_boxes, filtered_bboxes, highlight=None, delete=None):
    canvas = image.copy()
    show_objects(canvas, bounding_boxes)
    if highlight is not None:
        color = (255, 255, 0)
        if delete is not None:
            color = (0, 255, 0)
            if delete:
                color = (0, 0, 255)

        cv2.rectangle(canvas, (highlight[0], highlight[1]), (highlight[2], highlight[3]), color, 2)
    cv2.imshow("Bounding boxes", canvas)

    canvas2 = image.copy()
    show_objects(canvas2, filtered_bboxes)
    cv2.imshow("Non-Maximum Suppressed", canvas2)


def show_objects(image, locations, color=(0, 255, 255)):
    for pos in locations:
        cv2.rectangle(image, (pos[0], pos[1]), (pos[2], pos[3]), color, 2)