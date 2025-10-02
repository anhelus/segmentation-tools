def convert_to_yolo_format(box, image_size):
    """
    Converts a bounding box from [x_min, y_min, x_max, y_max] to YOLO format.
    """
    image_width, image_height = image_size
    x_min, y_min, x_max, y_max = box

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height

    return x_center_normalized, y_center_normalized, width_normalized, height_normalized
