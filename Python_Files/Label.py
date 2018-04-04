import pandas as pd


def creat_label():
    """
    :return: A Pandas DataFrame that have the classes associated with the names
    """
    label_name = {
        'Real_name': ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                      "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                      "bicycle", "autre"],
        'Class_name': ["class" + str(i) for i in range(19)] + ["class19"]}
    return pd.DataFrame(data=label_name)
