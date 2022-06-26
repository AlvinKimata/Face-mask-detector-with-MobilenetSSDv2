import os
import cv2
import time
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('model')
 }


files = {
    'PIPELINE_CONFIG':os.path.join('model', 'pipeline.config'),
    'LABELMAP': os.path.join('model', LABEL_MAP_NAME)
}
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


#Load pipeline config and build a detection model.
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training = False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-5')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


#Real-time detection from a webcam.
#sample_video = 'Learn ASL Alphabet.mp4'
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while cap.isOpened():
    start = time.time() #Start time.
    ret, frame = cap.read()
    image_np = np.array(frame)

    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype = tf.float32)

    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    #Detection_classes should be int.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw = 5,
                min_score_thresh = .4, agnostic_mode = False)
    
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    stop = time.time()
    seconds = stop - start
    
    #Frames per second.
    fps = 1 / seconds
    print(f'Estimated frames per second: {fps}')
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
