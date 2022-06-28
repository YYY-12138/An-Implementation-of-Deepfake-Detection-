import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import default_data_transforms


cuda = True


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
   
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=cuda):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=cuda):
   
    preprocessed_image = preprocess_image(image, cuda)
    output = model(preprocessed_image)
    output = post_function(output)
    _, prediction = torch.max(output, 1)  
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

def test_full_image_network(video_path, output_path, model=None, model_path=None,
                            start_frame=0, end_frame=None, threshold=.1, cuda=cuda):

    print('Starting: {}'.format(video_path))
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    if model is None:
        if model_path is not None:
            model = torch.load(model_path, map_location=lambda storage, loc: storage)
            print('Model found in {}'.format(model_path))
        else:
            print('No model found, initializing random model.')
    if cuda:
        model = model.cuda()
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    reader = cv2.VideoCapture(video_path)
    os.makedirs(output_path, exist_ok=True)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print('num_frames', num_frames)
    frame_num = 0
    predictions = []
    total_steps = 100
    frame_step = num_frames // total_steps

    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    print('end_frame', end_frame)
    pbar = tqdm(total=end_frame - start_frame)

    while reader.isOpened():
        reader.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, image = reader.read()
        if image is None:
            break
        frame_num += frame_step
        pbar.update(frame_step)

        height, width = image.shape[:2]

        face_detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            face = faces[0]
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            prediction, output = predict_with_model(cropped_face, model, cuda=cuda)
            predictions.append(prediction)
            tqdm.write(f'prediction = {prediction}')

            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 0 else 'real'
            color = (0, 0, 255) if prediction == 0 else (0, 255, 0)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list) + '=>' + label, (x, y + h + 30),
                        font_face, font_scale,
                        color, thickness, 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        print('frame_num', frame_num)
        #cv2.imwrite('E:/YJC/'+str(frame_num)+'.png', image)
        if frame_num >= end_frame:
            break
        cv2.imshow('Frame', image)
        cv2.waitKey(10)
    pbar.close()
    reader.release()
    cv2.destroyAllWindows()
    
    import numpy as np
    if np.mean(predictions) > threshold:
        return 1
    else:
        return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str, default='./test_video/2.chendaoming_1.mp4')
    p.add_argument('--model_path', '-m', type=str, default='model.pth')
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_full_image_network(**vars(args))

