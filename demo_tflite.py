#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf

import cvui


def run_inference(interpreter, image, exp_mean, nbins=14, scale_factor=20):
    # ONNX Infomation
    input_details = interpreter.get_input_details()
    input_size = input_details[0]['shape']
    input_width = input_size[2]
    input_height = input_size[1]

    # Pre process
    data_lowlight = cv.resize(image, dsize=(input_width, input_height))
    data_lowlight = cv.cvtColor(data_lowlight, cv.COLOR_BGR2RGB)
    data_lowlight = data_lowlight / 255.0
    data_lowlight = data_lowlight.astype('float32')

    low_im_filter_max = np.max(data_lowlight, axis=2, keepdims=True)
    hist = np.zeros([1, 1, int(nbins + 1)])
    xxx, _ = np.histogram(
        low_im_filter_max,
        bins=int(nbins - 2),
        range=(np.min(low_im_filter_max), np.max(low_im_filter_max)),
    )
    hist_c = np.reshape(xxx, [1, 1, nbins - 2])
    hist[:, :,
         0:nbins - 2] = np.array(hist_c, dtype=np.float32) / np.sum(hist_c)
    hist[:, :, nbins - 2:nbins - 1] = np.min(low_im_filter_max)
    hist[:, :, nbins - 1:nbins] = np.max(low_im_filter_max)
    hist[:, :, -1] = exp_mean

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.reshape(1, input_height, input_width,
                                          3).astype(np.float32)
    hist = hist.transpose(2, 0, 1)
    hist = hist.reshape(1, 15, 1, 1).astype(np.float32)
    hist = hist.transpose(0, 2, 3, 1)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], data_lowlight)
    interpreter.set_tensor(input_details[1]['index'], hist)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    result = interpreter.get_tensor(output_details[0]['index'])

    # Post process
    output_image = np.squeeze(result[0])
    output_image = np.clip(output_image * 255.0, 0, 255)
    output_image = output_image.astype(np.uint8)
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='model/saved_model_320x240/FLW-Net_320x240_float16.tflite',
    )
    parser.add_argument("--num_threads", type=int, default=1)

    args = parser.parse_args()
    model_path = args.model
    num_threads = args.num_threads

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        num_threads=num_threads,
    )
    interpreter.allocate_tensors()

    # Parameter Window
    cvui.init('FLW-Net Parameters')
    exp_mean = [0.4]
    parameter_window = np.zeros((80, 320, 3), np.uint8)

    while True:
        # Parameter Window Update
        parameter_window[:] = (49, 52, 49)

        cvui.text(parameter_window, 10, 10, 'exp_mean')
        cvui.trackbar(parameter_window, 10, 30, 300, exp_mean, 0.0, 2, 1,
                      '%.2Lf', cvui.TRACKBAR_DISCRETE, 0.1)
        cvui.update()

        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        output_image = run_inference(
            interpreter,
            frame,
            exp_mean[0],
        )

        output_image = cv.resize(output_image,
                                 dsize=(frame_width, frame_height))

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('FLW-Net Input', debug_image)
        cv.imshow('FLW-Net Output', output_image)
        cv.imshow('FLW-Net Parameters', parameter_window)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
