/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.icube.android.examples.medclassification;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import com.icube.android.examples.medclassification.env.ImageUtils;
import com.icube.android.examples.medclassification.tflite.Classifier;
import com.icube.android.examples.medclassification.tflite.Detector;
import com.icube.android.examples.medclassification.tflite.TFLiteObjectDetectionAPIModel;
import com.icube.android.examples.medclassification.env.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

abstract class ClassificationService {
    private static final Logger LOGGER = new Logger();
    private boolean debug = false;
    private static Runnable postInferenceCallback;

    private static Classifier.Model model = Classifier.Model.FLOAT_MOBILENET;
    private static Classifier.Device device = Classifier.Device.CPU;
    private static int numThreads = -1;

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "2c_4000.tflite";
    private static final String TF_OD_API_LABELS_FILE = "labelmap_med.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static Classifier classifier;
    private static Detector detector;
    private static Bitmap rgbFrameBitmap = null;
    private static Bitmap croppedBitmap = null;
    private static Bitmap cropCopyBitmap = null;
    private static boolean computingDetection = false;
    private static long timestamp = 0;
    private static Matrix frameToCropTransform;
    private static Matrix cropToFrameTransform;

    private int imageSizeX;
    private int imageSizeY;

    public static String processImage(Context context, Bitmap bm) {
        ++timestamp;
        final long currTimestamp = timestamp;

        if (computingDetection) {
            readyForNextImage();
            return null;
        }

        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");
        rgbFrameBitmap = Bitmap.createScaledBitmap(bm, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, true);

        final int cropSize = Math.min(rgbFrameBitmap.getWidth(), rgbFrameBitmap.getHeight());
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        bm.getWidth(), bm.getWidth(),
                        cropSize, cropSize,
                        0, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            context,
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing Detector!");
        }

        if (classifier != null) {
            LOGGER.d("Closing classifier.");
            classifier.close();
            classifier = null;
        }

        try {
            LOGGER.d(
                    "Creating classifier (model=%s, device=%s, numThreads=%d)", getModel(), getDevice(), getNumThreads());
            classifier = Classifier.create((Activity) context, getModel(), getDevice(), getNumThreads());
        } catch (IOException | IllegalArgumentException e) {
            LOGGER.e(e, "Failed to create classifier.");
        }

        LOGGER.i("Running detection on image " + currTimestamp);
        final List<Detector.Recognition> results = detector.recognizeImage(rgbFrameBitmap);

        cropCopyBitmap = Bitmap.createBitmap(rgbFrameBitmap);
        final Canvas canvas = new Canvas(cropCopyBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
        switch (MODE) {
            case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
        }

        String resultString = results.toString();
        for (final Detector.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= minimumConfidence) {
                List<Classifier.Recognition> cresults = new ArrayList<Classifier.Recognition>();

                canvas.drawRect(location, paint);
                cropToFrameTransform.mapRect(location);

                if (classifier != null) {
                    Log.e("cropped location : ", String.valueOf((int)location.left+ ", " + (int)location.top+ ", " +(int)location.right+ ", " +(int)location.bottom));

                    if (location.left < 0) {
                        location.left = 0;
                    }
                    if (location.top < 0 ) {
                        location.top = 0;
                    }
                    if(location.right > bm.getWidth()) {
                        location.right = bm.getWidth();
                    }
                    if (location.bottom > bm.getHeight()){
                        location.bottom = bm.getHeight();
                    }
                    Bitmap resize = Bitmap.createBitmap(bm, (int) location.left, (int) location.top, (int) location.width(), (int) location.height());
                    Bitmap b = Bitmap.createScaledBitmap(resize, 224, 224, false);

                    cresults = classifier.recognizeImage(b, 0);
                    resultString = cresults.get(0).getTitle();
                }

                result.setLocation(location);
            }
        }

        computingDetection = false;
        return resultString;
    }

    private enum DetectorMode {
        TF_OD_API;
    }

    protected static void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }

    protected static Classifier.Model getModel() {
        return model;
    }

    private void setModel(Classifier.Model model) {
        if (this.model != model) {
            LOGGER.d("Updating  model: " + model);
            this.model = model;
        }
    }

    protected static Classifier.Device getDevice() {
        return device;
    }

    protected static int getNumThreads() {
        return numThreads;
    }

}
