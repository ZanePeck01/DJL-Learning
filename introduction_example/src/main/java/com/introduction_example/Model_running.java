package com.introduction_example;

import java.awt.image.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import ai.djl.*;
import ai.djl.basicmodelzoo.basic.*;
import ai.djl.ndarray.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.translate.*;

public class Model_running {

    public static void main(String[] args) throws Exception {

        // ***** STEP 1: Load Image *****

        // load image from URL
        var img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        img.getWrappedImage();

        // ***** STEP 2: Load Model *****

        // load trained saved model
        Path modelDir = Paths
                .get("C:/Users/PC/OneDrive/Desktop/AI_Projects/DJL-Learning/introduction_example/build/mlp");
        // create instance of model
        Model model = Model.newInstance("mlp");

        // set nn architecture for input, output, and hidden layers
        model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));
        // load model parameters from directory
        model.load(modelDir);

        // ***** STEP 3: Create Translator *****

        // translator handles preprocessing of input image and postprocessing of model
        // output
        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

            // preprocess input image (turn intput into NDArray tensor)
            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);

                return new NDList(NDImageUtils.toTensor(array));
            }

            // postprocess model output (turn output NDArray tensor into Classifications
            // object)
            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                NDArray probabilities = list.singletonOrThrow().softmax(0);

                List<String> classNames = IntStream.range(0, 10)
                        .mapToObj(String::valueOf)
                        .collect(Collectors.toList());

                return new Classifications(classNames, probabilities);
            }

            // specify how to batch inputs and outputs
            @Override
            public Batchifier getBatchifier() {
                return Batchifier.STACK;
            }
        };

        // ***** STEP 4: Create Predictor *****S

        // predictor uses model and translator to perform inference
        var predictor = model.newPredictor(translator);

        // ***** STEP 5: Run Inference and Make Prediction *****

        // use predictor to classify input image
        var classifications = predictor.predict(img);

        System.out.println(classifications);

        /*
         * Example output:
         * Classifications:
         * {"class": "0", "probability": 0.99991}
         * {"class": "2", "probability": 0.00003}
         * {"class": "9", "probability": 0.00002}
         * {"class": "7", "probability": 0.00001}
         * {"class": "6", "probability": 3.7e-06}
         *
         * → The model is almost certain the image is a "0".
         */
    }
}
