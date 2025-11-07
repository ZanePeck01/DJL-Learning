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

        var img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        img.getWrappedImage();

        Path modelDir = Paths
                .get("C:/Users/PC/OneDrive/Desktop/AI_Projects/DJL-Learning/introduction_example/build/mlp");
        Model model = Model.newInstance("mlp");

        model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));

        model.load(modelDir);

        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);

                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                NDArray probabilities = list.singletonOrThrow().softmax(0);

                List<String> classNames = IntStream.range(0, 10)
                        .mapToObj(String::valueOf)
                        .collect(Collectors.toList());

                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                return Batchifier.STACK;
            }
        };

        var predictor = model.newPredictor(translator);

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
