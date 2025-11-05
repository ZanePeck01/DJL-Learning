package com.introduction_example;

import java.nio.file.*;

import ai.djl.*;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.ndarray.types.*;
import ai.djl.training.*;
import ai.djl.training.dataset.*;
import ai.djl.training.initializer.*;
import ai.djl.training.loss.*;
import ai.djl.training.listener.*;
import ai.djl.training.evaluator.*;
import ai.djl.training.optimizer.*;
import ai.djl.training.util.*;
import ai.djl.basicmodelzoo.cv.classification.*;
import ai.djl.basicmodelzoo.basic.*;

public class Model_training {
    public static void main(String[] args) throws Exception {

        int batchSize = 32;
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        mnist.prepare(new ProgressBar());

        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));

        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                // softmaxCrossEntropyLoss is a standard loss for classification problems
                .addEvaluator(new Accuracy()) // Use accuracy to understand how accurate the model is
                .addTrainingListeners(TrainingListener.Defaults.logging());

        Trainer trainer = model.newTrainer(config);

        trainer.initialize(new Shape(1, 28 * 28));

        int epoch = 2;

        EasyTrain.fit(trainer, epoch, mnist, null);

        Path modelDir = Paths
                .get("C:/Users/PC/OneDrive/Desktop/AI_Projects/DJL-Learning/introduction_example/build/mlp");
        Files.createDirectories(modelDir);

        model.setProperty("Epoch", String.valueOf(epoch));
        model.save(modelDir, "mlp");

        /*
         * The Model
         * Name: mlp
         * Model Location:
         * "C:\Users\PC\OneDrive\Desktop\AI_Projects\DJL-Learning\introduction_example\build\mlp"
         * Data Type: float32
         * Epoch: 2
         */
    }
}
