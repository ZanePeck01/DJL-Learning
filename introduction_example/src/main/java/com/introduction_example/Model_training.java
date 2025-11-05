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

//Model Training Example (for image classification)

public class Model_training {
    public static void main(String[] args) throws Exception {

        // ***** STEP 1: Prepare the Data *****

        // batch size determines how many samples to process before updating the model
        int batchSize = 32;
        // Load MNIST dataaset, processing in batches of 32 images and true means to
        // shuffle the data
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        // download and prepare dataset
        mnist.prepare(new ProgressBar());

        // ***** STEP 2: Define the Model *****

        // create a model instance, holds nn architecture and parameters
        Model model = Model.newInstance("mlp");
        // set nn architecture for input, output, and hidden layers
        model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));

        // ***** STEP 3: Configure the Training *****

        // set training configurations, including loss function, evaluation metrics, and
        // training listeners
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                // softmaxCrossEntropyLoss is a standard loss for classification problems
                .addEvaluator(new Accuracy()) // Use accuracy to understand how accurate the model is
                .addTrainingListeners(TrainingListener.Defaults.logging());

        // ***** STEP 4: Create Trainer *****

        // trainer manages training loop and parameter updates, uses configs defined
        // above
        Trainer trainer = model.newTrainer(config);

        // initialize trainer with input data shape (batch size, 1D array of 28*28
        // pixels)
        trainer.initialize(new Shape(1, 28 * 28));

        // ***** STEP 5: Train the Model *****

        // epochs represent number of times to go through entire dataset
        int epoch = 2;

        // runs training loop
        EasyTrain.fit(trainer, epoch, mnist, null);

        // ***** STEP 6: Save the Model *****

        // create directory to save model
        Path modelDir = Paths
                .get("C:/Users/PC/OneDrive/Desktop/AI_Projects/DJL-Learning/introduction_example/build/mlp");
        // create directories if not exist
        Files.createDirectories(modelDir);

        // save the model to the specified directory with the name "mlp"
        model.setProperty("Epoch", String.valueOf(epoch));
        // save model to disk
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
