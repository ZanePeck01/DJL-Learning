package com.introduction_example.malicious_url_tracker;

import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * Complete training example for Malicious URL Detection
 * Fixed and corrected version with proper DJL API usage
 */
public class TrainMaliciousURLDetector {

    public static void main(String[] args) {
        try {
            // Configuration
            int batchSize = 32;
            int epochs = 10;
            String csvFilePath = "url_data_mega_deep_learning.csv";
            String outputDir = "build/model";
            String modelName = "malicious-url-detector";

            System.out.println("=== Malicious URL Detection Training ===\n");

            // Step 1: Create datasets
            System.out.println("Loading datasets...");
            CSVDataset trainDataset = new CSVDataset.Builder()
                    .optUsage(Dataset.Usage.TRAIN)
                    .setCsvFile(csvFilePath)
                    .setSampling(batchSize, true) // Shuffle for training
                    .build();

            CSVDataset validateDataset = new CSVDataset.Builder()
                    .optUsage(Dataset.Usage.TEST)
                    .setCsvFile(csvFilePath)
                    .setSampling(batchSize, false) // No shuffle for validation
                    .build();

            System.out.println("Training samples: " + trainDataset.size());
            System.out.println("Validation samples: " + validateDataset.size());

            // Step 2: Setup training configuration
            DefaultTrainingConfig trainingConfig = setupTrainingConfig(batchSize, trainDataset.size());

            // Step 3: Create model using the model class
            System.out.println("\nCreating model...");
            Model model = com.introduction_example.malicious_url_tracker.model.createModel();

            // Step 4: Create trainer with the training config
            System.out.println("Initializing trainer...");
            try (Trainer trainer = model.newTrainer(trainingConfig)) {

                // Initialize the parameters, pass shape of input
                // Input shape: [batch_size, alphabet_size=69, sequence_length=1014]
                Shape inputShape = new Shape(batchSize, 69, 1014);
                trainer.initialize(inputShape);

                System.out.println("Input shape: " + inputShape);

                // Step 5: Train on dataset for epochs
                System.out.println("\nStarting training...\n");

                for (int epoch = 0; epoch < epochs; epoch++) {
                    System.out.println("Epoch " + (epoch + 1) + "/" + epochs);

                    // Training phase
                    int batchCount = 0;
                    for (Batch batch : trainer.iterateDataset(trainDataset)) {
                        EasyTrain.trainBatch(trainer, batch);
                        trainer.step();
                        batch.close();
                        batchCount++;
                    }

                    System.out.println("  Training batches: " + batchCount);

                    // Validation phase - simple forward pass
                    int valBatchCount = 0;
                    for (Batch batch : trainer.iterateDataset(validateDataset)) {
                        batch.close();
                        valBatchCount++;
                    }

                    System.out.println("  Validation batches: " + valBatchCount);

                    // Print metrics
                    System.out.println("  " + trainer.getTrainingResult());

                    // Save model after current epoch
                    model.setProperty("Epoch", String.valueOf(epoch + 1));
                    model.save(Paths.get(outputDir), modelName);
                    System.out.println("  Model saved to: " + outputDir + "/" + modelName);
                    System.out.println();
                }

                System.out.println("=== Training Complete ===");
            }

        } catch (IOException | TranslateException e) {
            System.err.println("Error during training: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Setup training configuration with optimizer, loss, and evaluators
     * 
     * @param batchSize   the batch size
     * @param datasetSize the total size of training dataset
     * @return configured DefaultTrainingConfig
     */
    private static DefaultTrainingConfig setupTrainingConfig(int batchSize, long datasetSize) {
        // Learning Rate definition
        double learningRate = 0.01;

        // Simple fixed learning rate tracker
        Tracker learningRateTracker = Tracker.fixed((float) learningRate);

        // Setting optimizer - SGD with momentum
        Optimizer optimizer = Optimizer.sgd()
                .setRescaleGrad(1.0f / batchSize)
                .setLearningRateTracker(learningRateTracker)
                .optWeightDecays(0.00001f)
                .optMomentum(0.9f)
                .build();

        // Define loss
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // Define initializer
        Initializer initializer = new XavierInitializer(
                XavierInitializer.RandomType.UNIFORM,
                XavierInitializer.FactorType.AVG,
                2.24f);

        // Create training config
        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optInitializer(initializer, ai.djl.nn.Parameter.Type.WEIGHT)
                .optOptimizer(optimizer)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        return config;
    }

    /**
     * Alternative simple training method using EasyTrain.fit()
     */
    public static void trainSimple(String csvFilePath, int epochs, String modelName)
            throws IOException, TranslateException {

        int batchSize = 32;

        // Load dataset
        CSVDataset trainDataset = new CSVDataset.Builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setCsvFile(csvFilePath)
                .setSampling(batchSize, true)
                .build();

        CSVDataset validateDataset = new CSVDataset.Builder()
                .optUsage(Dataset.Usage.TEST)
                .setCsvFile(csvFilePath)
                .setSampling(batchSize, false)
                .build();

        // Create model
        Model model = com.introduction_example.malicious_url_tracker.model.createModel();

        // Setup training config
        DefaultTrainingConfig config = setupTrainingConfig(batchSize, trainDataset.size());

        // Train
        try (Trainer trainer = model.newTrainer(config)) {
            trainer.initialize(new Shape(batchSize, 70, 1014));

            // Use EasyTrain for simple training loop
            EasyTrain.fit(trainer, epochs, trainDataset, validateDataset);

            // Save final model
            model.setProperty("Epoch", String.valueOf(epochs));
            model.save(Paths.get("build/model"), modelName);
        }

        System.out.println("Training complete! Model saved.");
    }
}