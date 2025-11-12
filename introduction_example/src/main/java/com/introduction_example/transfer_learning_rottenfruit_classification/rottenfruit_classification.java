package com.introduction_example.transfer_learning_rottenfruit_classification;

import java.io.IOException;
import java.nio.file.Paths;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FruitsFreshAndRotten;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.OneHot;
import ai.djl.modality.cv.transform.RandomFlipLeftRight;
import ai.djl.modality.cv.transform.RandomFlipTopBottom;
import ai.djl.modality.cv.transform.RandomResizedCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.FixedPerVarTracker;
import ai.djl.training.evaluator.Accuracy;

public class rottenfruit_classification {

        public static void main(String[] args)
                        throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {

                // Step 1: Load a pretrained ResNet18 model from DJL's model zoo
                // This model was trained on ImageNet and knows how to extract features from
                // images
                String modelUrl = "djl://ai.djl.pytorch/resnet18_embedding";

                // build the criteria to load the model
                Criteria<NDList, NDList> criteria = Criteria.builder()
                                .setTypes(NDList.class, NDList.class)
                                .optModelUrls(modelUrl)
                                .optEngine("PyTorch") // Use PyTorch engine
                                .optProgress(new ProgressBar())
                                .optOption("trainParam", "true") // allow pretrained weights to be updated (fine-tuning)
                                .build();

                // load pretrained model
                ZooModel<NDList, NDList> embedding = criteria.loadModel();

                // get the nn strcture (block) from the pretrained model
                Block baseBlock = embedding.getBlock();

                // Step 2: Modify the model for the specific classification task
                Block blocks = new SequentialBlock()
                                .add(baseBlock) // start with the pretrained model
                                .addSingleton(nd -> nd.squeeze(new int[] { 2, 3 })) // squeeze to remove extra
                                                                                    // dimensions
                                .add(Linear.builder().setUnits(6).build()); // add a new final layer for 6 classes
                                                                            // (fresh and rotten for 3 types of fruit)
                // Step 3: create a new model with the modified block
                Model model = Model.newInstance("TransferLearning_RottenFruit");

                model.setBlock(blocks);

                // Step 4: Setup Training Configurations
                DefaultTrainingConfig config = setupTrainingConfig(baseBlock);

                // Step 5: Configure discriminative learning rates
                // The idea: pretrained layers learn slowly, new layers learn fast
                float lr = 0.001f; // base learning rate for new layers

                FixedPerVarTracker.Builder learningRateTrackerBuilder = FixedPerVarTracker.builder()
                                .setDefaultValue(lr); // new layers get full learning rate (not pretrained layers)

                // loop through all parameters in the base block (pretrained layers)
                for (Pair<String, Parameter> paramPair : baseBlock.getParameters()) {
                        // set learning rate for pretrained layers to half the base learning rate
                        // this prevents them from "forgetting" what they learned on ImageNet
                        learningRateTrackerBuilder.put(paramPair.getValue().getId(), 0.5f * lr);
                }

                // build the learning rate tracker and set it in the optimizer (Adam in this
                // case)
                FixedPerVarTracker learningRateTracker = learningRateTrackerBuilder.build();
                Optimizer optimizer = Adam.builder().optLearningRateTracker(learningRateTracker).build();
                config.optOptimizer(optimizer);

                // Step 6: Create Trainer and Initialize
                Trainer trainer = model.newTrainer(config);

                // INitizale trainer with proper input shape
                int batchSize = 32;
                // [batch_size, channels, height, width] = [32, 3, 224, 224]
                Shape inputShape = new Shape(batchSize, 3, 224, 224);
                trainer.initialize(inputShape);

                // Step 7: Load Data
                RandomAccessDataset dataSetTrain = getData("train", batchSize); // training data with augmentations
                RandomAccessDataset dataSetValidate = getData("test", batchSize);// validation data without
                                                                                 // augmentations
                int numEpoch = 10; // 10 complete passes through data

                // Step 8: Train the model
                String SAVE_PATH = "C:\\Users\\PC\\OneDrive\\Desktop\\AI_Projects\\DJL-Learning\\introduction_example\\build\\fruits";
                EasyTrain.fit(trainer, numEpoch, dataSetTrain, dataSetValidate);

                // Step 9: Save the model and print final results
                ai.djl.training.TrainingResult result = trainer.getTrainingResult();
                System.out.println("Final Training Loss: " + result.getTrainLoss());
                System.out.println("Final Validation Loss: " + result.getValidateLoss());
                System.out.println("Final Validation Accuracy: " + result.getValidateEvaluation("Accuracy"));

                model.save(Paths.get(SAVE_PATH), "rotten_fruit_classifier");

                // Close resources
                model.close();
                embedding.close();

        }

        /**
         * Sets up the training configuration including:
         * - Loss function (how to measure error)
         * - Metrics (what to track during training)
         * - Device (CPU or GPU)
         * - Listeners (for logging and saving models)
         */
        private static DefaultTrainingConfig setupTrainingConfig(Block baseBlock) {
                String outputDir = "C:\\Users\\PC\\OneDrive\\Desktop\\AI_Projects\\DJL-Learning\\introduction_example\\build\\fruits";
                // create a listener to save the best model based on validation accuracy
                SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
                listener.setSaveModelCallback((ai.djl.training.Trainer trainer) -> {
                        ai.djl.training.TrainingResult result = trainer.getTrainingResult();
                        Model model = trainer.getModel();
                        float accuracy = result.getValidateEvaluation("Accuracy");
                        model.setProperty("Accuracy", String.format("%.5f", accuracy));
                        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

                // setup training configuration
                DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                                // this loss expects raw logits, so no softmax activation in the model
                                .addEvaluator(new Accuracy())
                                // Change from getDevices(1) to getDevices() to use all available GPUs
                                // Or keep getDevices(1) to use 1 GPU
                                .optDevices(Engine.getInstance().getDevices(1)) // Uses 1 GPU
                                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                                .addTrainingListeners(listener); // save models during training
                return config;
        }

        /**
         * Loads and prepares the fruit dataset with appropriate transformations
         * 
         * @param usage     Either "train" or "test" to determine which dataset split to
         *                  load
         * @param batchSize How many images to process at once
         */

        private static RandomAccessDataset getData(String usage, int batchSize) throws TranslateException, IOException {
                // ImageNet normalization values - these are standard for ResNet models
                // They normalize the image so pixel values have similar statistics to ImageNet
                float[] mean = { 0.485f, 0.456f, 0.406f }; // RGB mean
                float[] std = { 0.229f, 0.224f, 0.225f }; // RGB standard deviation

                // where the dataset is located
                Repository repository = Repository.newInstance("fruit", Paths.get(
                                "C:\\Users\\PC\\OneDrive\\Desktop\\AI_Projects\\DJL-Learning\\introduction_example\\images\\rotten_fresh_fruit\\dataset\\"
                                                + usage));

                FruitsFreshAndRotten.Builder builder = FruitsFreshAndRotten.builder()
                                .optRepository(repository);

                if (usage.equals("train")) {
                        // training data gets AUGMENTATION to create variations and prevent overfitting
                        builder.addTransform(new Resize(256, 256)) // Resize first
                                        .addTransform(new RandomResizedCrop(224, 224)) // Then random crop (zoom/scale
                                                                                       // varaition)
                                        .addTransform(new RandomFlipTopBottom()) // random vertical flip (50% chance)
                                        .addTransform(new RandomFlipLeftRight()) // random horizontal flip (50% chance)
                                        .addTransform(new ToTensor()) // convert from image to tensor
                                        .addTransform(new Normalize(mean, std)); // normalize to ImageNet stats
                } else {
                        // validation/test data gets NO augmentation
                        builder.addTransform(new Resize(256, 256)) // resize first
                                        .addTransform(new CenterCrop(224, 224)) // then center crop
                                        .addTransform(new ToTensor()) // convert from image to tensor
                                        .addTransform(new Normalize(mean, std)); // normalize to ImageNet stats
                }

                // set sampling to random batches of sizee 32
                builder.setSampling(batchSize, true);

                // build and prepare the dataset
                FruitsFreshAndRotten dataset = builder.build();
                dataset.prepare(); // load the data
                return dataset;
        }

}
