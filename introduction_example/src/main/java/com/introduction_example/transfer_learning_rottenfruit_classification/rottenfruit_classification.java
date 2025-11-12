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

                String modelUrl = "djl://ai.djl.pytorch/resnet18_embedding";

                Criteria<NDList, NDList> criteria = Criteria.builder()
                                .setTypes(NDList.class, NDList.class)
                                .optModelUrls(modelUrl)
                                .optEngine("PyTorch") // Use PyTorch engine
                                .optProgress(new ProgressBar())
                                .optOption("trainParam", "true") // change to false to freeze the embedding
                                .build();

                ZooModel<NDList, NDList> embedding = criteria.loadModel();

                Block baseBlock = embedding.getBlock();

                Block blocks = new SequentialBlock()
                                .add(baseBlock)
                                .addSingleton(nd -> nd.squeeze(new int[] { 2, 3 }))
                                .add(Linear.builder().setUnits(6).build());

                Model model = Model.newInstance("TransferLearning_RottenFruit");

                model.setBlock(blocks);

                DefaultTrainingConfig config = setupTrainingConfig(baseBlock);

                float lr = 0.001f;

                FixedPerVarTracker.Builder learningRateTrackerBuilder = FixedPerVarTracker.builder()
                                .setDefaultValue(lr);
                for (Pair<String, Parameter> paramPair : baseBlock.getParameters()) {
                        learningRateTrackerBuilder.put(paramPair.getValue().getId(), 0.5f * lr);
                }

                FixedPerVarTracker learningRateTracker = learningRateTrackerBuilder.build();
                Optimizer optimizer = Adam.builder().optLearningRateTracker(learningRateTracker).build();
                config.optOptimizer(optimizer);

                Trainer trainer = model.newTrainer(config);

                int batchSize = 32;
                Shape inputShape = new Shape(batchSize, 3, 224, 224);
                trainer.initialize(inputShape);

                RandomAccessDataset dataSetTrain = getData("train", batchSize);
                RandomAccessDataset dataSetValidate = getData("test", batchSize);
                int numEpoch = 10;

                String SAVE_PATH = "C:\\Users\\PC\\OneDrive\\Desktop\\AI_Projects\\DJL-Learning\\introduction_example\\build\\fruits";
                EasyTrain.fit(trainer, numEpoch, dataSetTrain, dataSetValidate);

                ai.djl.training.TrainingResult result = trainer.getTrainingResult();
                System.out.println("Final Training Loss: " + result.getTrainLoss());
                System.out.println("Final Validation Loss: " + result.getValidateLoss());
                System.out.println("Final Validation Accuracy: " + result.getValidateEvaluation("Accuracy"));

                model.save(Paths.get(SAVE_PATH), "rotten_fruit_classifier");

                model.close();
                embedding.close();

        }

        private static DefaultTrainingConfig setupTrainingConfig(Block baseBlock) {
                String outputDir = "C:\\Users\\PC\\OneDrive\\Desktop\\AI_Projects\\DJL-Learning\\introduction_example\\build\\fruits";
                SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
                listener.setSaveModelCallback((ai.djl.training.Trainer trainer) -> {
                        ai.djl.training.TrainingResult result = trainer.getTrainingResult();
                        Model model = trainer.getModel();
                        float accuracy = result.getValidateEvaluation("Accuracy");
                        model.setProperty("Accuracy", String.format("%.5f", accuracy));
                        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

                DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                                .addEvaluator(new Accuracy())
                                // Change from getDevices(1) to getDevices() to use all available GPUs
                                // Or keep getDevices(1) to use 1 GPU
                                .optDevices(Engine.getInstance().getDevices(1)) // Uses 1 GPU
                                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                                .addTrainingListeners(listener);
                return config;
        }

        private static RandomAccessDataset getData(String usage, int batchSize) throws TranslateException, IOException {
                float[] mean = { 0.485f, 0.456f, 0.406f };
                float[] std = { 0.229f, 0.224f, 0.225f };

                Repository repository = Repository.newInstance("fruit", Paths.get(
                                "C:\\Users\\PC\\OneDrive\\Desktop\\AI_Projects\\DJL-Learning\\introduction_example\\images\\rotten_fresh_fruit\\dataset\\"
                                                + usage));

                FruitsFreshAndRotten.Builder builder = FruitsFreshAndRotten.builder()
                                .optRepository(repository);

                if (usage.equals("train")) {
                        // Apply augmentations BEFORE ToTensor (while still in HWC format)
                        builder.addTransform(new Resize(256, 256))
                                        .addTransform(new RandomResizedCrop(224, 224)) // Moved before ToTensor
                                        .addTransform(new RandomFlipTopBottom()) // Moved before ToTensor
                                        .addTransform(new RandomFlipLeftRight()) // Moved before ToTensor
                                        .addTransform(new ToTensor()) // Now convert to CHW
                                        .addTransform(new Normalize(mean, std));
                } else {
                        // For validation/test: simple resize and crop
                        builder.addTransform(new Resize(256, 256))
                                        .addTransform(new CenterCrop(224, 224))
                                        .addTransform(new ToTensor())
                                        .addTransform(new Normalize(mean, std));
                }

                builder.setSampling(batchSize, true);

                FruitsFreshAndRotten dataset = builder.build();
                dataset.prepare();
                return dataset;
        }

}
