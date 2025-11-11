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
                .add(Linear.builder().setUnits(6).build())
                .addSingleton(nd -> nd.softmax(1));

        Model model = Model.newInstance("TransferLearning_RottenFruit");

        model.setBlock(blocks);

        DefaultTrainingConfig config = setupTrainingConfig(baseBlock);

        float lr = 0.001f;

        FixedPerVarTracker.Builder learningRateTrackerBuilder = FixedPerVarTracker.builder().setDefaultValue(lr);
        for (Pair<String, Parameter> paramPair : baseBlock.getParameters()) {
            learningRateTrackerBuilder.put(paramPair.getValue().getId(), 0.1f * lr);
        }
        FixedPerVarTracker learningRateTracker = learningRateTrackerBuilder.build();
        Optimizer optimizer = Adam.builder().optLearningRateTracker(learningRateTracker).build();
        config.optOptimizer(optimizer);

        Trainer trainer = model.newTrainer(config);

        int batchSize = 64;
        Shape inputShape = new Shape(batchSize, 3, 224, 224);
        trainer.initialize(inputShape);

        RandomAccessDataset dataSetTrain = getData("train", batchSize);
        RandomAccessDataset dataSetValidate = getData("test", batchSize);
        int numEpoch = 2;
        String SAVE_PATH = "build/fruits_model";

        EasyTrain.fit(trainer, numEpoch, dataSetTrain, dataSetValidate);
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
                .optDevices(Engine.getEngine("PyTorch").getDevices(1))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
        return config;
    }

    private static RandomAccessDataset getData(String usage, int batchSize) throws TranslateException, IOException {
        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        // usage is either "train" or "test"
        Repository repository = Repository.newInstance("fruit", Paths.get(
                "C:\\Users\\PC\\OneDrive\\Desktop\\AI_Projects\\DJL-Learning\\introduction_example\\images\\rotten_fresh_fruit\\dataset\\"
                        + usage));
        FruitsFreshAndRotten dataset = FruitsFreshAndRotten.builder()
                .optRepository(repository)
                .addTransform(new RandomResizedCrop(256, 256)) // only in training
                .addTransform(new RandomFlipTopBottom()) // only in training
                .addTransform(new RandomFlipLeftRight()) // only in training
                .addTransform(new Resize(256, 256))
                .addTransform(new CenterCrop(224, 224))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(mean, std))
                .addTargetTransform(new OneHot(6))
                .setSampling(batchSize, true)
                .build();
        dataset.prepare();
        return dataset;
    }

}
