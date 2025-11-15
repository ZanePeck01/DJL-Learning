package com.introduction_example.malicious_url_tracker;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv1d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

public class model {

    public static Model createModel() {

        Model model = Model.newInstance("malicious_url_detector");

        SequentialBlock mainBlock = new SequentialBlock();
        float dropoutProbability = 0.5f;
        int fullyConnected = 1024;
        int numberOfFilters = 256;

        mainBlock

                // ---- Conv Block 1 ----
                .add(Conv1d.builder()
                        .setKernelShape(new Shape(7))
                        .setFilters(numberOfFilters)
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool1dBlock(new Shape(3)))

                // ---- Conv Block 2 ----
                .add(Conv1d.builder()
                        .setKernelShape(new Shape(7))
                        .setFilters(numberOfFilters)
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool1dBlock(new Shape(3)))

                // ---- Conv Block 3–6 ----
                .add(Conv1d.builder().setKernelShape(new Shape(3)).setFilters(numberOfFilters).build())
                .add(Activation.reluBlock())
                .add(Conv1d.builder().setKernelShape(new Shape(3)).setFilters(numberOfFilters).build())
                .add(Activation.reluBlock())
                .add(Conv1d.builder().setKernelShape(new Shape(3)).setFilters(numberOfFilters).build())
                .add(Activation.reluBlock())
                .add(Conv1d.builder().setKernelShape(new Shape(3)).setFilters(numberOfFilters).build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool1dBlock(new Shape(3)))

                // ---- FC Layers ----
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(fullyConnected).build())
                .add(Activation.reluBlock())
                .add(Dropout.builder().optRate(dropoutProbability).build())

                .add(Linear.builder().setUnits(fullyConnected).build())
                .add(Activation.reluBlock())
                .add(Dropout.builder().optRate(dropoutProbability).build())

                // ---- Output (binary classification) ----
                .add(Linear.builder().setUnits(2).build());

        model.setBlock(mainBlock);

        return model;
    }
}
