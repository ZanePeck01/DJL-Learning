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

    /**
     * Builds and returns the 1D Convolutional Neural Network used
     * for URL maliciousness classification.
     *
     * model is inspired by the Character-Level CNN architecture
     */

    public static Model createModel() {

        // Create a new model
        Model model = Model.newInstance("malicious_url_detector");

        // sequential block to hold the layers
        SequentialBlock mainBlock = new SequentialBlock();

        // Model hyperparameters
        float dropoutProbability = 0.5f;
        int fullyConnected = 1024;
        int numberOfFilters = 256;

        mainBlock

                // ---- Conv Block 1 ----
                // learns local features from character embeddings (basic patterns/substrings)
                .add(Conv1d.builder()
                        .setKernelShape(new Shape(7)) // window size
                        .setFilters(numberOfFilters) // number of output channels
                        .build())
                .add(Activation.reluBlock()) // add non-linearity
                .add(Pool.maxPool1dBlock(new Shape(3))) // reduce length and fosuc on strong features

                // ---- Conv Block 2 ----
                // helps learn higher-level features (larger patterns/substrings)
                .add(Conv1d.builder()
                        .setKernelShape(new Shape(7))
                        .setFilters(numberOfFilters)
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool1dBlock(new Shape(3)))

                // ---- Conv Block 3–6 ----
                // use smaller kernerls to learn higher-level features
                // more layers help learn more complex patterns (like ".php", "/admin", "%20",
                // etc)
                .add(Conv1d.builder().setKernelShape(new Shape(3)).setFilters(numberOfFilters).build())
                .add(Activation.reluBlock())
                .add(Conv1d.builder().setKernelShape(new Shape(3)).setFilters(numberOfFilters).build())
                .add(Activation.reluBlock())
                .add(Conv1d.builder().setKernelShape(new Shape(3)).setFilters(numberOfFilters).build())
                .add(Activation.reluBlock())
                .add(Conv1d.builder().setKernelShape(new Shape(3)).setFilters(numberOfFilters).build())
                .add(Activation.reluBlock())
                // final pooling to reduce size before FC layers
                .add(Pool.maxPool1dBlock(new Shape(3)))

                // ---- Fully Connected Layers ----
                // flatten converts 3D tensors into single vectors for FC layers
                .add(Blocks.batchFlattenBlock())
                // first FC layer
                .add(Linear.builder().setUnits(fullyConnected).build())
                .add(Activation.reluBlock())
                .add(Dropout.builder().optRate(dropoutProbability).build())
                // second FC layer
                .add(Linear.builder().setUnits(fullyConnected).build())
                .add(Activation.reluBlock())
                .add(Dropout.builder().optRate(dropoutProbability).build())

                // ---- Output (binary classification) ----
                // final output layer with 2 units (malicious or benign)
                .add(Linear.builder().setUnits(2).build());

        // Set the block to the model
        model.setBlock(mainBlock);

        return model;
    }
}
