package com.introduction_example;

import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;

//Nueral Network Example (for image classification)

public class Main {
    public static void main(String[] args) {

        // define application type, use image classification for this example
        Application application = Application.CV.IMAGE_CLASSIFICATION;

        // input size is 28*28 for MNIST dataset (will faltten them into 1D array)
        long intputSize = 28 * 28;
        // output size is 10 for 10 possible classes the image classify as
        long outputSize = 10;

        // create sequential block that holds layers of nn and executes them in
        // sequecial order (simiilar to pipeline, input flows to output)
        SequentialBlock block = new SequentialBlock();

        /*
         * This is the Seqeuntial Block:
         * SequentialBlock {
         * batchFlatten (converts 28*28 image into 1D array of 784-element vector)
         * Linear (hidden layer with 128 neurons)
         * LambdaBlock (ReLu activation function)
         * Linear (hidden layer with 64 neurons)
         * LambdaBlock (ReLu activation function)
         * Linear (output layer with 10 neurons)
         * }
         */
        block.add(Blocks.batchFlattenBlock(intputSize));
        block.add(Linear.builder().setUnits(128).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(64).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(outputSize).build());

    }
}