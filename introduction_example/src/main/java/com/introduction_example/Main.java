package com.introduction_example;

import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;

//Nueral Network Example

public class Main {
    public static void main(String[] args) {

        Application application = Application.CV.IMAGE_CLASSIFICATION;

        long intputSize = 28 * 28;
        long outputSize = 10;

        SequentialBlock block = new SequentialBlock();

        /*
         * This is the Seqeuntial Block:
         * SequentialBlock {
         * batchFlatten
         * Linear
         * LambdaBlock
         * Linear
         * LambdaBlock
         * Linear
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