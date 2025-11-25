package com.movie_recommender;

import ai.djl.nn.Block;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.modality.nlp.Vocabulary;

import java.util.List;

public class model {

        public static Block getModel(Vocabulary userVocab, Vocabulary movieVocab, int embedSize) {

                Block userEmbedding = new TrainableWordEmbedding(userVocab, embedSize);
                Block movieEmbedding = new TrainableWordEmbedding(movieVocab, embedSize);

                Block embeddingLayer = new ParallelBlock(
                                list -> new ai.djl.ndarray.NDList(
                                                list.get(0).singletonOrThrow(),
                                                list.get(1).singletonOrThrow()),
                                List.of(userEmbedding, movieEmbedding));

                return new SequentialBlock()
                                .add(embeddingLayer)
                                .add(Blocks.batchFlattenBlock())
                                .add(Linear.builder().setUnits(128).build())
                                .add(Activation::relu)
                                .add(Dropout.builder().optRate(0.3f).build())
                                .add(Linear.builder().setUnits(64).build())
                                .add(Activation::relu)
                                .add(Linear.builder().setUnits(32).build())
                                .add(Activation::relu)
                                .add(Linear.builder().setUnits(1).build());
        }
}