package com.introduction_example.malicious_url_tracker;

import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toMap;

/**
 * URLTranslator for inference
 * Handles preprocessing (encoding) of URLs and postprocessing (classification)
 * of model output
 */
public class CSVTranslator implements Translator<String, Classifications> {

    private static final int FEATURE_LENGTH = 1014;
    private static final String ALL_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}";

    private List<Character> alphabets;
    private Map<Character, Integer> alphabetsIndex;

    /**
     * Constructor - initialize character encoding
     */
    public CSVTranslator() {
        // Create list of characters
        alphabets = ALL_CHARS.chars()
                .mapToObj(e -> (char) e)
                .collect(Collectors.toList());

        // Create index map for quick lookup
        alphabetsIndex = IntStream.range(0, alphabets.size())
                .boxed()
                .collect(toMap(alphabets::get, i -> i));
    }

    /**
     * processInput encodes the input URL string to a [69, 1014] NDArray
     * Works like the training data encoder
     */
    @Override
    public NDList processInput(TranslatorContext ctx, String url) {
        // One-hot encode the text to an array initialized to zeros
        NDArray encoded = ctx.getNDManager().zeros(new Shape(alphabets.size(), FEATURE_LENGTH));

        // Convert to lowercase for consistency
        char[] arrayText = url.toLowerCase().toCharArray();

        for (int i = 0; i < url.length(); i++) {
            if (i >= FEATURE_LENGTH) {
                break;
            }
            if (alphabetsIndex.containsKey(arrayText[i])) {
                encoded.set(new NDIndex(alphabetsIndex.get(arrayText[i]), i), 1);
            }
        }

        return new NDList(encoded);
    }

    /**
     * Converts the output NDArray (classification logits) to Classifications object
     */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray array = list.singletonOrThrow();

        // Apply softmax to get probabilities
        NDArray probabilities = array.softmax(0);

        // Define class labels
        List<String> classNames = new ArrayList<>();
        classNames.add("benign");
        classNames.add("malicious");

        return new Classifications(classNames, probabilities);
    }

    /**
     * Gets the batchifier for this translator
     */
    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
}