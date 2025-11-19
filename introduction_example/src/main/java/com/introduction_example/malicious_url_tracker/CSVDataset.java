package com.introduction_example.malicious_url_tracker;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Custom dataset for loading URL → label pairs from a CSV file.
 * Converts each URL into a character-level one-hot encoded tensor
 * and labels into numeric values (0 = good, 1 = malicious).
 */

public class CSVDataset extends RandomAccessDataset {

    // each url is encoded into a fixed length of 1014 characters
    private static final int FEATURE_LENGTH = 1014;
    // all possible characters in the URL encoding
    private static final String ALL_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}";

    //
    private List<Character> alphabets;
    private Map<Character, Integer> alphabetsIndex;
    private List<CSVRecord> dataset;
    private Usage usage;

    /**
     * Constructor used by Builder.
     * Initializes the character alphabet encoding map.
     */
    protected CSVDataset(Builder builder) {
        super(builder);
        this.usage = builder.usage;
        this.dataset = builder.dataset;

        // Initialize character encoding
        alphabets = new ArrayList<>();
        alphabetsIndex = new HashMap<>();

        // for each character, map it to an index
        for (char c : ALL_CHARS.toCharArray()) {
            alphabetsIndex.put(c, alphabets.size());
            alphabets.add(c);
        }
    }

    /**
     * Returns the encoded (data, label) pair for a specific row.
     */
    @Override
    public Record get(NDManager manager, long index) {

        // containers for input features and labels
        NDList datum = new NDList();
        NDList label = new NDList();

        // Get the CSV record at the specified index
        CSVRecord record = dataset.get(Math.toIntExact(index));

        // Get a single data, label pair, encode them using helpers into correct shape
        datum.add(encodeData(manager, record.get("url")));
        label.add(encodeLabel(manager, record.get("isMalicious")));

        return new Record(datum, label);
    }

    /**
     * Number of available samples in the dataset.
     */
    @Override
    protected long availableSize() {
        return dataset.size();
    }

    /**
     * Converts a URL string into a character-level one-hot matrix.
     * Each character gets its own column position.
     */
    private NDArray encodeData(NDManager manager, String url) {

        // create a large zero matrix of shape [alphabet_size, FEATURE_LENGTH]
        NDArray encoded = manager.zeros(new Shape(alphabets.size(), FEATURE_LENGTH));

        // Convert to lowercase
        char[] arrayText = url.toLowerCase().toCharArray();

        // loop through each character in the URL
        for (int i = 0; i < url.length(); i++) {

            // stop if we reach max feature length
            if (i >= FEATURE_LENGTH) {
                break;
            }
            // set the corresponding position to 1 for one-hot encoding
            if (alphabetsIndex.containsKey(arrayText[i])) {
                encoded.set(new NDIndex(alphabetsIndex.get(arrayText[i]), i), 1);
            }
        }
        return encoded;
    }

    /**
     * Converts label text into a numeric value.
     * "bad" → 1.0
     * "good" → 0.0
     */
    private NDArray encodeLabel(NDManager manager, String label) {
        // Convert "bad" to 1, "good" to 0
        float labelValue = label.trim().equalsIgnoreCase("bad") ? 1.0f : 0.0f;
        return manager.create(labelValue);
    }

    @Override
    public void prepare(Progress progress) {
        // No preparation needed for this dataset
    }

    /**
     * Builder class for CSVDataset
     */
    public static final class Builder extends BaseBuilder<Builder> {

        private Usage usage;
        private List<CSVRecord> dataset;
        private String csvFileLocation;

        /**
         * Constructs a new builder.
         */
        public Builder() {
            this.usage = Usage.TRAIN;
            this.csvFileLocation = "C:\\Users\\PC\\OneDrive\\Desktop\\AI_Projects\\url_data_mega_deep_learning.csv";
        }

        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Set TRAIN, TEST, or VALIDATION split.
         */
        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /**
         * Sets the path to the CSV file.
         */
        public Builder setCsvFile(String csvFileLocation) {
            this.csvFileLocation = csvFileLocation;
            return this;
        }

        /**
         * Builds the CSVDataset object
         */
        public CSVDataset build() throws IOException {
            // Read the CSV file
            try (Reader reader = Files.newBufferedReader(Paths.get(csvFileLocation));
                    CSVParser csvParser = new CSVParser(
                            reader,
                            CSVFormat.DEFAULT
                                    .withHeader("url", "isMalicious")
                                    .withFirstRecordAsHeader()
                                    .withIgnoreHeaderCase()
                                    .withTrim())) {

                // Load all records from the CSV
                List<CSVRecord> csvRecords = csvParser.getRecords();

                // Split dataset 80/20 for train/test
                int splitIndex = (int) (csvRecords.size() * 0.8);

                // Select records based on usage
                switch (usage) {
                    case TRAIN:
                        dataset = csvRecords.subList(0, splitIndex);
                        break;
                    case TEST:
                        dataset = csvRecords.subList(splitIndex, csvRecords.size());
                        break;
                    case VALIDATION:
                    default:
                        dataset = csvRecords;
                        break;
                }

                return new CSVDataset(this);
            }
        }
    }
}
