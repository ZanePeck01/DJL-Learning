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

public class CSVDataset extends RandomAccessDataset {

    private static final int FEATURE_LENGTH = 1014;
    private static final String ALL_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}";

    private List<Character> alphabets;
    private Map<Character, Integer> alphabetsIndex;
    private List<CSVRecord> dataset;
    private Usage usage;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given
     * necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    protected CSVDataset(Builder builder) {
        super(builder);
        this.usage = builder.usage;
        this.dataset = builder.dataset;

        // Initialize character encoding
        alphabets = new ArrayList<>();
        alphabetsIndex = new HashMap<>();

        for (char c : ALL_CHARS.toCharArray()) {
            alphabetsIndex.put(c, alphabets.size());
            alphabets.add(c);
        }
    }

    /**
     * Gets the {@link Record} for the given index from the dataset.
     *
     * @param manager the manager used to create the arrays
     * @param index   the index of the requested data item
     * @return a {@link Record} that contains the data and label of the requested
     *         data item
     */
    @Override
    public Record get(NDManager manager, long index) {
        NDList datum = new NDList();
        NDList label = new NDList();

        CSVRecord record = dataset.get(Math.toIntExact(index));

        // Get a single data, label pair, encode them using helpers
        datum.add(encodeData(manager, record.get("url")));
        label.add(encodeLabel(manager, record.get("isMalicious")));

        return new Record(datum, label);
    }

    /**
     * Returns the number of records available to be read in this {@code Dataset}.
     *
     * @return the number of records available to be read in this {@code Dataset}
     */
    @Override
    protected long availableSize() {
        return dataset.size();
    }

    /**
     * Encodes the URL text using character-level one-hot encoding
     * Based on: Character-level Convolutional Networks for Text Classification
     * https://arxiv.org/abs/1509.01626
     * 
     * @param manager the manager to create the NDArray
     * @param url     the URL string to encode
     * @return encoded NDArray of shape [alphabet_size, FEATURE_LENGTH]
     */
    private NDArray encodeData(NDManager manager, String url) {
        NDArray encoded = manager.zeros(new Shape(alphabets.size(), FEATURE_LENGTH));
        char[] arrayText = url.toLowerCase().toCharArray();

        for (int i = 0; i < url.length(); i++) {
            if (i >= FEATURE_LENGTH) {
                break;
            }
            if (alphabetsIndex.containsKey(arrayText[i])) {
                encoded.set(new NDIndex(alphabetsIndex.get(arrayText[i]), i), 1);
            }
        }
        return encoded;
    }

    /**
     * Encodes the label (0 for good, 1 for bad/malicious)
     * 
     * @param manager the manager to create the NDArray
     * @param label   the label string ("good" or "bad")
     * @return encoded label as NDArray
     */
    private NDArray encodeLabel(NDManager manager, String label) {
        // Convert "bad" to 1, "good" to 0
        float labelValue = label.trim().equalsIgnoreCase("bad") ? 1.0f : 0.0f;
        return manager.create(labelValue);
    }

    /**
     * Prepares the dataset for use with optional data processing.
     *
     * @param progress the progress tracker
     */
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

        /**
         * {@inheritDoc}
         */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the optional usage for the dataset.
         *
         * @param usage the usage (TRAIN or TEST)
         * @return this builder
         */
        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /**
         * Sets the path to the CSV file.
         *
         * @param csvFileLocation the path to the CSV file
         * @return this builder
         */
        public Builder setCsvFile(String csvFileLocation) {
            this.csvFileLocation = csvFileLocation;
            return this;
        }

        /**
         * Builds the {@link CSVDataset}.
         *
         * @return the {@link CSVDataset}
         * @throws IOException if the CSV file cannot be read
         */
        public CSVDataset build() throws IOException {
            try (Reader reader = Files.newBufferedReader(Paths.get(csvFileLocation));
                    CSVParser csvParser = new CSVParser(
                            reader,
                            CSVFormat.DEFAULT
                                    .withHeader("url", "isMalicious")
                                    .withFirstRecordAsHeader()
                                    .withIgnoreHeaderCase()
                                    .withTrim())) {

                List<CSVRecord> csvRecords = csvParser.getRecords();

                // Split dataset 80/20 for train/test
                int splitIndex = (int) (csvRecords.size() * 0.8);

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
