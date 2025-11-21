package com.movie_recommender;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class movies_dataset {

    // container class that represents a single rating: user, movie, rating
    // will be used as inputs to the model

    public static class Rating {
        public int user;
        public int movie;
        public float rating; // 0.0-5.0

        public Rating(int user, int movie, float rating) {
            this.user = user;
            this.movie = movie;
            this.rating = rating;
        }
    }

    // stores all ratings from the dataset
    private List<Rating> ratings = new ArrayList<>();

    // number of unique users and movies
    private int numUsers;
    private int numMovies;

    // constructor that loads the dataset from a CSV file (in this case from
    // MovieLens 9000)
    public movies_dataset(String csv_path) throws IOException {

        // maps to convert original user and movie IDs to contiguous indices
        Map<Integer, Integer> userMap = new HashMap<>();
        Map<Integer, Integer> movieMap = new HashMap<>();

        // indices for new user and movie IDs
        int idxUser = 0;
        int idxMovie = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(csv_path))) {
            // ignore header line (not actual data needed)
            String line = br.readLine();

            // processs each line of the CSV
            while ((line = br.readLine()) != null) {
                // CSV format: userId,movieId,rating,timestamp
                String[] values = line.split(",");

                // parse user ID, movie ID, and rating value
                int userId = Integer.parseInt(values[0]);
                int movieId = Integer.parseInt(values[1]);
                float ratingValue = Float.parseFloat(values[2]);

                // if user or movie is new, add to map with new index
                userMap.putIfAbsent(userId, idxUser++);
                movieMap.putIfAbsent(movieId, idxMovie++);

                // add rating to the list with mapped indices
                ratings.add(new Rating(userMap.get(userId), movieMap.get(movieId), ratingValue));
            }
        }

        // store the number of unique users and movies
        numUsers = userMap.size();
        numMovies = movieMap.size();
    }

    // getter methods
    public List<Rating> getRatings() {
        return ratings;
    }

    public int getNumUsers() {
        return numUsers;
    }

    public int getNumMovies() {
        return numMovies;
    }

}
