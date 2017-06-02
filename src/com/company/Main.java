package com.company;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonWriter;
import opennlp.tools.doccat.*;
import opennlp.tools.ml.naivebayes.NaiveBayesTrainer;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.*;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;


public class Main {
    private static final String TRAIN_DATA_FILE_PATH = "./train.txt";
    private static final String CELE_LIST_FILE_PATH = "./tmp.txt";
    private static final String TV_REVIEWS_DIR_PATH = "./reviews";
    private static JsonWriter writer = null;
    private static Map<String, String> reviewsMap = null;
    private static Date date = null;
    private static DoccatModel model;

    public static void main(String[] args) {
        trainModel();

        File dir = new File(TV_REVIEWS_DIR_PATH);
        if (!(dir.exists() && dir.isDirectory())) {
            boolean successful = dir.mkdir();
            if (successful) {
                System.out.println("directory was created successfully");
            } else {
                System.out.println("failed trying to create the directory");
            }
        }

        reviewsMap = new HashMap<String, String>();
        date = new Date();
        PrintStream out = null;

        try {
            out = new PrintStream(new FileOutputStream("output.txt"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        System.setOut(out);

        try (BufferedReader br = new BufferedReader(new FileReader(CELE_LIST_FILE_PATH))) {
            String line;

            while ((line = br.readLine()) != null) {
                String[] tmp = line.split(",");
                System.out.println(String.format("\n\n%s, %s:", tmp[0], tmp[1]));
                int[] results = getCeleReviewPoints(tmp[0], tmp[1]);
                System.out.println(String.format("positive point: %d, negative point: %d", results[0], results[1]));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void trainModel() {
        InputStream dataIn = null;

        try {
            MarkableFileInputStreamFactory factory = new MarkableFileInputStreamFactory(new File("./traindata.txt"));
            ObjectStream<String> lineStream = new PlainTextByLineStream(factory, "UTF-8");
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

            /*TrainingParameters params = new TrainingParameters();
            params.put(TrainingParameters.CUTOFF_PARAM, Integer.toString(3));
            params.put(TrainingParameters.ALGORITHM_PARAM, NaiveBayesTrainer.NAIVE_BAYES_VALUE);*/

            model = DocumentCategorizerME.train("en", sampleStream, TrainingParameters.defaultParams(), new DoccatFactory());
            //model = DocumentCategorizerME.train("en", sampleStream, params, new DoccatFactory());

            /*OutputStream modelOut = null;
            File modelFileTmp = new File("sentimentModel.bin");
            modelOut = new BufferedOutputStream(new FileOutputStream(modelFileTmp));
            model.serialize(modelOut);*/
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int[] getCeleReviewPoints(String name, String attr) {
        int results[] = new int[2];

            try {
                String personId = getCeleId(name);
                String[] worksIds = getCeleWorksId(personId, attr);

                for (String workId : worksIds) {
                    String searchData = "";
                    // get reviews if memory if exists
                    if (reviewsMap.containsKey(workId)) {
                        searchData = reviewsMap.get(workId);
                    } else {
                        searchData = getReviewRec(workId);
                        reviewsMap.put(workId, searchData);
                    }

                    JsonParser parser = new JsonParser();
                    JsonObject jsonTree = parser.parse(searchData).getAsJsonObject();
                    JsonArray comments = jsonTree.get("data").getAsJsonObject().get("user_comments").getAsJsonArray();

                    for (JsonElement je : comments) {
                        JsonObject commentObj = je.getAsJsonObject();

                        int userScore = 1;
                        // filter the potential fake review, only select the review with more than 50% find it useful
                        if (commentObj.get("user_score") != null) {
                            userScore = commentObj.get("user_score").getAsInt();
                            if (commentObj.get("user_score").getAsInt() * 2 < commentObj.get("user_score_count").getAsInt()) {
                                continue;
                            }
                        }

                        // deal with the comment content
                        String commentContent = commentObj.get("text").getAsString();
                        String fname = name.substring(0, name.indexOf(" "));
                        String lname = name.substring(name.lastIndexOf(" ") + 1);
                        if (commentContent.toLowerCase().contains(name.toLowerCase()) ||
                                commentContent.toLowerCase().contains(fname.toLowerCase()) ||
                                commentContent.toLowerCase().contains(lname.toLowerCase())) {
                            int[] times = getPosNegTimes(commentContent, fname, lname);
                            // positive
                            results[0] += userScore * times[0];
                            System.out.println(String.format("Debug: userScore: %d, positive: %d, negative: %d", userScore, times[0], times[1]));
                            // negative
                            results[1] += userScore * times[1];
                        }
                    }
                }
            } catch(Exception ex){
                ex.printStackTrace();
            }
        return results;
    }

    private static String getCeleId(String name) {
        String id = "";
        String query = name.replace(' ', '+');
        String requestPath = String.format("http://www.imdb.com/xml/find?q=%s&json=1&nr=1&nn=on", query);

        URL url = null;
        while(id.length() == 0) {
            try {
                url = new URL(requestPath);
                System.out.println(url);
                BufferedReader br = new BufferedReader(new InputStreamReader(url.openStream()));
                String searchData = "";
                String strTemp = "";

                while (null != (strTemp = br.readLine())) {
                    searchData += strTemp;
                }
                System.out.println(searchData);

                JsonParser parser = new JsonParser();
                JsonObject jsonTree = parser.parse(searchData).getAsJsonObject();
                if (jsonTree.isJsonObject() && jsonTree.get("name_popular") != null) {
                    id = jsonTree.get("name_popular").getAsJsonArray().get(0).getAsJsonObject().get("id").getAsString();
                } else if (jsonTree.isJsonObject() && jsonTree.get("name_exact") != null) {
                    id = jsonTree.get("name_exact").getAsJsonArray().get(0).getAsJsonObject().get("id").getAsString();
                } else if (jsonTree.isJsonObject() && jsonTree.get("name_approx") != null) {
                    id = jsonTree.get("name_approx").getAsJsonArray().get(0).getAsJsonObject().get("id").getAsString();
                }
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return id;
    }

    private static String[] getCeleWorksId(String personId, String attr) {
        LinkedList<String> results = new LinkedList<String>();

        long timestamp = date.getTime();
        String requestPath = String.format("https://app.imdb.com/name/maindetails?nconst=%s&apiKey=d2bb34ec6f6d4ef3703c9b0c36c4791ef8b9ca9b&apiPolicy=app1_1&locale=en_US&timestamp=%d&api=v1&appid=iphone1_1", personId, timestamp);
        String searchData = "";

        URL url = null;
        // the imdb app api needs a time break every access, retry if not successful
        while (true) {
            try {
                Thread.sleep(5000);

                url = new URL(requestPath);
                BufferedReader br = new BufferedReader(new InputStreamReader(url.openStream()));

                String strTemp = "";
                searchData = "";

                if (null != (strTemp = br.readLine())) {
                    searchData += strTemp;
                }

                break;
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        JsonParser parser = new JsonParser();
        JsonObject jsonTree = parser.parse(searchData).getAsJsonObject();
        if (jsonTree.isJsonObject() && jsonTree.get("data") != null && jsonTree.get("data").getAsJsonObject().get("known_for") != null) {
            JsonArray worksArray = jsonTree.get("data").getAsJsonObject().get("known_for").getAsJsonArray();
            for (JsonElement je : worksArray) {
                String attrStr = je.getAsJsonObject().get("attr").getAsString().toLowerCase();
                if (attr.toLowerCase().equals("actor") && (attrStr.toLowerCase().contains("actor") || attrStr.toLowerCase().contains("actress"))) {
                    results.add(je.getAsJsonObject().get("title").getAsJsonObject().get("tconst").getAsString());
                } else if (attr.toLowerCase().equals("creater") && attrStr.toLowerCase().contains("director")) {
                    results.add(je.getAsJsonObject().get("title").getAsJsonObject().get("tconst").getAsString());
                } else if (attr.toLowerCase().contains("producer") && attrStr.toLowerCase().contains("producer")) {
                    results.add(je.getAsJsonObject().get("title").getAsJsonObject().get("tconst").getAsString());
                }
            }
        }

        return results.toArray(new String[results.size()]);
    }


    private static String getReviewRec(String workId) {
        long timestamp = date.getTime();
        String requestPath = String.format("https://app.imdb.com/title/usercomments?apiKey=d2bb34ec6f6d4ef3703c9b0c36c4791ef8b9ca9b&apiPolicy=app1_1&locale=en_US&timestamp=%d&tconst=%s&api=v1&appid=Diphone1_1&limit=%d", timestamp, workId, 1000);
        String searchData = "";

        URL url = null;
        while (true) {
            try {
                Thread.sleep(5000);

                url = new URL(requestPath);

                BufferedReader br = new BufferedReader(new InputStreamReader(url.openStream()));
                String strTemp = "";
                searchData = "";

                while (null != (strTemp = br.readLine())) {
                    searchData += strTemp;
                }

                break;
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        System.out.println(workId);
        System.out.println(url);

        return searchData;
    }

    private static int[] getPosNegTimes(String commentContent, String fname, String lname) {
        int[] times = new int[2];
        InputStream modelIn = null;

        try {
            // split sentences
            modelIn = new FileInputStream("en-sent.bin");
            SentenceModel model = new SentenceModel(modelIn);
            SentenceDetectorME sentenceDetector = new SentenceDetectorME(model);
            String sentences[] = sentenceDetector.sentDetect(commentContent);

            for (String sentence : sentences) {
                if (sentence.contains(fname) || sentence.contains(lname)) {
                    int result = classifyNewReview(sentence);
                    if (result == 1) {
                        times[0]++;
                    } else {
                        times[1]++;
                    }
                    System.out.println(String.format("<tag>: %d, \t<sentence> %s", result, sentence));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (modelIn != null) {
                try {
                    modelIn.close();
                }
                catch (IOException e) {
                }
            }
        }

        return times;
    }

    private static int classifyNewReview(String sentence) {
        DocumentCategorizerME myCategorizer = new DocumentCategorizerME(model);

        InputStream modelIn = null;
        TokenizerModel tmodel = null;
        try {
            modelIn = new FileInputStream("en-token.bin");
            tmodel = new TokenizerModel(modelIn);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (modelIn != null) {
                try {
                    modelIn.close();
                }
                catch (IOException e) {
                }
            }
        }
        Tokenizer tokenizer = new TokenizerME(tmodel);
        String[] tokens = tokenizer.tokenize(sentence);

        double[] outcomes = myCategorizer.categorize(tokens);
        for (double outcome : outcomes) {
            System.out.println("outcomes[i]: " + outcome);
        }
        String category = myCategorizer.getBestCategory(outcomes);

        return Integer.parseInt(category);
    }
}
