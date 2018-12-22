/** Catherine Wang
 * cs8bfeq
 * File MarkovModel.java
 *
 * Description here: This java file MarkovModel encapsulates what goes into 
 * implementing the Markov Method of predicting and generating written text 
 * from computers, we are able to customize a degree of words and generate
 * and read through a given file of text and create a number of words the
 * user inputs that is predicted and generated randomly
 * */

import java.util.HashMap;
import java.util.ArrayList;
import java.io.*;
import java.nio.file.*;
import java.util.Random;
import java.util.Scanner;

public class MarkovModel {

  protected HashMap<ArrayList<String>, WordCountContainer> predictionMap;

  protected int degree;
  protected Random random;
  protected boolean isWordModel;



  public MarkovModel (int degree, boolean isWordModel) {

    this.degree = degree;
    this.isWordModel = isWordModel;
    //isWordModel true would be words, false would be single characters

  }



  /*trainFromText takes in a filename, generates it into a single 
   * string named content, creates a wraparound text, and trained 
   * our word model and returns nothing
   */
  public void trainFromText(String filename) {

    String content;
    try {
      content = new String(Files.readAllBytes(Paths.get(filename)));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }
    // `content` contains everything from the file, in one single string

    //wrap around
    //list is the converted content string to an ArrayList of Strings 
    //through scanning content one by one
    predictionMap = new HashMap<ArrayList<String>, WordCountContainer>();
    Scanner myScanner = new Scanner(content);
    ArrayList<String> list = new ArrayList<String>();

    while (myScanner.hasNext()){

      String scnWord = myScanner.next();
      list.add(scnWord);
    }

    //loops through the list ArrayList to get the n degree from
    //the beginning to the end of the ArrayList list
    String word;

    for (int i = 0; i < degree; i++){

      word = list.get(i);
      list.add(word);
    }

    //training word model      
    String prefix;

    for (int start = 0; start < list.size() - degree; start++){

      //prefList will be reseted everytime we finish n degree per
      //new starting index
      ArrayList<String> prefList = new ArrayList<String>();   

      //creates the prefList with the right words
      for (int i = start; i < degree + start; i++){

        prefix = list.get(i);
        prefList.add(prefix.toLowerCase());
      }

      //if the hashmap contains our prefList we add the existed prefList
      //and grabs our next word
      if (predictionMap.containsKey(prefList)){

        WordCountContainer exist = predictionMap.get(prefList);
        exist.add(list.get(start + degree));
      } 
      //if the hashmap does not contain our prefList we will put it into
      //our hashmap and create an empty new WordCountContainer and grab 
      //our next word
      else{

        WordCountContainer empty = new WordCountContainer();   
        empty.add(list.get(start + degree));
        predictionMap.put(prefList, empty);
      }
    }
  }



  /*getFlattenedList takes in a a prefix that is an ArrayList of Strings
   * and gets the key prefix's value of a WordCountContainer, turns
   * the WordCountContainer value into an ArrayList of Strings that
   * correspond to the counts of each word and returns the ArrayList
   * of Strings
   */
  public ArrayList<String> getFlattenedList(ArrayList<String> prefix){

    WordCountContainer wCountCont = predictionMap.get(prefix);
    ArrayList<WordCount> value = wCountCont.list;
    ArrayList<String> fList = new ArrayList<String>();

    //loops one WordCount object word by one WordCount object word
    for (int i = 0; i < value.size(); i++){

      WordCount wCount = value.get(i);
      String word = wCount.getWord();
      int limit = wCount.getCount();

      //for every WordCount object word, we will add into the flattened
      //ArrayList of Strings according to the count number of times
      for (int count = 0; count < limit; count++){

        fList.add(word);
      }
    }
    return fList;
  }



  /*generateNext takes in an ArrayList of Strings called prefix and
   * will get the flattened ArrayList of Strings by calling getFlattenedList
   * onto the parameter prefix. It will then generate a new random index
   * within the size of the ArrayList of Strings called list and returns the
   * String of the index we randomly find
   */
  public String generateNext(ArrayList<String> prefix) {

    //use getFlattenedList onto the prefix parameter we are given
    ArrayList<String> list = 
      new ArrayList<String>(this.getFlattenedList(prefix));
    random = new Random();
    //find a random word in the flattened list of prefix
    int ranIndex = random.nextInt(list.size());

    return list.get(ranIndex);
  }



  /*generate takes in a count that represents the number of words required
   * to be generated and returns a string of words generated
   */
  public String generate(int count) {

    ArrayList<ArrayList<String>> keys = 
      new ArrayList<ArrayList<String>>(predictionMap.keySet());
    //with tracker, our purpose is to constantly delete the first and last word
    //everytime
    String finString = new String();

    //random int to get a random key
    random = new Random();
    int ranIndex = random.nextInt(keys.size());
    ArrayList<String> prefix = new ArrayList<String>(keys.get(ranIndex));
    //we copy prefix onto tracker so we already have the randomly picked prefix
    //to start on our tracker to find our first predicted word
    ArrayList<String> tracker = new ArrayList<String>(prefix);      
    
    //loops for the generating of count number of words
    for (int curr = 0; curr < count; curr++){   
    
      String pWord = this.generateNext(tracker);
      tracker.add(pWord.toLowerCase());

      finString += " " + pWord; 
      tracker.remove(0);
    }      

    return finString;   
  }



    /*toString will utilize the ArrayList toString() and the WordCountContainer 
     * toString() 
     */
    @Override 
    public String toString(){

      String finString = new String();

      for (HashMap.Entry<ArrayList<String>, WordCountContainer> entry
          : predictionMap.entrySet()){

        ArrayList<String> keys = entry.getKey();
        WordCountContainer values = entry.getValue();
        finString += keys.toString() + ": " + values.toString() + "\n";
          }
      
      return finString;
    }

  }
