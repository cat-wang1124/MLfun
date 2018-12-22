//Filename: WordCountContainer.java
//Name: Catherine Wang
//ID: cs8bfeq
//Date: October 16, 2018
//
//Description: WordCountContainer.java contains methods getWordsFromFile(), removeCommon(), topNWords(), toString(),
//and outputWords(). 


import java.util.*;
import java.io.*;

public class WordCountContainer {

  ArrayList<WordCount> list;

  public WordCountContainer() {
    list = new ArrayList<WordCount>();
  }


  //getWordsFromFile scans in the parameter filename and loops through to compare if the 
  //ArrayList contains the word from the scanner. If it already does, the WordCount increments
  //in the ArrayList because it already exists. If it does not already exist, a WordCount object
  //is created in the ArrayList with a set count of 1
  public void getWordsFromFile( String filename ) throws IOException {

    Scanner myScanner = new Scanner(new File(filename));
    WordCount scnObj = new WordCount("");
    boolean checker = false;

    while (myScanner.hasNext()){ //loops through file

      if(list.size() == 0){ //checks if ArrayList is empty, adds first token scanned word
        scnObj = new WordCount(myScanner.next());
        list.add(scnObj);
      }

      String scnWord = myScanner.next();
      scnObj = new WordCount(scnWord);

      for(int i = 0; i < list.size(); i++){ //loops through list to see if word has been added
        WordCount tester = list.get(i);
        String strCheck = tester.getWord();

        if(scnWord.equalsIgnoreCase(strCheck)){
          tester.increment();
          checker = true;
        }
      }

      if(checker != true){
        list.add(scnObj);
      }
      checker = false;
    }
  }


  //removeCommon scans the given omitFilename and checks with the ArrayList list to
  //see if the ArrayList list contains any common words from the omitFilename. If so,
  //removeCommon removes that current word
  public void removeCommon( String omitFilename ) throws IOException {

    Scanner myScanner = new Scanner(new File(omitFilename));

    while(myScanner.hasNext()){
      String scnWord = myScanner.next();

      if(list.size() == 0){ //checks if ArrayList is empty, adds first token scanned word
        WordCount scnObj = new WordCount(scnWord);
        list.add(scnObj);
      }

      for(int i = 0; i < list.size(); i++){ //loops through ArrayList to check common words
        String strCheck = list.get(i).getWord();

        if(scnWord.equalsIgnoreCase(strCheck)){
          list.remove(i);
        }
      }
    }
  }


  //topNWords takes in an integer number and returns the top integer number
  //of largest to smaller WordObject counts. If empty, it should return empty
  //brackets.
  public ArrayList<WordCount> topNWords(int n) {

    ArrayList<WordCount> nList = new ArrayList<WordCount>();
    int counter = 0;
    int max = 0;

    //if the number of words is less than n, return all words
    if(list.size() < n){
      return list;
    }

    //looks for the max count and makes it negative
    for(int i = 0; i < list.size(); i++){
      if(list.get(i).getCount() >= max && counter < n){
        max = list.get(i).getCount();
        nList.add(list.get(i));
        list.get(i).setCount((-1)*max);
        counter++;
      }
    }

    //changes all the negative values of original list back to positive 
    for(int i = 0; i < list.size(); i++){
      if(list.get(i).getCount() < 0 ){
        list.get(i).setCount((-1)*list.get(i).getCount());
      }
    }

    return nList;
  }


  //toString runs through the ArrayList and converts the current WordCount object's Sting and loops through
  //by concatenating into a final string everytime
  public String toString() {

    String finString = new String("");
    String currString = new String("");

    for(int i = 0; i < list.size(); i++){

      if(i == list.size()-1){
        currString = list.get(i).getWord() + "(" + list.get(i).getCount() + ")";
      } else{
        currString = list.get(i).getWord() + "(" + list.get(i).getCount() + ") ";
      }

      finString = finString + currString; //concats the converted string to the final string of strings 
    }

    return finString;
  }


  //outputWords .toStrings() prints the list if false and .toStrings() prints the list onto the file
  //myOutput.out with the PrintWriter class
  public void outputWords(boolean printToFile) throws IOException{
    PrintWriter output = new PrintWriter(new File("myOutput.out"));

    if(printToFile == false){
      System.out.println(this.toString());
    }
    if(printToFile == true){
      output.print(this.toString());
    }
    output.close();
  }

  public ArrayList<WordCount> getList(){
    return list;
  }


  //add method takes in the String word, where it will increment the count 
  //of the word if it exists in the list, if else it will append a new 
  //WordCount object to the list
  public void add(String word){

    if (list.size() == 0){

      WordCount curr = new WordCount(word);
      list.add(curr);
      return;
    }

    for (int i = 0; i < list.size(); i++){

      WordCount wCheck = list.get(i);
      String strCheck = wCheck.getWord();
      if (word.equalsIgnoreCase(strCheck)){

        wCheck.increment();
        return;
      }
    }
       
    WordCount curr = new WordCount(word);    
    list.add(curr);  
  }

}
