import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        String[] logs = {"dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"};
        String[] answers = leetcode937(logs);

        for(String ans : answers)
            System.out.println(ans);
    }

    private static String[] leetcode937(String[] logs){
        List<String> letterList = new ArrayList<>();
        List<String> digitList = new ArrayList<>();

        for(String log : logs){
            if(Character.isDigit(log.split(" ")[1].charAt(0))){
                digitList.add(log);
            }
            else{
                letterList.add(log);
            }
        }

        letterList.sort((s1, s2) -> {
            String[] s1x = s1.split(" ",2);
            String[] s2x = s2.split(" ",2);

            int compared = s1x[1].compareTo(s2x[1]);
            if(compared == 0){
                return s1x[0].compareTo(s2x[0]);
            }
            return compared;
            }
        );
        letterList.addAll(digitList);

        return letterList.toArray(new String[0]);

    }

    private static void reverseString(char[] s) {
        int start = 0;
        int end = s.length-1;

        char temp;
        while(start < end){
            temp = s[start];
            s[start] = s[end];
            s[end] = temp;
            start++;
            end--;
        }
    }
}
