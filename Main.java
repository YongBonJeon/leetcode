import java.util.*;

public class Main {

    static int left;
    static int right;

    public static void main(String[] args) {
        String str = "babad";
        System.out.println(longestPalindrome(str));
    }

    public static String longestPalindrome(String s) {

        if(s.length() == 1)
            return s;

        for(int i = 0 ; i < s.length()-1 ; i++){
            checkPalindrome(s, i, i+2);
            checkPalindrome(s, i, i+1);
        }

        return s.substring(left+1, right);
    }

    private static void checkPalindrome(String s, int l, int r){
        System.out.println(l + " " + r);
        while(l > 0 && r < s.length()-1 && s.charAt(l) == s.charAt(r)){
            System.out.println(s.charAt(l) + " " + s.charAt(r));
            l--;
            r++;
        }
        if(right - left < r-l){
            System.out.println(s.substring(l+1,r));
            right = r;
            left = l;
        }
    }

    private static List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> anagramsMap = new HashMap<>();
        String temp;

        for(String str : strs){
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);

            if(!anagramsMap.containsKey(key))
                anagramsMap.put(key, new ArrayList<>());
            anagramsMap.get(key).add(str);
            //System.out.println(key + " " + str);
        }
        return new ArrayList<>(anagramsMap.values());
    }

    private static String mostCommonWord(String paragraph, String[] banned) {
        HashSet<String> ban = new HashSet<>(Arrays.asList(banned));
        Map<String, Integer> countMap = new HashMap<>();
        String[] words = paragraph.replaceAll("\\W+", " ").toLowerCase().split(" ");

        for (String word : words) {
            if(!ban.contains(word))
                countMap.put(word, countMap.getOrDefault(word, 0) + 1);
        }

        return Collections.max(countMap.entrySet(), Map.Entry.comparingByValue()).getKey();
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
