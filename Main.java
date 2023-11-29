import java.util.*;
import java.util.stream.Collectors;

public class Main {

    static int left;
    static int right;

    public static void main(String[] args) {
        int[] nums = {-4, -1, -1, 0, 1, 2};
        threeSum(nums);
    }

    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        int[] result = new int[];
        Arrays.sort(nums);
        System.out.println(Arrays.toString(nums));

        for(int i = 0 ; i < nums.length-2 ; i++){
            int target = nums[i]*-1;
            int left = i+1;
            int right = nums.length-1;
            int sum;

            while(left < right){
                sum = nums[left] + nums[right];
                if(sum > target)
                    right--;
                else if(sum < target)
                    left++;
                else{
                    System.out.println(nums[i] + " " + nums[left] + " " + nums[right]);
                    result.add(Arrays.asList(nums[i],nums[left],nums[right]));
                    while(left < right && nums[left] == nums[left+1])
                        left++;
                    while(left < right && nums[right] == nums[right-1])
                        right--;
                    right--;
                    left++;
                }
            }

            while(i < nums.length-1 && nums[i] == nums[i+1])
                i++;
        }
        return result;
    }

    public static int trap(int[] height) {
        int temp = height[0];
        int [] maxL = new int[height.length];
        int [] maxR = new int[height.length];
        for(int i = 1 ; i < height.length ; i++){
            temp = Math.max(temp, height[i-1]);
            maxL[i] = temp;
        }
        temp = height[height.length-1];
        for(int i = height.length-2 ; i >= 0 ; i--){
            temp = Math.max(temp, height[i+1]);
            maxR[i] = temp;
        }
        System.out.println(Arrays.toString(maxL));
        System.out.println(Arrays.toString(maxR));

        int ans = 0;
        for(int i = 0 ; i < height.length ; i++){
            ans += Math.max(Math.min(maxL[i], maxR[i])-height[i],0);
        }
        return ans;
    }

    public static int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0 ; i < nums.length ; i++) {
            map.put(nums[i], i);
        }

        for (int i = 0; i < nums.length; i++) {
            int temp = target - nums[i];
            if(map.containsKey(temp) && i != map.get(temp)){
                return new int[]{i, map.get(temp)};
            }
        }
        return null;
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
