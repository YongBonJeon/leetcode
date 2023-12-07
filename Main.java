import java.util.*;
import java.util.stream.Collectors;

public class Main {

    static int left;
    static int right;

    public class ListNode {
        int val;
        ListNode next;ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public static void main(String[] args) {
        String s = "dbacdcbc";
        System.out.println(removeDuplicateLetters(s));
    }

    public int solution(int[] scoville, int K) {
        int answer = 0;

        PriorityQueue<Integer> pq = new PriorityQueue<>();

        for (int i : scoville) {
            pq.add(i);
        }

        while (pq.size() >= 1 && pq.peek() < K) {
            int f1 = pq.poll();
            int f2 = pq.poll();

            pq.add(f1+f2*2);
            answer++;
        }

        return answer;
    }

    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((o1, o2) -> {
            Integer ed1 = o1[0]*o1[0]+o1[1]*o1[1];
            Integer ed2 = o2[0]*o2[0]+o2[1]*o2[1];

            if(ed1 == ed2)
                return 0;
            else if(ed1 > ed2)
                return 1;
            else
                return -1;
        });

        for (int[] point : points) {
            pq.add(point);
        }

        int[][] ans = new int[k][2];

        for (int i = 0; i < k; i++) {
            ans[k] = pq.poll();
        }

        return ans;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>((o1, o2) -> {
            if(o1.val == o2.val)
                return 0;
            else if(o1.val > o2.val)
                return 1;
            else
                return -1;
        });

        for (ListNode list : lists) {
            pq.add(list);
        }

        ListNode root = new ListNode();
        ListNode cur = root;

        while (!pq.isEmpty()) {
            ListNode temp = pq.poll();
            while (temp != null) {
                cur.next = temp;
                temp = temp.next;
                cur = cur.next;
            }
        }
        return root.next;
    }

    public int[] dailyTemperatures(int[] temperatures) {
        Deque<Integer> stack = new ArrayDeque<>();

        int[] result = new int[temperatures.length];

        for (int i = 0 ; i < temperatures.length ; i++) {
            int cnt = 1;
            while (stack.peek() < temperatures[i]) {
                int last = stack.pop();
                result[last] = i - last;
            }
            stack.push(i);
        }
        Queue<Integer> queue = new LinkedList<>();
        int[] ints = new int[5];
        return null;




    }

    public static String removeDuplicateLetters(String s) {
        Map<Character, Integer> counter = new HashMap<>();
        Map<Character, Boolean> checker = new HashMap<>();

        Deque<Character> stack = new ArrayDeque<>();

        for (int i = 0; i < s.length(); i++) {
            Character c = s.charAt(i);
            counter.put(c, counter.get(c) == null ? 1 : counter.get(c) + 1);
        }

        for (int i = 0; i < s.length(); i++) {
            Character c = s.charAt(i);

            if (stack.isEmpty()) {
                stack.push(c);
                //System.out.println("push " + c);
                checker.put(c, true);
                continue;
            }

            if (checker.get(c) != null && checker.get(c) == true) {
                counter.put(c, counter.get(c) - 1);
                continue;
            }

            while (!stack.isEmpty() && c < stack.peek() && counter.get(stack.peek()) > 1) {
                counter.put(stack.peek(), counter.get(stack.peek()) - 1);
                checker.put(stack.peek(), false);
                stack.pop();
                //System.out.println("pop " + stack.pop());

            }
            stack.push(c);
            //System.out.println("push " + c);
            checker.put(c, true);
        }

        String ans = "";
        while (!stack.isEmpty()) {
            ans += stack.pop();
        }
        return new StringBuilder(ans).reverse().toString();
    }

    public boolean isValid(String s) {
        Deque<Character> stack = new ArrayDeque<>();

        Map<Character, Character> map = new HashMap<>();

        map.put(']','[');
        map.put('}','{');
        map.put(')','(');

        for(int i = 0 ; i < s.length() ; i++){
            Character c = s.charAt(i);
            if(c == '{' || c == '(' ||  c == '['){
                stack.push(c);
            }
            else{
                if(stack.isEmpty())
                    return false;
                if(map.get(stack.peekLast()) == stack.peekLast()){
                    stack.pop();
                }
                else
                    return false;
            }
        }
        return true;
    }

    public ListNode swapPairs(ListNode head) {
        ListNode cur = head;
        ListNode next;

        while(cur != null && cur.next != null){
            next = cur.next;
            cur.next = next.next;
            next.next = cur;

            cur = cur.next.next;
        }

        return head;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode result = new ListNode();

        while (l1 != null && l2 != null) {
            result.next = new ListNode(l1.val + l2.val);
        }

        return result.next;
    }

    public ListNode reverseList(ListNode head) {
        Deque<Integer> deque = new LinkedList<>();

        while (head != null) {
            deque.add(head.val);
            head = head.next;
        }

        ListNode result = new ListNode();
        ListNode cur = result;

        while (!deque.isEmpty()) {
            cur.next = new ListNode(deque.pollLast());
            cur = cur.next;
        }

        return result.next;
    }

    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode result = new ListNode();
        ListNode cur = result;

        while(list1 != null && list2 != null){
            if(list1.val <= list2.val) {
                cur.next = new ListNode(list1.val);
                list1 = list1.next;
                cur = cur.next;
            }
            else {
                cur.next = new ListNode(list2.val);
                list2 = list2.next;
                cur = cur.next;
            }
        }
        while(list1 != null){
            cur.next = new ListNode(list1.val);
            list1 = list1.next;
            cur = cur.next;
        }
        while(list2 != null){
            cur.next = new ListNode(list2.val);
            list2 = list2.next;
            cur = cur.next;
        }
        return result;
    }
    public boolean isPalindrome(ListNode head) {
        Deque<Integer> deque = new LinkedList<>();

        ListNode node = head;
        while (node != null) {
            deque.add(node.val);
            node = node.next;
        }

        while(!deque.isEmpty() && deque.size() > 1) {
            if(deque.pollFirst() != deque.pollLast())
                return false;
        }
        return true;
    }

    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
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
