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
        Map<Character, Integer> map = new HashMap<>();

        PriorityQueue<int[]> pq = new PriorityQueue<>();

        for (Map.Entry<Character, Integer> e : map.entrySet()) {

        }
    }

    public int solution(int[][] maps) {
        int answer = 0;
        int n = maps.length;
        int m = maps[0].length;
        int[][] dist = new int[n][m];

        Arrays.fill(dist, 0);

        PriorityQueue<List<Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.get(2)));

        pq.add(Arrays.asList(0,0,0));

        while (!pq.isEmpty()) {
            List<Integer> cur = pq.poll();

            int i = cur.get(0);
            int j = cur.get(1);
            int cur_dist = cur.get(2);

            if (i == n - 1 && j == m - 1) {
                return cur_dist;
            }

            for (int d = 0; d < 4; d++) {
                int nexti = i + dir[d][0];
                int nextj = j + dir[d][1];

                if(nexti < 0 || nextj < 0 || nexti >= n || nextj >= m)
                    continue;

                pq.add(Arrays.asList(nexti, nextj, cur_dist + 1));
            }
        }

        return -1;
    }

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        Map<Integer, Map<Integer, Integer>> graph = new HashMap<>();

        for (int[] flight : flights) {
            graph.putIfAbsent(flight[0], new HashMap<>());
            graph.get(flight[0]).put(flight[1], flight[2]);
        }

        Map<Integer, Integer> visited = new HashMap<>();

        PriorityQueue<List<Integer>> pq
                = new PriorityQueue<>(Comparator.comparingInt(a -> a.get(1)));
        pq.add(Arrays.asList(src, 0, 0));

        while (!pq.isEmpty()) {
            List<Integer> cur = pq.poll();

            int u = cur.get(0);
            int distU = cur.get(1);
            int k_visited = cur.get(2);

            if (u == dst) {
                return distU;
            }

            visited.put(u, k_visited);

            if (!graph.containsKey(u)) {
                if (k_visited + 1 <= k) {
                    for (Map.Entry<Integer, Integer> v : graph.get(u).entrySet()) {
                        if(!visited.containsKey(v.getKey()) || k_visited + 1 < visited.get(v.getKey()))
                            pq.add(Arrays.asList(v.getKey(), v.getValue() + distU, k_visited + 1));
                    }
                }
            }
        }
        return -1;
    }

    public int networkDelayTime(int[][] times, int n, int k) {
        Map<Integer, Map<Integer, Integer>> graph = new HashMap<>();

        for (int[] time : times) {
            graph.putIfAbsent(time[0], new HashMap<>());
            graph.get(time[0]).put(time[1], time[2]);
        }

        Map<Integer, Integer> distance = new HashMap<>();
        PriorityQueue<Map.Entry<Integer, Integer>> pq
                = new PriorityQueue<>(Map.Entry.comparingByValue());
        pq.add(new AbstractMap.SimpleEntry<>(k, 0));

        while (!pq.isEmpty()) {
            Map.Entry<Integer, Integer> cur = pq.poll();

            int u = cur.getKey();
            int distU = cur.getValue();

            if (!distance.containsKey(u)) {
                distance.put(u, distU);

                if (graph.containsKey(u)) {
                    for (Map.Entry<Integer, Integer> v : graph.get(u).entrySet()) {
                        pq.add(new AbstractMap.SimpleEntry<>(v.getKey(), v.getValue() + distU));
                    }
                }
            }
        }

        if (distance.size() == n) {
            return Collections.max(distance.values());
        } else
            return -1;
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> finishToTakeMap = new HashMap<>();

        for (int[] pre : prerequisites) {
            if (finishToTakeMap.get(pre[0]) == null) {
                finishToTakeMap.put(pre[0], new ArrayList<>());
            }
            finishToTakeMap.get(pre[0]).add(pre[1]);
        }

        for (Integer finish : finishToTakeMap.keySet()) {
            gogo(finishToTakeMap, finish, new ArrayList<Integer>(), new ArrayList<Integer>())
        }
    }

    private Boolean gogo(Map<Integer, List<Integer>> finishToTakeMap, Integer finish
            , List<Integer> takes, List<Integer> visited) {

        if(takes.contains(finish))
            return false;

        if(visited.contains(finish))
            return true;

        if (finishToTakeMap.get(finish) != null) {
            takes.add(finish);
            for (Integer take : finishToTakeMap.get(finish)) {
                if(!gogo(finishToTakeMap, take, takes, visited))
                    return false;
            }
            takes.remove(finish);
            visited.add(finish);
        }
        return true;
    }

    public List<String> findItinerary(List<List<String>> tickets) {
        Map<String, PriorityQueue<String>> fromToMap = new HashMap<>();

        for (List<String> ticket : tickets) {
            fromToMap.put(ticket.get(0), new PriorityQueue<>());
            fromToMap.get(ticket.get(0)).add(ticket.get(1));
        }

        List<String> result = new ArrayList<>();

        go("JFK", fromToMap, result);

        result.toArray()
    }

    private void go(String cur, Map<String, PriorityQueue<String>> fromToMap, List<String> result) {
        result.add(cur);
        String next = fromToMap.get(cur).poll();
        go(next, fromToMap, result);
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        backtracking_subset(0, nums, cur, result);

        return result;
    }

    private void backtracking_subset(int start, int[] nums, List<Integer> cur, List<List<Integer>> result) {
        result.add(cur);

        for (int i = start; i < nums.length; i++) {
            List<Integer> next = new ArrayList<>(cur);
            next.add(nums[i]);
            backtracking_combinationSum(i+1, numss, next, result);
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {

        List<List<Integer>> result = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        backtracking_combinationSum(0, target, candidates, cur, result);

        return result;
    }

    private void backtracking_combinationSum(int start, int target, int[] candidates, List<Integer> cur, List<List<Integer>> result) {
        int sum = cur.stream().mapToInt(i -> i).sum();
        if(sum == target)
            result.add(cur);

        for (int i = start; i < candidates.length; i++) {
            List<Integer> next = new ArrayList<>(cur);
            next.add(candidates[i]);
            backtracking_combinationSum(i, target, candidates, next, result);
        }
    }

    public List<List<Integer>> combine(int n, int k) {

        List<List<Integer>> result = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        combination_dfs(0, n, k, cur, result);
    }

    private void combination_dfs(int num, int n, int k, List<Integer> cur, List<List<Integer>> result) {

        if (num == k) {
            result.add(cur);
            return ;
        }

        for (int i = 1; i <= n; i++) {
            if (!cur.contains(i)) {
                List<Integer> next = new ArrayList<>(cur);
                next.add(i);
                combination_dfs(num + 1, n, k, next, result);
            }
        }
    }

    public List<List<Integer>> permute(int[] nums) {

        List<List<Integer>> result = new ArrayList<>();
        List<Integer> path = new ArrayList<>();

        permutation_dfs(nums, path,  result);

    }

    private void permutation_dfs(int[] nums, List<Integer> path, List<List<Integer>> result) {
        if (nums.length == path.size()) {
            result.add(path);
            return ;
        }

        for (int i = 0; i < nums.length; i++) {
            if (!path.contains(nums[i])) {
                path.add(nums[i]);
                permutation_dfs(nums, path, result);
                path.remove(nums[i]);
            }
        }
    }

    Map<Integer, List<Character>> map = new HashMap<>();

    public List<String> letterCombinations(String digits) {
        map.put(2, Arrays.asList('a', 'b', 'c'));
        map.put(3, Arrays.asList('d', 'e', 'f'));
        map.put(4, Arrays.asList('g', 'h', 'i'));
        map.put(5, Arrays.asList('j', 'k', 'l'));
        map.put(6, Arrays.asList('m', 'n', 'o'));
        map.put(7, Arrays.asList('p', 'q', 'r', 's'));
        map.put(8, Arrays.asList('t', 'u', 'v'));
        map.put(9, Arrays.asList('w', 'x', 'y', 'z'));


        backtracking(0, digits, new StringBuilder());

    }

    private void backtracking(int index, String digits, StringBuilder path) {

        if (path.length() == digits.length()) {
            String.valueOf(path);
            System.out.println(path);
            return ;
        }

        for (Character c : map.get(index)) {
            backtracking(index+1, digits, new StringBuilder(path).append(c));
        }
    }

    int[][] dir = {{0,1}, {0,-1}, {1, 0}, {-1,0}};

    public int numIslands(char[][] grid) {
        Boolean[][] used = new Boolean[grid.length][grid[0].length];
        for (Boolean[] booleans : used) {
            for (Boolean aBoolean : booleans) {
                aBoolean = false;
            }
        }

        int ans = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (!used[i][j]) {
                    dfs(i, j, used);
                    ans++;
                }
            }
        }
        return ans;
    }

    private void dfs(int i, int j, Boolean[][] used) {
        // 방문
        used[i][j] = true;

        for (int d = 0; d < 4; d++) {
            int nexti = i + dir[d][0];
            int nextj = j + dir[d][1];

            if (!used[nexti][nextj]) {
                dfs(nexti, nextj, used);
            }
        }
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
