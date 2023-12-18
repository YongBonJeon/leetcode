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

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static void main(String[] args) {
        String s = "dbacdcbc";
        System.out.println(removeDuplicateLetters(s));
        Map<Character, Integer> map = new HashMap<>();

        PriorityQueue<int[]> pq = new PriorityQueue<>();

        for (Map.Entry<Character, Integer> e : map.entrySet()) {

        }
        List<Integer> result = new ArrayList<>();

        int[] ints = new int[];
        Arrays.stream(ints).max().getAsInt();
    }

    public int rob(int[] nums) {
        int[] dp = new int[nums.length + 1];

        dp[0] = nums[0];
        dp[1] = nums[1];

        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i-1], dp[i-2]) + nums[i];
        }
        return Math.max(dp[nums.length], dp[nums.length - 1]);
    }

    public int climbStairs(int n) {
        int[] dp = new int[n+2];

        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }

        return dp[n];

    }

    public int maxSubArray(int[] nums) {
        int maxSum = Integer.MIN_VALUE;

        int localSum = 0;
        for (int num : nums) {
            localSum += num;

            if(localSum < 0)
                localSum = 0;

            maxSum = Math.max(maxSum, localSum);
        }
        return maxSum;
    }

    public int fib(int n) {
        int[] dp = new int[31];
        dp[0] = 0;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            dp[n] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }

    public List<Integer> diffWaysToCompute(String expression) {

        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < expression.length(); i++) {

            char c = expression.charAt(i);

            if(c == '+' || c == '-' || c == '*') {
                List<Integer> left = diffWaysToCompute(expression.substring(0, i));
                List<Integer> right = diffWaysToCompute(expression.substring(i + 1));

                for (Integer l : left) {
                    for (Integer r : right) {
                        if(c == '+')
                            result.add(l + r);
                        else if(c == '-')
                            result.add(l - r);
                        else if(c == '*')
                            result.add(l * r);
                    }
                }
            }
        }
        if(result.isEmpty())
            result.add(Integer.parseInt(expression));

        return result;
    }

    public int majorityElement(int[] nums) {
        Map<Integer, Integer> count = new HashMap<>();

        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }

        Integer maxValue = count.values().stream().max(Integer::compareTo).orElse(0);

        if (maxValue > nums.length / 2) {
            for (Integer key : count.keySet()) {
                if (count.get(key) == maxValue) {
                    return key;
                }
            }
        }
        return -1;
    }

    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);

        int ans = 0;
        // 7 8 9 10
        // 5 6 7 8
        int child = 0, cookie = 0;
        while (child < g.length && cookie < s.length) {
            if (g[child] <= s[cookie]) {
                child++;
                cookie++;
                ans++;
            } else {
                cookie++;
            }
        }
        return ans;
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        if(Arrays.stream(gas).sum() < Arrays.stream(cost).sum())
            return -1;

        int fuel = 0, start = 0;
        for (int i = 0; i < gas.length; i++) {
            if (fuel + gas[i] - cost[i] < 0) {
                start = i + 1;
                fuel = 0;
            } else {
                fuel += gas[i] - cost[i];
            }
        }
        return start;
    }

    public int leastInterval(char[] tasks, int n) {
        int[] nums = new int[26];
        int maxIdx = 0, maxValue = 0;
        for (char task : tasks) {
            nums[task-'A']++;
            if (nums[task - 'A'] > maxValue) {
                maxValue = nums[task-'A'];
                maxIdx = task-'A';
            }
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.reverseOrder());

        for (int i = 0; i < 26; i++) {
            if (nums[i] > 0 && i != maxIdx) {
                pq.add(nums[i]);
            }
        }

        int idle = (maxValue - 1) * n;

        while (!pq.isEmpty() && idle > 0) {
            idle -= Math.min(pq.poll(), maxValue - 1);
        }

        if(idle > 0)
            return idle + tasks.length;
        else
            return tasks.length;
    }

    public int[][] reconstructQueue(int[][] people) {
        Queue<int[]> pq = new PriorityQueue<>((o1, o2) ->
                o1[0] != o2[0] ? o2[0] - o1[0] : o1[1] - o2[1]);
        for (int[] person : people) {
            pq.add(person);
        }

        List<int[]> result = new ArrayList<>();
        while (!pq.isEmpty()) {
            int[] person = pq.poll();
            result.add(person[1], person);
        }
        return result.toArray(new int[result.size()][2]);
    }

    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        for (int i = 0; i < prices.length-1; i++) {
            if(prices[i+1] - prices[i] > 0)
                maxProfit += prices[i+1] - prices[i];
        }
        return maxProfit;
    }

    public int characterReplacement(String s, int k) {
        Map<Character, Integer> count = new HashMap<>();

        int left = 0, right = 0;
        int maxCount = 0;

        for(right = 1 ; right <= s.length() ; right++){
            count.put(s.charAt(right-1), count.getOrDefault(s.charAt(right-1), 0) + 1);
            maxCount = Collections.max(count.values());

            if (right - left - maxCount > k) {
                count.put(s.charAt(left), count.getOrDefault(s.charAt(left), 0) - 1);
                left++;
            }
        }
        return s.length() - left;
    }

    public String minWindow(String s, String t) {
        Map<Character, Integer> need = new HashMap<>();
        int total_need = t.length();
        for (char c : t.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }

        int left = 0, right = 0, start = 0, end = 0;
        int minLength = Integer.MAX_VALUE;

        for (char c : s.toCharArray()) {
            right++;

            if (need.containsKey(c) && need.get(c) > 0) {
                total_need--;
            }

            need.put(c, need.getOrDefault(c, 0) - 1);

            if (total_need == 0) {
                while (left < right && need.get(s.charAt(left)) < 0) {
                    need.put(s.charAt(left), need.get(s.charAt(left)) + 1);
                    left++;
                }

                if (minLength > right - left + 1) {
                    minLength = right - left + 1;
                    right = end;
                    left = start;
                }
                need.put(s.charAt(left), need.get(left) + 1);
                total_need++;
                left++;
            }
        }
        return s.substring(start, end);
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));
        List<Integer> maxResult = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            pq.add(new int[]{nums[i], i});
            if(i < k-1) continue;

            while (!pq.isEmpty() && pq.peek()[1] <= i - k) {
                pq.poll();
            }
            maxResult.add(pq.peek()[0]);
        }
        int[] result = new int[maxResult.size()];
        for (int i = 0; i < maxResult.size(); i++) {
            result[i] = maxResult.get(i);
        }
        return result;
    }
    public int[] maxSlidingWindowFail(int[] nums, int k) {
        List<Integer> maxResult = new ArrayList<>();
        Integer localMax = Integer.MIN_VALUE;
        Queue<Integer> window = new LinkedList<>();

        for(int i = 0 ; i < k ; i++)
            window.add(nums[i]);

        localMax = Collections.max(window);
        maxResult.add(localMax);

        for (int i = k; i < nums.length; i++) {
            Integer pop = window.poll();
            window.add(nums[i]);
            if (pop == localMax) {
                localMax = Collections.min(window);
            }

            maxResult.add(localMax);
        }
        int[] result = new int[maxResult.size()];
        for (int i = 0; i < maxResult.size(); i++) {
            result[i] = maxResult.get(i);
        }
        return result;
    }

    public int hammingDistance(int x, int y) {
        int result = x ^ y;
        return Integer.bitCount(result);
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int row = 0;
        int col = matrix[0].length-1;

        while (matrix[row][col] != target) {
            if (matrix[row][col] > target) {
                col--;
            } else if (matrix[row][col] < target) {
                row++;
            }
            if (row < 0 || col < 0 || row >= matrix.length || col >= matrix[0].length) {
                return false;
            }
        }
        return true;
    }

    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> result = new HashSet<>()

        for(int n : nums1) {
            int k;
            if(Arrays.binarySearch(nums2, n) != -1)
                result.add(n);
        }
        Integer[] ans = result.toArray(new Integer[0]);

        return Arrays.stream(ans).mapToInt(i -> i).toArray();
    }

    public int search1(int[] nums, int target) {
        int max_index = 0, max_value = nums[0];
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > max_value) {
                max_index = i;
                max_value = nums[i];
            }
        }
        if (target <= max_value) {
            return binarySearch(nums, 0, max_index, target);
        } else {
            return binarySearch(nums, max_index + 1, nums.length - 1, target);
        }
    }

    public int search(int[] nums, int target) {
        return binarySearch(nums, 0, nums.length, target);
    }

    private int binarySearch(int[] nums, int l, int r, int target) {
        int mid = (l + r) / 2;
        if(nums[mid] == target)
            return mid;

        if(nums[mid] > target)
            return binarySearch(nums, mid+1, r, target);
        else
            return binarySearch(nums, l, mid - 1, target);
    }

    public boolean isAnagram(String s, String t) {
        char[] tArr = t.toCharArray();
        tArr.toString();
        Arrays.sort(tArr);
        return s.equals(t);
    }

    public String largestNumber(int[] nums) {
        Integer[] arr = Arrays.stream(nums).boxed().toArray(Integer[]::new);
        Arrays.sort(arr, (o1, o2) -> (String.valueOf(o1) + o2).compareTo(String.valueOf(o2) + o1));

        StringBuilder s = new StringBuilder();
        for(Integer i : arr)
            s.append(i);
        return String.valueOf(Long.parseLong(s.toString()));


    }

    public int[] solution(String[] operations) {
        int[] answer = {};

        Queue<Integer> minHeap = new PriorityQueue<>();
        Queue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());

        for (String operation : operations) {
            String[] op = operation.split(" ");

            if (op[0].equals("I")) {
                maxHeap.add(Integer.valueOf(op[1]));
                minHeap.add(Integer.valueOf(op[1]));
            } else if (op[0].equals("D")) {
                if(minHeap.isEmpty()) continue;
                if (op[1].equals("-1")) {
                    maxHeap.remove(minHeap.poll());
                } else {
                    minHeap.remove(maxHeap.poll());
                }
            }
        }
        if(minHeap.isEmpty())
            return new int[]{0,0};
        else if(minHeap.size() == 1)
            return new int[]{minHeap.peek(), minHeap.peek()};
        return new int[]{maxHeap.poll(), minHeap.peek()};
    }

    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[k-1];
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        if(nums.length == 0) return null;
        return construct(nums, 0, nums.length-1);
    }

    private TreeNode construct(int[] nums, int l, int r) {
        if(l > r) return null;

        int mid = (l+r)/2;

        TreeNode node = new TreeNode(nums[mid]);

        node.left = construct(nums, l, mid - 1);
        node.right = construct(nums, mid + 1, r);

        return node;
    }

    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        Map<Integer, List<Integer>> graph = new HashMap<>();

        for (int[] edge : edges) {
            graph.putIfAbsent(edge[0], new ArrayList<>());
            graph.get(edge[0]).add(edge[1]);

            graph.putIfAbsent(edge[1], new ArrayList<>());
            graph.get(edge[1]).add(edge[0]);
        }

        List<Integer> leaves = new ArrayList<>();

        List<Integer> temp = new ArrayList<>();
        while(leaves.size() < n) {
            temp = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                // leaf 노드
                if (graph.get(i).size() == 1) {
                    temp.add(i);
                }
            }

            for (Integer leaf : temp) {
                int v = graph.get(leaf).get(0);
                graph.get(v).remove(Integer.valueOf(leaf));
                graph.get(leaf).remove(Integer.valueOf(v));
            }
            leaves.addAll(temp);
        }

        for (int i = 0; i < n; i++) {
            System.out.println(graph.get(i).size());
        }
        return temp;
    }
}

    public boolean isBalanced(TreeNode root) {
        return dfs4(root) != -1 ? true : false;
    }

    private int dfs4(TreeNode node) {
        if(node == null) return 0;

        int left = dfs4(node.left);
        int right = dfs4(node.right);

        if (Math.abs(left - right) > 1 || left == -1 || right == -1) {
            return -1;
        }

        return Math.max(left, right) + 1;
    }

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if(root == null) return null;

        StringBuilder sb = new StringBuilder();
        Queue<TreeNode> q = new LinkedList<>();

        q.add(root);
        sb.append("#,");
        sb.append(root.val);

        while (!q.isEmpty()) {
            TreeNode cur = q.poll();

            if (cur.left != null) {
                q.add(cur.left);
                sb.append("," + cur.left.val);
            } else
                sb.append("," + "#");
            if (cur.right != null) {
                q.add(cur.right);
                sb.append("," + cur.right.val);
            } else
                sb.append("," + "#");
        }
        System.out.println(sb.toString());
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if(data.equals("")) return null;

        String[] nodes = data.split(",");

        Queue<TreeNode> q = new LinkedList<>();
        TreeNode root = new TreeNode(Integer.parseInt(nodes[0]));
        q.add(root);

        int idx = 2;
        while (!q.isEmpty()) {
            TreeNode cur = q.poll();

            if (nodes[idx].equals("#")) {
                cur.left = new TreeNode(Integer.parseInt(nodes[idx]));
                q.add(cur.left);
            }

            idx++;
            if (nodes[idx].equals("#")) {
                cur.right = new TreeNode(Integer.parseInt(nodes[idx]));
                q.add(cur.right);
            }

            idx++;
        }
        return root;
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        TreeNode result = new TreeNode();
        return dfs3(root1, root2);
    }

    private TreeNode dfs3(TreeNode root1, TreeNode root2) {
        if(root1 == null) return root2;
        if(root2 == null) return root1;

        TreeNode node = null;

        node.left = dfs3(root1.left, root2.left);
        node.right = dfs3(root1.right, root2.right);

        node.val = root1.val + root2.val;

        return node;

    }

    int result = 0;

    public int longestUnivaluePath(TreeNode root) {
        dfs2(root);
    }

    private int dfs2(TreeNode node) {

        if(node == null)
            return 0;

        int left = dfs2(node.left);
        int right = dfs2(node.right);

        if (node.left != null && node.left.val == node.val) {
            left += 1;
        } else
            left = 0;

        if (node.right != null && node.right.val == node.val) {
            right += 1;
        } else
            right = 0;

        result = Math.max(result, left + right);

        return Math.max(left, right);
    }

    int longest = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        dfs1(root);
        return longest;
    }

    private int dfs1(TreeNode node) {

        if(node == null)
            return -1;

        int left = dfs1(node.left);
        int right = dfs1(node.right);

        longest = Math.max(longest, left + right + 2);

        return Math.max(left,right) + 1;
    }

    public int maxDepth(TreeNode root) {
        int depth = 0;
        if(root == null)
            return 0;

        Queue<TreeNode> q = new ArrayDeque<>();
        q.add(root);

        while (!q.isEmpty()) {
            depth += 1;
            int q_size = q.size();

            for (int i = 0; i < q_size; i++) {
                TreeNode cur = q.poll();
                if (cur.left != null) {
                    q.add(cur.left);
                }
                if (cur.right != null) {
                    q.add(cur.right);
                }
            }
        }
        return depth;
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
