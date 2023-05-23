#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <unordered_map>

using namespace std;

//  Definition for a binary tree node.
struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

//  Definition for singly-linked list.
struct ListNode
{
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// 279. Perfect Squares
class Solution
{
public:
    int numSquares(int n)
    {
        vector<int> dp(n + 1, INT_MAX);
        dp[0] = 0;
        int count = 1;
        while (count * count <= n)
        {
            int sq = count * count;
            for (int i = sq; i < n + 1; i++)
            {
                dp[i] = min(dp[i], dp[i - sq] + 1);
            }
            count++;
        }
        return dp[n];
    }
};

// 111
class Solution
{
public:
    int minDepth(TreeNode *root)
    {
        if (root == NULL)
        {
            return 0;
        }
        int left = minDepth(root->left);
        int right = minDepth(root->right);
        // here can not use 1 + (min(left, right) || max(left, right)), since params between || will be converted to boolean (false - 0, true - 1), and result would be 1 + 0(false) or 1(true)
        return 1 + (min(left, right) ? min(left, right) : max(left, right));
        //          ^ turn to boolean     ^ integer          ^ integer
    }
};

class Solution
{
public:
    int minDepth(TreeNode *root)
    {
        if (!root)
        {
            return 0;
        }
        queue<TreeNode *> q;
        q.push(root);
        int level = 0;
        while (!q.empty())
        {
            int size = q.size();
            level++;
            for (int i = 0; i < size; i++)
            {
                TreeNode *node = q.front();
                q.pop();
                if (node->left)
                    q.push(node->left);
                if (node->right)
                    q.push(node->right);
                if (!node->left && !node->right)
                    return level;
            }
        }
        return level;
    }
};

// 94
class Solution
{
public:
    void inorder(TreeNode *root, vector<int> &vct)
    {
        if (root->left)
            inorder(root->left, vct);
        vct.push_back(root->val);
        if (root->right)
            inorder(root->right, vct);
    }

    vector<int> inorderTraversal(TreeNode *root)
    {
        vector<int> vct;
        if (root == NULL)
        {
            return vct;
        }
        inorder(root, vct);
        return vct;
    }
};

class Solution
{
public:
    vector<int> inorderTraversal(TreeNode *root)
    {
        vector<int> ans;
        if (root == NULL)
            return ans;
        stack<TreeNode *> st;
        while (!st.empty() || root != NULL)
        {
            while (root != NULL)
            {
                st.push(root);
                root = root->left;
            }
            TreeNode *temp = st.top();
            st.pop();
            ans.push_back(temp->val);
            root = temp->right;
        }
        return ans;
    }
};

// 144
class Solution
{
public:
    void preorder(TreeNode *root, vector<int> &vct)
    {
        vct.push_back(root->val);
        if (root->left)
            preorder(root->left, vct);
        if (root->right)
            preorder(root->right, vct);
    }

    vector<int> preorderTraversal(TreeNode *root)
    {
        if (root == NULL)
        {
            return {};
        }
        vector<int> vct;
        preorder(root, vct);
        return vct;
    }
};

class Solution
{
public:
    vector<int> preorderTraversal(TreeNode *root)
    {
        vector<int> ans;
        stack<TreeNode *> st;
        while (!st.empty() || root != NULL)
        {
            while (root != NULL)
            {
                st.push(root);
                ans.push_back(root->val);
                root = root->left;
            }
            TreeNode *temp = st.top();
            st.pop();
            root = temp->right;
        }
        return ans;
    }
};

// 145
class Solution
{
public:
    void postorder(TreeNode *root, vector<int> &vct)
    {
        if (root->left)
            postorder(root->left, vct);
        if (root->right)
            postorder(root->right, vct);
        vct.push_back(root->val);
    }
    vector<int> postorderTraversal(TreeNode *root)
    {
        vector<int> vct;
        if (root == NULL)
        {
            return vct;
        }
        postorder(root, vct);
        return vct;
    }
};

class Solution
{
public:
    vector<int> postorderTraversal(TreeNode *root)
    {
        if (root == NULL)
        {
            return {};
        }
        vector<int> ans;
        stack<TreeNode *> st;
        TreeNode *prev = NULL;
        while (!st.empty() || root != NULL)
        {
            while (root != NULL)
            {
                st.push(root);
                root = root->left;
            }
            root = st.top();
            st.pop();
            if (root->right && root->right != prev)
            {
                st.push(root);
                root = root->right;
            }
            else
            {
                ans.push_back(root->val);
                prev = root;
                root = NULL;
            }
        }
        return ans;
    }
};

// 1
class Solution
{
public:
    vector<int> twoSum(vector<int> &nums, int target)
    {
        for (int i = 0; i < nums.size(); i++)
        {
            for (int j = i + 1; j < nums.size(); j++)
            {
                if (nums[i] + nums[j] == target)
                {
                    return {i, j};
                }
            }
        }
        return {-1, -1};
    }
};

class Solution
{
public:
    vector<int> twoSum(vector<int> &nums, int target)
    {
        unordered_map<int, int> mp;
        for (int i = 0; i < nums.size(); i++)
        {
            if (mp.find(target - nums[i]) == mp.end())
            {
                mp[nums[i]] = i;
            }
            else
            {
                return {mp[target - nums[i]], i};
            }
        }
        return {-1, -1};
    }
};

// 20
class Solution
{
public:
    bool isValid(string s)
    {
        stack<char> stk;
        unordered_map<char, char> mp;
        mp.insert(make_pair('}', '{'));
        mp.insert(make_pair(']', '['));
        mp.insert(make_pair(')', '('));
        for (char c : s)
        {
            if (mp.find(c) == mp.end())
            {
                stk.push(c);
            }
            else
            {
                if (stk.empty())
                {
                    return false;
                }
                else if (stk.top() == mp[c])
                {
                    stk.pop();
                }
                else
                {
                    return false;
                }
            }
        }
        return stk.empty();
    }
};

// 21
class Solution
{
public:
    ListNode *merge(ListNode *list1, ListNode *list2)
    {
        if (list1 == NULL)
            return list2;
        if (list2 == NULL)
            return list1;
        if (list1->val < list2->val)
        {
            list1->next = merge(list1->next, list2);
            return list1;
        }
        else
        {
            list2->next = merge(list1, list2->next);
            return list2;
        }
    }
    ListNode *mergeTwoLists(ListNode *list1, ListNode *list2)
    {
        return merge(list1, list2);
    }
};

class Solution
{
public:
    ListNode *mergeTwoLists(ListNode *list1, ListNode *list2)
    {
        if (!list1)
            return list2;
        if (!list2)
            return list1;
        ListNode *ptr = list1;
        if (list1->val <= list2->val)
        {
            list1 = list1->next;
        }
        else
        {
            ptr = list2;
            list2 = list2->next;
        }
        ListNode *cur = ptr;
        while (list1 && list2)
        {
            if (list1->val <= list2->val)
            {
                cur->next = list1;
                list1 = list1->next;
            }
            else
            {
                cur->next = list2;
                list2 = list2->next;
            }
            cur = cur->next;
        }
        if (!list1)
            cur->next = list2;
        if (!list2)
            cur->next = list1;
        return ptr;
    }
};

// 121
class Solution
{
public:
    int maxProfit(vector<int> &prices)
    {
        if (prices.empty())
            return 0;
        int profit = 0;
        int lowest = prices[0];
        for (int i = 1; i < prices.size(); i++)
        {
            if (prices[i] - lowest > profit)
            {
                profit = prices[i] - lowest;
            }
            lowest = min(lowest, prices[i]);
        }
        return profit;
    }
};

class Solution
{
public:
    int maxProfit(vector<int> &prices)
    {
        int n = prices.size();
        vector<int> maxPrices(n, 0);
        maxPrices[n - 1] = prices[n - 1];
        for (int i = n - 2; i >= 0; i--)
        {
            maxPrices[i] = max(maxPrices[i + 1], prices[i]);
        }
        int maxProfit = 0;
        for (int i = 0; i < n; i++)
        {
            maxProfit = max(maxProfit, maxPrices[i] - prices[i]);
        }
        return maxProfit;
    }
};

// 125
class Solution
{
public:
    bool isPalindrome(string s)
    {
        int start = 0, end = s.length() - 1;
        while (start < end)
        {
            if (!isalnum(s[start]))
                start++;
            else if (!isalnum(s[end]))
                end--;
            else
            {
                if (tolower(s[start++]) != tolower(s[end--]))
                    return false;
            }
        }
        return true;
    }
};

// TLE
class Solution
{
public:
    bool checkPalindrome(vector<char> ch, int start)
    {
        int n = ch.size();
        if (start >= n / 2)
            return true;
        if (ch[start] == ch[n - 1 - start])
            return checkPalindrome(ch, start + 1);
        return false;
    }
    bool isPalindrome(string s)
    {
        vector<char> ch;
        for (char c : s)
        {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
            {
                ch.push_back(c);
            }
            if (c >= 'A' && c <= 'Z')
            {
                ch.push_back((char)tolower(c));
            }
        }
        return checkPalindrome(ch, 0);
    }
};

// 226
class Solution
{
public:
    TreeNode *invertTree(TreeNode *root)
    {
        if (root == NULL)
            return NULL;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty())
        {
            TreeNode *cur = q.front();
            q.pop();
            TreeNode *left = cur->left;
            cur->left = cur->right;
            cur->right = left;
            if (cur->left)
                q.push(cur->left);
            if (cur->right)
                q.push(cur->right);
        }
        return root;
    }
};

class Solution
{
public:
    TreeNode *invertTree(TreeNode *root)
    {
        if (!root)
            return NULL;
        TreeNode *left = root->left;
        root->left = invertTree(root->right);
        root->right = invertTree(left);
        return root;
    }
};

// 242
class Solution
{
public:
    bool isAnagram(string s, string t)
    {
        if (s.length() != t.length())
            return false;
        sort(s.begin(), s.end());
        sort(t.begin(), t.end());
        return s == t;
    }
};

class Solution
{
public:
    bool isAnagram(string s, string t)
    {
        if (s.length() != t.length())
            return false;
        int arr[26] = {0};
        for (int i = 0; i < s.length(); i++)
        {
            arr[s[i] - 'a']++;
            arr[t[i] - 'a']--;
        }
        for (int i = 0; i < sizeof(arr) / sizeof(int); i++)
        {
            if (arr[i] != 0)
                return false;
        }
        return true;
    }
};

// 704
class Solution
{
public:
    int search(vector<int> &nums, int target)
    {
        int l = 0;
        int r = nums.size() - 1;
        while (l < r)
        {
            int mid = (l + r + 1) / 2;
            if (nums[mid] <= target)
                l = mid;
            else
                r = mid - 1;
        }
        if (nums[l] == target)
            return l;
        return -1;
    }
};

class Solution
{
public:
    int search(vector<int> &nums, int target)
    {
        int l = 0;
        int r = nums.size() - 1;
        while (l < r)
        {
            int mid = (l + r) / 2;
            if (nums[mid] >= target)
                r = mid;
            else
                l = mid + 1;
        }
        if (nums[l] == target)
            return l;
        return -1;
    }
};

// 733
class Solution
{
public:
    vector<vector<int>> floodFill(vector<vector<int>> &image, int sr, int sc, int color)
    {
        vector<vector<int>> img = image;
        if (img[sr][sc] == color)
            return img;
        int c = img[sr][sc];
        vector<pair<int, int>> del;
        del.push_back({-1, 0});
        del.push_back({1, 0});
        del.push_back({0, -1});
        del.push_back({0, 1});
        int m = img.size();
        int n = img[0].size();
        queue<pair<int, int>> q;
        q.push({sr, sc});
        while (!q.empty())
        {
            int row = q.front().first;
            int col = q.front().second;
            q.pop();
            img[row][col] = color;
            for (pair<int, int> p : del)
            {
                if (row + p.first >= 0 && row + p.first < m && col + p.second >= 0 && col + p.second < n && img[row + p.first][col + p.second] == c)
                {
                    q.push({row + p.first, col + p.second});
                }
            }
        }
        return img;
    }
};

class Solution
{
public:
    void dfs(vector<vector<int>> &image, int m, int n, int sr, int sc, int c, int color)
    {
        if (sr >= 0 && sr < m && sc >= 0 && sc < n && image[sr][sc] == c)
        {
            image[sr][sc] = color;
            dfs(image, m, n, sr - 1, sc, c, color);
            dfs(image, m, n, sr + 1, sc, c, color);
            dfs(image, m, n, sr, sc - 1, c, color);
            dfs(image, m, n, sr, sc + 1, c, color);
        }
    }

    vector<vector<int>> floodFill(vector<vector<int>> &image, int sr, int sc, int color)
    {
        vector<vector<int>> img = image;
        if (image[sr][sc] == color)
            return img;
        dfs(img, img.size(), img[0].size(), sr, sc, image[sr][sc], color);
        return img;
    }
};

// 235
class Solution
{
public:
    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
    {
        if (!root)
            return NULL;
        if (root == p || root == q)
            return root;
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);
        if (left != NULL && right == NULL)
            return left;
        if (right != NULL && left == NULL)
            return right;
        if (left != NULL && right != NULL)
            return root;
        return NULL;
    }
};

class Solution
{
public:
    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
    {
        int small = min(p->val, q->val);
        int big = max(p->val, q->val);
        while (root)
        {
            if (root->val < small)
                root = root->right;
            else if (root->val > big)
                root = root->left;
            else
                return root;
        }
        return root;
    }
};

// 110
class Solution
{
public:
    bool ans = true;
    int countDepth(TreeNode *root)
    {
        if (!ans)
            return 0;
        if (root == NULL)
            return 0;
        int leftDepth = countDepth(root->left);
        int rightDepth = countDepth(root->right);
        if (abs(leftDepth - rightDepth) > 1)
            ans = false;
        return max(leftDepth, rightDepth) + 1;
    }
    bool isBalanced(TreeNode *root)
    {
        countDepth(root);
        return ans;
    }
};

// 141
class Solution
{
public:
    bool hasCycle(ListNode *head)
    {
        ListNode *fast = head;
        ListNode *slow = head;
        while (fast != NULL && fast->next != NULL)
        {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow)
                return true;
        }
        return false;
    }
};

// 232
class MyQueue
{
public:
    stack<int> input, output;

    MyQueue()
    {
    }

    void push(int x)
    {
        input.push(x);
    }

    int pop()
    {
        moveToQueue();
        int val = output.top();
        output.pop();
        return val;
    }

    int peek()
    {
        moveToQueue();
        return output.top();
    }

    bool empty()
    {
        return input.empty() && output.empty();
    }

    void moveToQueue()
    {
        if (output.empty())
        {
            while (!input.empty())
            {
                output.push(input.top());
                input.pop();
            }
        }
    }
};

// 278
class Solution
{
public:
    int firstBadVersion(int n)
    {
        int left = 0, right = n;
        while (left < right)
        {
            int mid = left + (right - left) / 2;
            if (isBadVersion(mid))
                right = mid;
            else
                left = mid + 1;
        }
        return left;
    }
};

// 383
class Solution
{
public:
    bool canConstruct(string ransomNote, string magazine)
    {
        if (magazine.length() < ransomNote.length())
            return false;
        int chars[26] = {0}; // the length is fixed, so array is used
        for (int i = 0; i < magazine.length(); i++)
        {
            chars[int(magazine[i]) - int('a')] += 1;
        }
        for (int j = 0; j < ransomNote.length(); j++)
        {
            if (chars[int(ransomNote[j]) - int('a')] == 0)
                return false;
            chars[int(ransomNote[j]) - int('a')] -= 1;
        }
        return true;
    }
};

class Solution
{
public:
    bool canConstruct(string ransomNote, string magazine)
    {
        int m = ransomNote.size();
        int n = magazine.size();
        unordered_map<char, int> map;
        for (char c : magazine)
        {
            map[c]++;
        }
        for (char c : ransomNote)
        {
            map[c]--;
        }
        for (auto pair : map)
        {
            if (pair.second < 0)
                return false;
        }
        return true;
    }
};

// 70

// fastest
class Solution
{
public:
    int dp(int n, vector<int> *cache)
    {
        if (cache->at(n) != -1)
            return cache->at(n);
        int res = 0;
        if (n - 1 > 0)
            res += dp(n - 1, cache);
        if (n - 2 > 0)
            res += dp(n - 2, cache);
        cache->at(n) = res;
        return cache->at(n);
    }
    int climbStairs(int n)
    {
        vector<int> cache(n + 1, -1);
        if (n >= 1)
            cache[1] = 1;
        if (n >= 2)
            cache[2] = 2;
        if (n == 1 || n == 2)
            return cache[n];
        dp(n, &cache);
        return cache[n];
    }
};

// time: O(n), space: O(n) + O(n)
class Solution
{
public:
    int dp(int n, vector<int> &cache)
    {
        if (n <= 2)
            return n;
        if (cache[n] != -1)
            return cache[n];
        cache[n] = dp(n - 1, cache) + dp(n - 2, cache);
        return cache[n];
    }
    int climbStairs(int n)
    {
        vector<int> cache(n + 1, -1);
        if (n >= 1)
            cache[1] = 1;
        if (n >= 2)
            cache[2] = 2;
        if (n == 1 || n == 2)
            return cache[n];
        dp(n, cache);
        return cache[n];
    }
};

// time: O(n), space: O(n)
class Solution
{
public:
    int climbStairs(int n)
    {
        if (n <= 2)
            return n;
        vector<int> dp(n + 1, 0);
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++)
            dp[i] = dp[i - 1] + dp[i - 2];
        return dp[n];
    }
};

// fastest
// time: O(n), space: O(1)
class Solution
{
public:
    int climbStairs(int n)
    {
        if (n <= 2)
            return n;
        int one_step = 2;
        int two_step = 1;
        int count = 0;
        for (int i = 3; i <= n; i++)
        {
            count = one_step + two_step;
            two_step = one_step;
            one_step = count;
        }
        return count;
    }
};

// 409
class Solution
{
public:
    int longestPalindrome(string s)
    {
        int evens = 0;
        int haveOdds = 0;
        unordered_map<char, int> map;
        for (int i = 0; i < s.length(); i++)
        {
            map[s[i]]++;
        }
        for (auto pair : map)
        {
            evens += 2 * (pair.second / 2);
            if (pair.second % 2 == 1)
                haveOdds = 1;
        }
        return evens + haveOdds;
    }
};

// 206
class Solution
{
public:
    ListNode *reverseList(ListNode *head)
    {
        ListNode *reversed = NULL;
        ListNode *temp = reversed;
        while (head != NULL)
        {
            reversed = head;
            head = head->next;
            reversed->next = temp;
            temp = reversed;
        }
        return reversed;
    }
};

class Solution
{
public:
    ListNode *reverseList(ListNode *head)
    {
        if (head == NULL || head->next == NULL)
            return head;
        ListNode *reversed = reverseList(head->next);
        head->next->next = head;
        head->next = NULL;
        return reversed;
    }
};

// 169
class Solution
{
public:
    int majorityElement(vector<int> &nums)
    {
        int ele = nums[0];
        int count = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            if (count == 0)
                ele = nums[i]; // all elements offseted, start again from index i
            count += (ele == nums[i]) ? 1 : -1;
        }
        return ele;
    }
};

// 67
class Solution
{
public:
    string addBinary(string a, string b)
    {
        int m = a.length();
        int n = b.length();
        int i = 1, j = 1;
        int carry = 0;
        string ans = "";
        while (i <= m && j <= n)
        {
            int sum = carry;
            sum += a[m - i] - '0';
            sum += b[n - j] - '0';
            carry = sum / 2;
            ans = to_string(sum % 2) + ans;
            i++;
            j++;
        }
        while (i <= m)
        {
            int sum = carry;
            sum += a[m - i] - '0';
            carry = sum / 2;
            ans = to_string(sum % 2) + ans;
            i++;
        }
        while (j <= n)
        {
            int sum = carry;
            sum += b[n - j] - '0';
            carry = sum / 2;
            ans = to_string(sum % 2) + ans;
            j++;
        }
        if (carry == 1)
            ans = "1" + ans;
        return ans;
    }
};