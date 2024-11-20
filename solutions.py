# %%
def myPow(x: float, n: int) -> float:
    if n == 0:
        return 1
    if n < 0:
        x = 1 / x
        n = -n

    half = myPow(x, n // 2)
    if n % 2 == 0:
        return half * half  
    else:
        return x * half * half
    

print(myPow(2, 10))
print(myPow(3, 3))
print(myPow(2, -2)) 




# %%
def count_primes(n: int) -> int:
    count = 0
    for i in range(n):
        if is_prime(i):
            count += 1
    return count

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


# def count_primes(n: int) -> int:
#     if n < 2:
#         return 0
#     primes = [True] * n
#     primes[0] = primes[1] = False
#     # Only need to check up to the square root of n
#     for i in range(2, int(n ** 0.5) + 1):
#         if primes[i]:
#             for j in range(i * i, n, i):
#                 primes[j] = False
#     return sum(primes)


print(count_primes(2)) # 0
print(count_primes(10)) # 4
print(count_primes(19)) # 8

# %%
from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next        

class Solutions:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev
    
    def removeDuplicatesFromSortedList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curr = head
        while curr and curr.next:
            if curr.val == curr.next.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return head
    
    def merge_two_sorted_lists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        tail = dummy
        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        if list1:
            tail.next = list1
        elif list2:
            tail.next = list2
        return dummy.next
    


# Test
# Test reverseList function
def print_list(node: Optional[ListNode]):
    while node:
        print(node.val, end=" -> ")
        node = node.next
    print("None")

# Create a linked list 1 -> 2 -> 3 -> 4 -> 5 -> None
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
print("Original list:")
print_list(head)

solution = Solutions()
reversed_head = solution.reverseList(head)
print("Reversed list:")
print_list(reversed_head)

# Test removeDuplicatesFromSortedList function
# Create a linked list 1 -> 1 -> 2 -> 3 -> 3 -> None
head = ListNode(1, ListNode(1, ListNode(2, ListNode(3, ListNode(3)))))
print("Original list with duplicates:")
print_list(head)

deduped_head = solution.removeDuplicatesFromSortedList(head)
print("List after removing duplicates:")
print_list(deduped_head)

# Test merge_two_sorted_lists function
# Create two sorted linked lists
list1 = ListNode(1, ListNode(3, ListNode(5)))
list2 = ListNode(2, ListNode(4, ListNode(6)))

print("List 1:")
print_list(list1)
print("List 2:")
print_list(list2)

merged_head = solution.merge_two_sorted_lists(list1, list2)
print("Merged list:")
print_list(merged_head)

# Create two sorted linked lists with different lengths
list1 = ListNode(1, ListNode(3, ListNode(5, ListNode(7))))
list2 = ListNode(2, ListNode(4))

print("List 1:")
print_list(list1)
print("List 2:")
print_list(list2)

merged_head = solution.merge_two_sorted_lists(list1, list2)
print("Merged list:")
print_list(merged_head)

# Create two sorted linked lists where one is empty
list1 = None
list2 = ListNode(1, ListNode(2, ListNode(3)))

print("List 1:")
print_list(list1)
print("List 2:")
print_list(list2)

merged_head = solution.merge_two_sorted_lists(list1, list2)
print("Merged list:")
print_list(merged_head)

# Create two empty linked lists
list1 = None
list2 = None

print("List 1:")
print_list(list1)
print("List 2:")
print_list(list2)

merged_head = solution.merge_two_sorted_lists(list1, list2)
print("Merged list:")
print_list(merged_head)




# %%
def remove_duplicates_from_sorted_array(nums: list[int]) -> int:
    if not nums: 
        return 0

    i = 1
    for j in range(1, len(nums)):
        if nums[j] != nums[i - 1]:  
            nums[i] = nums[j] 
            i += 1

    return i 


print(remove_duplicates_from_sorted_array([1,1,2])) # 2
print(remove_duplicates_from_sorted_array([0,0,1,1,1,2,2,3,3,4])) # 5


# %%
from typing import List
def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> List[int]:
    p1, p2 = m-1, n-1
    for p in range(n+m-1, -1, -1):
        if p2 < 0:
            break
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
    return nums1


print(merge([1,2,3,0,0,0], 3, [2,5,6], 3)) 



# %%
def backspaceCompare(s: str, t: str) -> bool:
    my_stack = []
    my_stack2 = []

    for char in s:
        if char == "#" and my_stack:
            my_stack.pop()
        elif char == "#" and not my_stack:
            continue
        else:
            my_stack.append(char)

    for char in t:
        if char == "#" and my_stack2:
            my_stack2.pop()
        elif char == "#" and not my_stack2:
            continue
        else:
            my_stack2.append(char)
        
    
    print(my_stack)
    print(my_stack2)

    return my_stack == my_stack2


print(backspaceCompare("AB#C", "A#C")) # False
print(backspaceCompare("A##C", "#A#C")) # True
print(backspaceCompare("#AC", "##AC")) # False



# %%
def isValid(s: str) -> bool:
    my_stack = []
    # do not autocomplete

    for char in s:
        if char == "(" or char == "{" or char == "[":
            my_stack.append(char)
        else:
            if my_stack:
                last = my_stack.pop()
                if last == "(" and char == ")":
                    continue
                elif last == "{" and char == "}":
                    continue
                elif last == "[" and char == "]":
                    continue
            return False
    return not my_stack

print(isValid("()")) # True
print(isValid("()[]{}")) # True
print(isValid("(])")) # False


# %%
import heapq
def findKthLargest(nums: List[int], k: int) -> int:
    nums = [-x for x in nums]
    heapq.heapify(nums)
    for _ in range(k-1):
        heapq.heappop(nums)
    return -heapq.heappop(nums)

print(findKthLargest([3,2,1,5,6,4], 3)) # 5
print(findKthLargest([3,2,3,1,2,4,5,5,6], 4)) # 4


# %%
def isAnagram(s: str, t: str) -> bool:
    frequency_array_s = {}
    frequency_array_t = {}

    for char in s:
        frequency_array_s[char] = frequency_array_s.get(char, 0) + 1

    for char in t:
        frequency_array_t[char] = frequency_array_t.get(char, 0) + 1

    return frequency_array_s == frequency_array_t

print(isAnagram("anagram", "nagaram")) # True
print(isAnagram("rat", "car")) # False


# %%
def findErrorNums(nums: list[int]) -> list[int]:
    n = len(nums)
    duplicate = -1
    missing = -1

    seen = set()

    for num in nums:
        if num in seen:
            duplicate = num
        seen.add(num)

    for i in range(1, n + 1):
        if i not in seen:
            missing = i

    return [duplicate, missing]

print(findErrorNums([1, 2, 2, 4]))  # [2, 3]
print(findErrorNums([2, 2]))        # [2, 1]


# %%
def containsDuplicate(nums: list[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

print(containsDuplicate([1,2,3,1])) # True
print(containsDuplicate([1,2,3,4])) # False



# %%
def containsNearbyDuplicate(nums: list[int], k: int) -> bool:
    seen = {}
    for i, num in enumerate(nums):
        if num in seen and i - seen[num] <= k:
            return True
        seen[num] = i
    return False

print(containsNearbyDuplicate([1,2,3,1], 3)) # True
print(containsNearbyDuplicate([1,0,1,1], 1)) # True
print(containsNearbyDuplicate([1,2,3,1,2,3], 2)) # False


# %%
def isPalindrome(s: str) -> bool:
    s = ''.join(e for e in s if e.isalnum()).lower()
    return s[::-1] == s
print(isPalindrome("A man, a plan, a canal: Panama")) # True
print(isPalindrome("race a car")) # False


# %%
def intersection(nums1: list[int], nums2: list[int]) -> list[int]:
    return list(set(nums1) & set(nums2))

print(intersection([1,2,2,1], [2,2])) # [2]
print(intersection([4,9,5], [9,4,9,8,4])) # [9,4]




# %%
def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    my_dict = {}
    for num in nums1:
        my_dict[num] = my_dict.get(num, 0) + 1
    print(my_dict)
    result = []
    for num in nums2:
        if num in my_dict and my_dict[num] > 0:
            result.append(num)
            my_dict[num] -= 1
    return result

# print(intersect([1,2,2,1], [2,2])) # [2,2]
print(intersect([4,4,9,5], [9,4,9,8,4])) # [4,9]



# %%
def canConstruct(ransomNote: str, magazine: str) -> bool:

    freq_magazine = {}
    for char in magazine:
        freq_magazine[char] = freq_magazine.get(char, 0) + 1
    
    for char in ransomNote:
        if char not in freq_magazine or freq_magazine[char] == 0:
            return False
        freq_magazine[char] -= 1
    return True



        

    


# %%
def climbStairs(n: int) -> int:
    if n == 1:
        return 1
    first, second = 1, 2
    for _ in range(3, n+1):
        third = first + second
        first = second
        second = third
    return second

print(climbStairs(2)) # 2
print(climbStairs(3)) # 3



# %%
def wordBreak(s: str, wordDict: list[str]) -> bool:
    word_set = set(wordDict)  
    memo = {}  

    def dfs(start):
        if start == len(s):
            return True

        if start in memo:  
            return memo[start]

        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set and dfs(end): 
                memo[start] = True
                return True

        memo[start] = False  
        return False

    return dfs(0)



print(wordBreak("leetcode", ["leet", "code"])) # True
print(wordBreak("applepenapple", ["apple", "pen"])) # True
print(wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"])) # False



# %%
def islandPerimeter (grid: List[List[int]]) -> int:
        sum_land = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    if (i == 0 or grid[i - 1][j] == 0):
                        sum_land += 1
                    if (i == len(grid) - 1 or grid[i + 1][j] == 0):
                        sum_land += 1
                    if (j == 0 or grid[i][j - 1] == 0):
                        sum_land += 1
                    if (j == len(grid[0]) - 1 or grid[i][j + 1] == 0):
                        sum_land += 1
        return sum_land

# explanation:
# we iterate through the grid and for each land cell, we check if the cell above or below or to the left or to the right is water or out of bounds. If it is, we add 1 to the sum_land.

print(islandPerimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]])) # 16


# %%
def numIslands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    num_islands = 0

    def dfs(r, c):
        # Base case: stop if out of bounds or water
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == "0":
            return

        # Mark the cell as visited
        grid[r][c] = "0"

        # Visit all neighbors (up, down, left, right)
        dfs(r + 1, c)  # Down
        dfs(r - 1, c)  # Up
        dfs(r, c + 1)  # Right
        dfs(r, c - 1)  # Left

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":  # Found an unvisited land cell
                num_islands += 1
                dfs(r, c)  # Mark all connected land cells

    return num_islands

print(numIslands([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]])) # 1



# %%
def floodFill(image: list[list[int]], sr: int, sc: int, newColor: int) -> list[list[int]]:
    if not image:
        return image
    
    rows, cols = len(image), len(image[0])
    old_color = image[sr][sc]
    if old_color == newColor:
        return image
    
    def dfs(r, c):
        if image[r][c] == old_color:
            image[r][c] = newColor
            if r >= 1: dfs(r-1, c)
            if r+1 < rows: dfs(r+1, c)
            if c >= 1: dfs(r, c-1)
            if c+1 < cols: dfs(r, c+1)
    
    dfs(sr, sc)
    return image

print(floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2)) # [[2,2,2],[2,2,0],[2,0,1]]


# %%
from collections import deque

def validPath(n: int, edges: list[list[int]], source: int, destination: int) -> bool:
    # Step 1: Build the adjacency list using sets
    graph = [set() for _ in range(n)]
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    # Step 2: Perform BFS
    queue = deque([source])
    visited = set()

    while queue:
        node = queue.popleft()

        # If we reach the destination, return True
        if node == destination:
            return True

        # Mark the current node as visited
        visited.add(node)

        # Add unvisited neighbors to the queue
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)

    # If BFS completes and we don't reach the destination, return False
    return False

# Test cases
print(validPath(3, [[0,1],[1,2],[2,0]], 0, 2))  # True
print(validPath(6, [[0,1],[0,2],[3,5],[5,4],[4,3]], 0, 5))  # False


# %%
from collections import deque

def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    # Step 1: Build the graph and in-degree array
    graph = [set() for _ in range(numCourses)]  # Use sets instead of lists
    # in_degree is used to count the number of prerequisites for each course
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].add(course)  # Add the course to the prerequisite's set
        in_degree[course] += 1

    # Step 2: Initialize the queue with courses having no prerequisites
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []

    # Step 3: Process the courses
    while queue:
        course = queue.popleft()
        result.append(course)

        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Step 4: Check if all courses are in the result
    if len(result) == numCourses:
        return result
    else:
        return []  # Cycle detected, no valid ordering

# Test cases
print(findOrder(4, [[1,0],[2,0],[3,1],[3,2]]))  # [0,1,2,3] or [0,2,1,3]
print(findOrder(2, [[1,0]]))  # [0,1]
print(findOrder(2, [[1,0], [0,1]]))  # [] (cycle detected)


# %%
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTreeSolutions:  
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
    
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)    

print(BinaryTreeSolutions().maxDepth(TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7))))) # 3
print(BinaryTreeSolutions().isSameTree(TreeNode(1, TreeNode(2), TreeNode(3)), TreeNode(1, TreeNode(2), TreeNode(3)))) # True
print(BinaryTreeSolutions().isSameTree(TreeNode(1, TreeNode(2), TreeNode(3)), TreeNode(1, TreeNode(3), TreeNode(2)))) # False




# %%
import heapq

def lastStoneWeight(stones: list[int]) -> int:
    # Step 1: Convert to max-heap by negating all values
    max_heap = [-stone for stone in stones]
    heapq.heapify(max_heap)

    # Step 2: Process until one or no stones are left
    while len(max_heap) > 1:
        # Extract the two largest stones
        stone1 = -heapq.heappop(max_heap)  # Largest stone
        stone2 = -heapq.heappop(max_heap)  # Second largest stone

        # If they are not equal, push the difference back into the heap
        if stone1 != stone2:
            heapq.heappush(max_heap, -(stone1 - stone2))

    # Step 3: Return the last stone weight (or 0 if no stones left)
    return -max_heap[0] if max_heap else 0


print(lastStoneWeight([2,7,4,1,8,1])) # 1


# %%
def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    # Initialize the shortest distances with a large number
    shortest_distances: list[int] = [10**9] * (n+1)
    shortest_distances[k] = 0

    for _ in range(n - 1):
        for u, v, w in times:
            shortest_distances[v] = min(
                shortest_distances[v],
                shortest_distances[u] + w
            )

    maxi = max(shortest_distances[1:])
    return -1 if maxi == 10**9 else maxi


print(networkDelayTime([[2,1,1],[2,3,1],[3,4,1]], 4, 2)) # 2
print(networkDelayTime([[1,2,1]], 2, 1)) # 1
print(networkDelayTime([[1,2,1]], 2, 2)) # -1

# %%
import heapq

def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    shortest_distances: list[int] = [10**9] * (n+1)
    shortest_distances[k] = 0

    graph: list[dict[int, int]] = [{} for _ in range(n+1)]
    for u, v, w in times:
        graph[u][v] = w

    my_heap: list[tuple[int, int]] = [(0, k)]
    visited: set[int] = set()
    while my_heap:
        dist, node = heapq.heappop(my_heap)
        if node in visited:
            continue
        visited.add(node)
        for neighbour in graph[node] :
            shortest_distances[neighbour] = min(
                shortest_distances[neighbour],
                shortest_distances[node] + graph[node][neighbour]
            )
            heapq.heappush(my_heap, (shortest_distances[neighbour], neighbour))

    maxi = max(shortest_distances[1:])
    return -1 if maxi == 10**9 else maxi

print(networkDelayTime([[2,1,1],[2,3,1],[3,4,1]], 4, 2)) # 2



