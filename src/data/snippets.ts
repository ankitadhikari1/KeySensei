export type LanguageKey = "c" | "cpp" | "java" | "python" | "javascript"

export interface Snippet {
  id: string
  language: LanguageKey
  title: string
  code: string
}

interface TemplateFactory {
  (variant: number, index: number): { title: string; code: string }
}

const SNIPPET_COUNT_PER_LANGUAGE = 120

export const LEVEL_SEGMENTS = [24, 36, 32, 28] as const

const formatCode = (code: string): string => {
  const trimmed = code.replace(/^\n/, "").replace(/\s+$/, "")
  const lines = trimmed.split("\n")
  const indents = lines
    .filter((line) => line.trim().length > 0)
    .map((line) => line.match(/^\s*/)?.[0].length ?? 0)
  const baseIndent = indents.length ? Math.min(...indents) : 0
  return lines
    .map((line) => (baseIndent ? line.slice(baseIndent) : line))
    .join("\n")
}

const buildNumericSequence = (seed: number, length: number, step: number): number[] =>
  Array.from({ length }, (_, index) => ((seed + 11) * (index + 3) * step) % 97 + 3)

const joinNumbers = (values: number[]) => values.join(", ")

const createSnippetSet = (
  language: LanguageKey,
  templates: TemplateFactory[],
  count = SNIPPET_COUNT_PER_LANGUAGE,
  startOffset = 0,
): Snippet[] =>
  Array.from({ length: count }, (_, index) => {
    const template = templates[index % templates.length]
    const variant = Math.floor(index / templates.length) + 1
    const { title, code } = template(variant, index)
    const position = startOffset + index + 1
    return {
      id: `${language}-${position}`,
      language,
      title,
      code: formatCode(code),
    }
  })

const cppBasicTemplates: TemplateFactory[] = [
  (variant) => {
    const limit = 5 + (variant % 4)
    return {
      title: `C++ Warmup loop ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        int main() {
            for (int value = 1; value <= ${limit}; ++value) {
                cout << value << " ";
            }
            cout << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const size = 4 + (variant % 4)
    return {
      title: `C++ Array sum drill ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        int main() {
            vector<int> values = { ${joinNumbers(buildNumericSequence(variant + 2, size, 3))} };
            long long total = 0;
            for (int value : values) {
                total += value;
            }
            cout << "total: " << total << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const number = 3 + (variant % 5)
    return {
      title: `C++ Factorial basics ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        long long factorial(int value) {
            long long result = 1;
            for (int index = 2; index <= value; ++index) {
                result *= index;
            }
            return result;
        }

        int main() {
            cout << factorial(${number}) << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const text = `dojo${variant}`
    return {
      title: `C++ Reverse string ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        int main() {
            string word = "${text}";
            reverse(word.begin(), word.end());
            cout << word << "\n";
            return 0;
        }
`,
    }
  },
]

const cppAdvancedTemplates: TemplateFactory[] = [
  (variant) => {
    const values = buildNumericSequence(variant + 3, 10, 4)
    values.sort((a, b) => a - b)
    const target = values[(variant * 3) % values.length]
    return {
      title: `C++ Binary Search Duel ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        int binary_search_${variant}(const vector<int> &arr, int target) {
            int left = 0;
            int right = static_cast<int>(arr.size()) - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (arr[mid] == target) {
                    return mid;
                }
                if (arr[mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return -1;
        }

        int main() {
            vector<int> data = { ${joinNumbers(values)} };
            int target = ${target};
            int index = binary_search_${variant}(data, target);
            cout << "target: " << target << " index: " << index << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 5)
    return {
      title: `C++ BFS Arena ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        vector<int> breadth_first_${variant}(const vector<vector<int>> &graph, int start) {
            vector<int> order;
            vector<int> distance(graph.size(), -1);
            queue<int> q;
            q.push(start);
            distance[start] = 0;
            while (!q.empty()) {
                int node = q.front();
                q.pop();
                order.push_back(node);
                for (int neighbour : graph[node]) {
                    if (distance[neighbour] == -1) {
                        distance[neighbour] = distance[node] + 1;
                        q.push(neighbour);
                    }
                }
            }
            return order;
        }

        int main() {
            vector<vector<int>> graph(${nodes});
            for (int index = 0; index < ${nodes}; ++index) {
                graph[index].push_back((index + 1) % ${nodes});
                graph[index].push_back((index + ${2 + (variant % 3)}) % ${nodes});
            }
            vector<int> order = breadth_first_${variant}(graph, 0);
            for (int node : order) {
                cout << node << ' ';
            }
            cout << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const size = 7 + (variant % 6)
    const edges: string[] = []
    for (let index = 1; index < size; index += 1) {
      const parent = Math.floor((index - 1) / 2)
      edges.push(`tree[${parent}].push_back(${index});`)
    }
    return {
      title: `C++ DFS Expedition ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        void depth_first_${variant}(int node, const vector<vector<int>> &tree, vector<int> &order) {
            order.push_back(node);
            for (int next : tree[node]) {
                depth_first_${variant}(next, tree, order);
            }
        }

        int main() {
            vector<vector<int>> tree(${size});
${edges.map((line) => `            ${line}`).join("\n")}
            vector<int> order;
            depth_first_${variant}(0, tree, order);
            for (int node : order) {
                cout << node << ' ';
            }
            cout << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const nodes = 5 + (variant % 4)
    const edgesLines: string[] = []
    for (let node = 0; node < nodes; node += 1) {
      const to = (node + 1) % nodes
      const weight = ((variant + node) * 7) % 19 + 3
      edgesLines.push(`            graph[${node}].push_back({${to}, ${weight}});`)
      const alt = (node + 2 + (variant % 2)) % nodes
      const altWeight = ((variant + node) * 5) % 13 + 4
      edgesLines.push(`            graph[${node}].push_back({${alt}, ${altWeight}});`)
    }
    return {
      title: `C++ Dijkstra Clash ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        vector<int> dijkstra_${variant}(const vector<vector<pair<int, int>>> &graph, int start) {
            const int INF = 1e9;
            vector<int> distance(graph.size(), INF);
            using Node = pair<int, int>;
            priority_queue<Node, vector<Node>, greater<Node>> pq;
            distance[start] = 0;
            pq.push({0, start});
            while (!pq.empty()) {
                auto [dist, node] = pq.top();
                pq.pop();
                if (dist != distance[node]) {
                    continue;
                }
                for (auto [next, weight] : graph[node]) {
                    if (distance[next] > dist + weight) {
                        distance[next] = dist + weight;
                        pq.push({distance[next], next});
                    }
                }
            }
            return distance;
        }

        int main() {
            vector<vector<pair<int, int>>> graph(${nodes});
${edgesLines.map((line) => line).join("\n")}
            vector<int> distance = dijkstra_${variant}(graph, 0);
            for (int dist : distance) {
                cout << dist << ' ';
            }
            cout << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 4)
    const edges: string[] = []
    for (let node = 0; node < nodes - 1; node += 1) {
      edges.push(`            indegree[${(node + 1) % nodes}]++; graph[${node}].push_back(${(node + 1) % nodes});`)
      if (node + 2 < nodes) {
        edges.push(`            indegree[${(node + 2) % nodes}]++; graph[${node}].push_back(${(node + 2) % nodes});`)
      }
    }
    return {
      title: `C++ Topological Arena ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        vector<int> topological_order_${variant}(vector<vector<int>> graph) {
            vector<int> indegree(graph.size(), 0);
            for (const auto &neighbours : graph) {
                for (int node : neighbours) {
                    indegree[node]++;
                }
            }
            queue<int> q;
            for (int index = 0; index < static_cast<int>(graph.size()); ++index) {
                if (indegree[index] == 0) {
                    q.push(index);
                }
            }
            vector<int> order;
            while (!q.empty()) {
                int node = q.front();
                q.pop();
                order.push_back(node);
                for (int next : graph[node]) {
                    indegree[next]--;
                    if (indegree[next] == 0) {
                        q.push(next);
                    }
                }
            }
            return order;
        }

        int main() {
            vector<vector<int>> graph(${nodes});
${edges.map((line) => line).join("\n")}
            vector<int> order = topological_order_${variant}(graph);
            for (int node : order) {
                cout << node << ' ';
            }
            cout << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const values = buildNumericSequence(variant + 9, 12, 6)
    values.sort((a, b) => a - b)
    return {
      title: `C++ Segment Tree Trial ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        struct SegmentTree${variant} {
            int n;
            vector<int> tree;
            explicit SegmentTree${variant}(const vector<int> &values) {
                n = 1;
                while (n < static_cast<int>(values.size())) {
                    n <<= 1;
                }
                tree.assign(2 * n, 0);
                for (int index = 0; index < static_cast<int>(values.size()); ++index) {
                    tree[n + index] = values[index];
                }
                for (int index = n - 1; index > 0; --index) {
                    tree[index] = tree[index << 1] + tree[(index << 1) | 1];
                }
            }
            void update(int position, int value) {
                int node = position + n;
                tree[node] = value;
                while (node > 1) {
                    node >>= 1;
                    tree[node] = tree[node << 1] + tree[(node << 1) | 1];
                }
            }
            int range_query(int left, int right) const {
                int result = 0;
                for (left += n, right += n; left <= right; left >>= 1, right >>= 1) {
                    if (left & 1) {
                        result += tree[left++];
                    }
                    if (!(right & 1)) {
                        result += tree[right--];
                    }
                }
                return result;
            }
        };

        int main() {
            vector<int> values = { ${joinNumbers(values)} };
            SegmentTree${variant} tree(values);
            cout << tree.range_query(1, ${Math.min(5 + (variant % 4), values.length - 1)}) << "\n";
            tree.update(${variant % values.length}, ${values[(variant * 2) % values.length] + 3});
            cout << tree.range_query(0, ${Math.min(6 + (variant % 3), values.length - 1)}) << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const values = buildNumericSequence(variant + 21, 14, 5)
    values.sort((a, b) => a - b)
    return {
      title: `C++ Fenwick Tree Gauntlet ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        struct Fenwick${variant} {
            vector<int> bit;
            explicit Fenwick${variant}(int size) : bit(size + 1, 0) {}
            void add(int index, int value) {
                for (++index; index < static_cast<int>(bit.size()); index += index & -index) {
                    bit[index] += value;
                }
            }
            int sum(int index) const {
                int result = 0;
                for (++index; index > 0; index -= index & -index) {
                    result += bit[index];
                }
                return result;
            }
            int range_sum(int left, int right) const {
                return sum(right) - (left ? sum(left - 1) : 0);
            }
        };

        int main() {
            vector<int> values = { ${joinNumbers(values)} };
            Fenwick${variant} tree(values.size());
            for (int index = 0; index < static_cast<int>(values.size()); ++index) {
                tree.add(index, values[index]);
            }
            cout << tree.range_sum(0, ${Math.min(5 + (variant % 5), values.length - 1)}) << "\n";
            tree.add(${variant % values.length}, ${(variant % 7) + 5});
            cout << tree.range_sum(2, ${Math.min(7 + (variant % 4), values.length - 1)}) << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const size = 8 + (variant % 5)
    return {
      title: `C++ Disjoint Set Siege ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        struct DisjointSet${variant} {
            vector<int> parent;
            vector<int> rankValue;
            explicit DisjointSet${variant}(int size) : parent(size), rankValue(size, 0) {
                iota(parent.begin(), parent.end(), 0);
            }
            int find(int x) {
                if (parent[x] != x) {
                    parent[x] = find(parent[x]);
                }
                return parent[x];
            }
            void unite(int a, int b) {
                a = find(a);
                b = find(b);
                if (a == b) {
                    return;
                }
                if (rankValue[a] < rankValue[b]) {
                    swap(a, b);
                }
                parent[b] = a;
                if (rankValue[a] == rankValue[b]) {
                    rankValue[a]++;
                }
            }
        };

        int main() {
            DisjointSet${variant} dsu(${size});
            for (int index = 0; index < ${size - 1}; ++index) {
                dsu.unite(index, (index + ${(variant % 3) + 1}) % ${size});
            }
            unordered_map<int, vector<int>> groups;
            for (int index = 0; index < ${size}; ++index) {
                groups[dsu.find(index)].push_back(index);
            }
            for (const auto &entry : groups) {
                cout << "leader: " << entry.first << " -> ";
                for (int node : entry.second) {
                    cout << node << ' ';
                }
                cout << "\n";
            }
            return 0;
        }
`,
    }
  },
  (variant) => {
    const text = `battleplan${variant}`
    const pattern = `plan${variant % 4}`
    return {
      title: `C++ KMP Arena ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        vector<int> prefix_function_${variant}(const string &pattern) {
            vector<int> prefix(pattern.size(), 0);
            for (size_t index = 1; index < pattern.size(); ++index) {
                int j = prefix[index - 1];
                while (j > 0 && pattern[index] != pattern[j]) {
                    j = prefix[j - 1];
                }
                if (pattern[index] == pattern[j]) {
                    j++;
                }
                prefix[index] = j;
            }
            return prefix;
        }

        vector<int> kmp_search_${variant}(const string &text, const string &pattern) {
            vector<int> occurrences;
            vector<int> prefix = prefix_function_${variant}(pattern);
            int j = 0;
            for (int index = 0; index < static_cast<int>(text.size()); ++index) {
                while (j > 0 && text[index] != pattern[j]) {
                    j = prefix[j - 1];
                }
                if (text[index] == pattern[j]) {
                    j++;
                }
                if (j == static_cast<int>(pattern.size())) {
                    occurrences.push_back(index - j + 1);
                    j = prefix[j - 1];
                }
            }
            return occurrences;
        }

        int main() {
            string text = "${text.repeat(2)}";
            string pattern = "${pattern}";
            vector<int> occurrences = kmp_search_${variant}(text, pattern);
            for (int position : occurrences) {
                cout << position << ' ';
            }
            cout << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const capacity = 10 + (variant % 6)
    return {
      title: `C++ Knapsack Rally ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        int knapsack_${variant}(const vector<int> &weights, const vector<int> &values, int capacity) {
            vector<int> dp(capacity + 1, 0);
            for (size_t index = 0; index < weights.size(); ++index) {
                for (int cap = capacity; cap >= weights[index]; --cap) {
                    dp[cap] = max(dp[cap], dp[cap - weights[index]] + values[index]);
                }
            }
            return dp[capacity];
        }

        int main() {
            vector<int> weights = { ${joinNumbers(buildNumericSequence(variant + 4, 6, 3).map((value) => (value % 7) + 2))} };
            vector<int> values = { ${joinNumbers(buildNumericSequence(variant + 6, 6, 5).map((value) => (value % 13) + 4))} };
            cout << knapsack_${variant}(weights, values, ${capacity}) << "\n";
            return 0;
        }
`,
    }
  },
  (variant) => {
    const nodes = 4 + (variant % 4)
    return {
      title: `C++ Floyd Warshall Siege ${variant}`,
      code: `
        #include<bits/stdc++.h>
        using namespace std;

        const int INF = 1e8;

        vector<vector<int>> floyd_warshall_${variant}(vector<vector<int>> distance) {
            int n = static_cast<int>(distance.size());
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (distance[i][k] + distance[k][j] < distance[i][j]) {
                            distance[i][j] = distance[i][k] + distance[k][j];
                        }
                    }
                }
            }
            return distance;
        }

        int main() {
            vector<vector<int>> distance(${nodes}, vector<int>(${nodes}, INF));
            for (int index = 0; index < ${nodes}; ++index) {
                distance[index][index] = 0;
            }
            for (int index = 0; index < ${nodes - 1}; ++index) {
                distance[index][index + 1] = ${(variant % 5) + 3} + index;
                distance[index + 1][index] = ${(variant % 4) + 2} + index;
            }
            auto result = floyd_warshall_${variant}(distance);
            for (const auto &row : result) {
                for (int value : row) {
                    cout << (value >= INF ? -1 : value) << ' ';
                }
                cout << "\n";
            }
            return 0;
        }
`,
    }
  },
]

const pythonBasicTemplates: TemplateFactory[] = [
  (variant) => {
    const limit = 5 + (variant % 5)
    return {
      title: `Python counting drill ${variant}`,
      code: `
for value in range(1, ${limit}):
    print(value, end=" ")
print()
`,
    }
  },
  (variant) => {
    const numbers = buildNumericSequence(variant + 3, 5, 4)
    return {
      title: `Python list sum ${variant}`,
      code: `
values = [${joinNumbers(numbers)}]
total = sum(values)
print(total)
`,
    }
  },
  (variant) => {
    const word = `dojo${variant}`
    return {
      title: `Python reverse string ${variant}`,
      code: `
word = "${word}"
print(word[::-1])
`,
    }
  },
  (variant) => {
    const items = buildNumericSequence(variant + 4, 6, 3)
    return {
      title: `Python squares table ${variant}`,
      code: `
values = [${joinNumbers(items)}]
squares = [value * value for value in values]
print(squares)
`,
    }
  },
]

const pythonAdvancedTemplates: TemplateFactory[] = [
  (variant) => {
    const values = buildNumericSequence(variant + 3, 10, 4)
    values.sort((a, b) => a - b)
    const target = values[(variant * 5) % values.length]
    return {
      title: `Python Binary Search Quest ${variant}`,
      code: `
from bisect import bisect_left

def binary_search_${variant}(values: list[int], target: int) -> int:
    index = bisect_left(values, target)
    if index < len(values) and values[index] == target:
        return index
    return -1

if __name__ == "__main__":
    data = [${joinNumbers(values)}]
    target = ${target}
    print(binary_search_${variant}(data, target))
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 4)
    return {
      title: `Python BFS League ${variant}`,
      code: `
from collections import deque

def breadth_first_${variant}(graph: list[list[int]], start: int) -> list[int]:
    order: list[int] = []
    distance = [-1] * len(graph)
    queue: deque[int] = deque([start])
    distance[start] = 0
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph[node]:
            if distance[neighbour] == -1:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return order

if __name__ == "__main__":
    graph: list[list[int]] = [[] for _ in range(${nodes})]
    for index in range(${nodes}):
        graph[index].append((index + 1) % ${nodes})
        graph[index].append((index + ${(variant % 3) + 2}) % ${nodes})
    print(breadth_first_${variant}(graph, 0))
`,
    }
  },
  (variant) => {
    const size = 7 + (variant % 5)
    return {
      title: `Python DFS Challenge ${variant}`,
      code: `
def build_tree_${variant}(size: int) -> list[list[int]]:
    tree = [[] for _ in range(size)]
    for index in range(1, size):
        parent = (index - 1) // 2
        tree[parent].append(index)
    return tree

def depth_first_${variant}(tree: list[list[int]], start: int) -> list[int]:
    order: list[int] = []
    stack = [start]
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbour in reversed(tree[node]):
            stack.append(neighbour)
    return order

if __name__ == "__main__":
    tree = build_tree_${variant}(${size})
    print(depth_first_${variant}(tree, 0))
`,
    }
  },
  (variant) => {
    const nodes = 5 + (variant % 4)
    return {
      title: `Python Dijkstra Challenge ${variant}`,
      code: `
import heapq

def dijkstra_${variant}(graph: list[list[tuple[int, int]]], start: int) -> list[int]:
    distance = [10 ** 9] * len(graph)
    distance[start] = 0
    queue: list[tuple[int, int]] = [(0, start)]
    while queue:
        dist, node = heapq.heappop(queue)
        if dist != distance[node]:
            continue
        for neighbour, weight in graph[node]:
            next_dist = dist + weight
            if next_dist < distance[neighbour]:
                distance[neighbour] = next_dist
                heapq.heappush(queue, (next_dist, neighbour))
    return distance

if __name__ == "__main__":
    size = ${nodes}
    graph: list[list[tuple[int, int]]] = [[] for _ in range(size)]
    for index in range(size):
        graph[index].append(((index + 1) % size, (index * ${variant + 3}) % 9 + 4))
        graph[index].append(((index + ${(variant % 2) + 2}) % size, (index * ${variant + 5}) % 11 + 5))
    print(dijkstra_${variant}(graph, 0))
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 5)
    return {
      title: `Python Topological Cup ${variant}`,
      code: `
from collections import deque

def topological_${variant}(graph: list[list[int]]) -> list[int]:
    indegree = [0] * len(graph)
    for neighbours in graph:
        for node in neighbours:
            indegree[node] += 1
    queue: deque[int] = deque([index for index, value in enumerate(indegree) if value == 0])
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph[node]:
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                queue.append(neighbour)
    return order

if __name__ == "__main__":
    graph: list[list[int]] = [[] for _ in range(${nodes})]
    for index in range(${nodes - 1}):
        graph[index].append(index + 1)
        if index + 2 < ${nodes}:
            graph[index].append(index + 2)
    print(topological_${variant}(graph))
`,
    }
  },
  (variant) => {
    return {
      title: `Python Union Find Trial ${variant}`,
      code: `
class DisjointSet${variant}:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node: int) -> int:
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def unite(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

if __name__ == "__main__":
    size = ${8 + (variant % 5)}
    dsu = DisjointSet${variant}(size)
    for index in range(size - 1):
        dsu.unite(index, (index + ${(variant % 3) + 1}) % size)
    groups: dict[int, list[int]] = {}
    for node in range(size):
        leader = dsu.find(node)
        groups.setdefault(leader, []).append(node)
    for leader, members in groups.items():
        print(leader, members)
`,
    }
  },
  (variant) => {
    const numbers = buildNumericSequence(variant + 9, 12, 5)
    numbers.sort((a, b) => a - b)
    return {
      title: `Python Segment Tree Run ${variant}`,
      code: `
class SegmentTree${variant}:
    def __init__(self, values: list[int]) -> None:
        self.size = 1
        while self.size < len(values):
            self.size <<= 1
        self.tree = [0] * (2 * self.size)
        for index, value in enumerate(values):
            self.tree[self.size + index] = value
        for index in range(self.size - 1, 0, -1):
            self.tree[index] = self.tree[2 * index] + self.tree[2 * index + 1]

    def update(self, position: int, value: int) -> None:
        index = position + self.size
        self.tree[index] = value
        index //= 2
        while index >= 1:
            self.tree[index] = self.tree[2 * index] + self.tree[2 * index + 1]
            index //= 2

    def range_sum(self, left: int, right: int) -> int:
        result = 0
        left += self.size
        right += self.size
        while left <= right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 0:
                result += self.tree[right]
                right -= 1
            left //= 2
            right //= 2
        return result

if __name__ == "__main__":
    values = [${joinNumbers(numbers)}]
    tree = SegmentTree${variant}(values)
    print(tree.range_sum(1, ${Math.min(5 + (variant % 4), numbers.length - 1)}))
    tree.update(${variant % numbers.length}, ${numbers[(variant * 2) % numbers.length] + 3})
    print(tree.range_sum(0, ${Math.min(6 + (variant % 3), numbers.length - 1)}))
`,
    }
  },
  (variant) => {
    const patternValue = variant % 2 === 0 ? "ena" : "ar"
    return {
      title: `Python KMP Showdown ${variant}`,
      code: `
def prefix_function_${variant}(pattern: str) -> list[int]:
    prefix = [0] * len(pattern)
    j = 0
    for index in range(1, len(pattern)):
        while j > 0 and pattern[index] != pattern[j]:
            j = prefix[j - 1]
        if pattern[index] == pattern[j]:
            j += 1
        prefix[index] = j
    return prefix

def kmp_search_${variant}(text: str, pattern: str) -> list[int]:
    prefix = prefix_function_${variant}(pattern)
    result: list[int] = []
    j = 0
    for index, char in enumerate(text):
        while j > 0 and char != pattern[j]:
            j = prefix[j - 1]
        if char == pattern[j]:
            j += 1
        if j == len(pattern):
            result.append(index - j + 1)
            j = prefix[j - 1]
    return result

if __name__ == "__main__":
    text = "arena${variant}" * 2
    pattern = "${patternValue}"
    print(kmp_search_${variant}(text, pattern))
`,
    }
  },
  (variant) => {
    const capacity = 12 + (variant % 6)
    return {
      title: `Python Knapsack Arena ${variant}`,
      code: `
def knapsack_${variant}(weights: list[int], values: list[int], capacity: int) -> int:
    dp = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for remaining in range(capacity, weight - 1, -1):
            dp[remaining] = max(dp[remaining], dp[remaining - weight] + value)
    return dp[capacity]

if __name__ == "__main__":
    weights = [${joinNumbers(buildNumericSequence(variant + 5, 6, 4).map((value) => (value % 7) + 2))}]
    values = [${joinNumbers(buildNumericSequence(variant + 7, 6, 6).map((value) => (value % 11) + 5))}]
    print(knapsack_${variant}(weights, values, ${capacity}))
`,
    }
  },
  (variant) => {
    return {
      title: `Python Floyd Warshall Duel ${variant}`,
      code: `
INF = 10 ** 9

def floyd_warshall_${variant}(distance: list[list[int]]) -> list[list[int]]:
    size = len(distance)
    for middle in range(size):
        for left in range(size):
            for right in range(size):
                if distance[left][middle] + distance[middle][right] < distance[left][right]:
                    distance[left][right] = distance[left][middle] + distance[middle][right]
    return distance

if __name__ == "__main__":
    size = ${4 + (variant % 4)}
    distance = [[INF for _ in range(size)] for _ in range(size)]
    for index in range(size):
        distance[index][index] = 0
    for index in range(size - 1):
        distance[index][index + 1] = (index * ${variant + 2}) % 9 + 3
        distance[index + 1][index] = (index * ${variant + 4}) % 7 + 4
    result = floyd_warshall_${variant}(distance)
    for row in result:
        print([value if value < INF else -1 for value in row])
`,
    }
  },
]

const javascriptBasicTemplates: TemplateFactory[] = [
  (variant) => {
    const limit = 5 + (variant % 5)
    return {
      title: `JavaScript counting warmup ${variant}`,
      code: `
const values${variant} = []
for (let index = 1; index <= ${limit}; index += 1) {
    values${variant}.push(index)
}
console.log(values${variant}.join(" "))
`,
    }
  },
  (variant) => {
    const numbers = buildNumericSequence(variant + 2, 6, 4)
    return {
      title: `JavaScript array sum ${variant}`,
      code: `
const values${variant} = [${joinNumbers(numbers)}]
const total${variant} = values${variant}.reduce((sum, value) => sum + value, 0)
console.log(total${variant})
`,
    }
  },
  (variant) => {
    const word = `dojo${variant}`
    return {
      title: `JavaScript reverse string ${variant}`,
      code: `
const word${variant} = "${word}"
const reversed${variant} = word${variant}.split("").reverse().join("")
console.log(reversed${variant})
`,
    }
  },
  (variant) => {
    return {
      title: `JavaScript squares table ${variant}`,
      code: `
const values${variant} = Array.from({ length: 5 }, (_, index) => index + 1)
const squares${variant} = values${variant}.map((value) => value * value)
console.log(squares${variant})
`,
    }
  },
]

const BASIC_SEGMENT_COUNT = LEVEL_SEGMENTS[0]

const buildLanguageSnippets = (
  language: LanguageKey,
  basicTemplates: TemplateFactory[],
  advancedTemplates: TemplateFactory[],
): Snippet[] => {
  const basicSnippets = createSnippetSet(language, basicTemplates, BASIC_SEGMENT_COUNT, 0)
  const advancedCount = SNIPPET_COUNT_PER_LANGUAGE - BASIC_SEGMENT_COUNT
  const advancedSnippets = createSnippetSet(
    language,
    advancedTemplates,
    advancedCount,
    BASIC_SEGMENT_COUNT,
  )
  return [...basicSnippets, ...advancedSnippets]
}

const javascriptAdvancedTemplates: TemplateFactory[] = [
  (variant) => {
    const values = buildNumericSequence(variant + 2, 10, 3)
    values.sort((a, b) => a - b)
    const target = values[(variant * 4) % values.length]
    return {
      title: `JavaScript Binary Search Sprint ${variant}`,
      code: `
function binarySearch${variant}(array, target) {
    let left = 0
    let right = array.length - 1
    while (left <= right) {
        const mid = Math.floor((left + right) / 2)
        if (array[mid] === target) {
            return mid
        }
        if (array[mid] < target) {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1
}

const data${variant} = [${joinNumbers(values)}]
const target${variant} = ${target}
console.log(binarySearch${variant}(data${variant}, target${variant}))
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 4)
    return {
      title: `JavaScript BFS Relay ${variant}`,
      code: `
function breadthFirst${variant}(graph, start) {
    const order = []
    const distance = Array(graph.length).fill(-1)
    const queue = [start]
    distance[start] = 0
    while (queue.length) {
        const node = queue.shift()
        order.push(node)
        for (const neighbour of graph[node]) {
            if (distance[neighbour] === -1) {
                distance[neighbour] = distance[node] + 1
                queue.push(neighbour)
            }
        }
    }
    return order
}

const graph${variant} = Array.from({ length: ${nodes} }, () => [])
for (let index = 0; index < ${nodes}; index += 1) {
    graph${variant}[index].push((index + 1) % ${nodes})
    graph${variant}[index].push((index + ${(variant % 3) + 2}) % ${nodes})
}
console.log(breadthFirst${variant}(graph${variant}, 0))
`,
    }
  },
  (variant) => {
    const nodes = 5 + (variant % 4)
    return {
      title: `JavaScript DFS Raid ${variant}`,
      code: `
function buildTree${variant}(size) {
    const tree = Array.from({ length: size }, () => [])
    for (let index = 1; index < size; index += 1) {
        const parent = Math.floor((index - 1) / 2)
        tree[parent].push(index)
    }
    return tree
}

function depthFirst${variant}(tree, start) {
    const order = []
    const stack = [start]
    while (stack.length) {
        const node = stack.pop()
        order.push(node)
        const neighbours = tree[node]
        for (let index = neighbours.length - 1; index >= 0; index -= 1) {
            stack.push(neighbours[index])
        }
    }
    return order
}

const tree${variant} = buildTree${variant}(${nodes})
console.log(depthFirst${variant}(tree${variant}, 0))
`,
    }
  },
  (variant) => {
    const nodes = 5 + (variant % 3)
    return {
      title: `JavaScript Dijkstra Run ${variant}`,
      code: `
function dijkstra${variant}(graph, start) {
    const distance = Array(graph.length).fill(Number.POSITIVE_INFINITY)
    const visited = new Set()
    distance[start] = 0
    while (visited.size < graph.length) {
        let candidate = -1
        let best = Number.POSITIVE_INFINITY
        for (let index = 0; index < graph.length; index += 1) {
            if (!visited.has(index) && distance[index] < best) {
                best = distance[index]
                candidate = index
            }
        }
        if (candidate === -1) {
            break
        }
        visited.add(candidate)
        for (const [next, weight] of graph[candidate]) {
            const nextDistance = best + weight
            if (nextDistance < distance[next]) {
                distance[next] = nextDistance
            }
        }
    }
    return distance
}

const graph${variant} = Array.from({ length: ${nodes} }, () => [])
for (let index = 0; index < ${nodes}; index += 1) {
    graph${variant}[index].push([(index + 1) % ${nodes}, (index * ${(variant + 3)}) % 9 + 3])
    graph${variant}[index].push([(index + ${(variant % 2) + 2}) % ${nodes}, (index * ${(variant + 5)}) % 11 + 4])
}
console.log(dijkstra${variant}(graph${variant}, 0))
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 4)
    return {
      title: `JavaScript Topological Duel ${variant}`,
      code: `
function topological${variant}(graph) {
    const indegree = Array(graph.length).fill(0)
    for (const neighbours of graph) {
        for (const node of neighbours) {
            indegree[node] += 1
        }
    }
    const queue = []
    for (let index = 0; index < indegree.length; index += 1) {
        if (indegree[index] === 0) {
            queue.push(index)
        }
    }
    const order = []
    while (queue.length) {
        const node = queue.shift()
        order.push(node)
        for (const neighbour of graph[node]) {
            indegree[neighbour] -= 1
            if (indegree[neighbour] === 0) {
                queue.push(neighbour)
            }
        }
    }
    return order
}

const graph${variant} = Array.from({ length: ${nodes} }, () => [])
for (let index = 0; index < ${nodes} - 1; index += 1) {
    graph${variant}[index].push(index + 1)
    if (index + 2 < ${nodes}) {
        graph${variant}[index].push(index + 2)
    }
}
console.log(topological${variant}(graph${variant}))
`,
    }
  },
  (variant) => {
    const size = 8 + (variant % 5)
    return {
      title: `JavaScript Disjoint Set Dash ${variant}`,
      code: `
class DisjointSet${variant} {
    constructor(size) {
        this.parent = Array.from({ length: size }, (_, index) => index)
        this.rank = Array(size).fill(0)
    }
    find(node) {
        if (this.parent[node] !== node) {
            this.parent[node] = this.find(this.parent[node])
        }
        return this.parent[node]
    }
    unite(a, b) {
        let rootA = this.find(a)
        let rootB = this.find(b)
        if (rootA === rootB) {
            return
        }
        if (this.rank[rootA] < this.rank[rootB]) {
            ;[rootA, rootB] = [rootB, rootA]
        }
        this.parent[rootB] = rootA
        if (this.rank[rootA] === this.rank[rootB]) {
            this.rank[rootA] += 1
        }
    }
}

const dsu${variant} = new DisjointSet${variant}(${size})
for (let index = 0; index < ${size - 1}; index += 1) {
    dsu${variant}.unite(index, (index + ${(variant % 3) + 1}) % ${size})
}
const groups${variant} = new Map()
for (let index = 0; index < ${size}; index += 1) {
    const leader = dsu${variant}.find(index)
    if (!groups${variant}.has(leader)) {
        groups${variant}.set(leader, [])
    }
    groups${variant}.get(leader).push(index)
}
console.log([...groups${variant}.entries()])
`,
    }
  },
  (variant) => {
    const values = buildNumericSequence(variant + 11, 12, 4)
    values.sort((a, b) => a - b)
    return {
      title: `JavaScript Segment Tree Clash ${variant}`,
      code: `
class SegmentTree${variant} {
    constructor(values) {
        this.size = 1
        while (this.size < values.length) {
            this.size <<= 1
        }
        this.tree = Array(this.size * 2).fill(0)
        for (let index = 0; index < values.length; index += 1) {
            this.tree[this.size + index] = values[index]
        }
        for (let index = this.size - 1; index >= 1; index -= 1) {
            this.tree[index] = this.tree[index * 2] + this.tree[index * 2 + 1]
        }
    }
    update(position, value) {
        let index = position + this.size
        this.tree[index] = value
        index = Math.floor(index / 2)
        while (index >= 1) {
            this.tree[index] = this.tree[index * 2] + this.tree[index * 2 + 1]
            index = Math.floor(index / 2)
        }
    }
    rangeSum(left, right) {
        let result = 0
        left += this.size
        right += this.size
        while (left <= right) {
            if (left % 2 === 1) {
                result += this.tree[left]
                left += 1
            }
            if (right % 2 === 0) {
                result += this.tree[right]
                right -= 1
            }
            left = Math.floor(left / 2)
            right = Math.floor(right / 2)
        }
        return result
    }
}

const values${variant} = [${joinNumbers(values)}]
const tree${variant} = new SegmentTree${variant}(values${variant})
console.log(tree${variant}.rangeSum(1, ${Math.min(5 + (variant % 4), values.length - 1)}))
tree${variant}.update(${variant % values.length}, ${values[(variant * 3) % values.length] + 5})
console.log(tree${variant}.rangeSum(0, ${Math.min(6 + (variant % 3), values.length - 1)}))
`,
    }
  },
  (variant) => {
    const text = `champion${variant}`
    const pattern = `amp${variant % 5}`
    return {
      title: `JavaScript KMP Sprint ${variant}`,
      code: `
function prefix${variant}(pattern) {
    const prefix = Array(pattern.length).fill(0)
    let j = 0
    for (let index = 1; index < pattern.length; index += 1) {
        while (j > 0 && pattern[index] !== pattern[j]) {
            j = prefix[j - 1]
        }
        if (pattern[index] === pattern[j]) {
            j += 1
        }
        prefix[index] = j
    }
    return prefix
}

function kmp${variant}(text, pattern) {
    const result = []
    const prefix = prefix${variant}(pattern)
    let j = 0
    for (let index = 0; index < text.length; index += 1) {
        while (j > 0 && text[index] !== pattern[j]) {
            j = prefix[j - 1]
        }
        if (text[index] === pattern[j]) {
            j += 1
        }
        if (j === pattern.length) {
            result.push(index - j + 1)
            j = prefix[j - 1]
        }
    }
    return result
}

console.log(kmp${variant}("${text.repeat(2)}", "${pattern}"))
`,
    }
  },
  (variant) => {
    const capacity = 12 + (variant % 5)
    return {
      title: `JavaScript Knapsack Battle ${variant}`,
      code: `
function knapsack${variant}(weights, values, capacity) {
    const dp = Array(capacity + 1).fill(0)
    for (let index = 0; index < weights.length; index += 1) {
        for (let cap = capacity; cap >= weights[index]; cap -= 1) {
            dp[cap] = Math.max(dp[cap], dp[cap - weights[index]] + values[index])
        }
    }
    return dp[capacity]
}

const weights${variant} = [${joinNumbers(buildNumericSequence(variant + 4, 6, 3).map((value) => (value % 7) + 2))}]
const values${variant} = [${joinNumbers(buildNumericSequence(variant + 6, 6, 5).map((value) => (value % 11) + 4))}]
console.log(knapsack${variant}(weights${variant}, values${variant}, ${capacity}))
`,
    }
  },
  (variant) => {
    return {
      title: `JavaScript Floyd Warshall Clash ${variant}`,
      code: `
const INF = 1e9

function floydWarshall${variant}(matrix) {
    const size = matrix.length
    for (let middle = 0; middle < size; middle += 1) {
        for (let left = 0; left < size; left += 1) {
            for (let right = 0; right < size; right += 1) {
                const candidate = matrix[left][middle] + matrix[middle][right]
                if (candidate < matrix[left][right]) {
                    matrix[left][right] = candidate
                }
            }
        }
    }
    return matrix
}

const size${variant} = ${4 + (variant % 3)}
const matrix${variant} = Array.from({ length: size${variant} }, (_, row) =>
    Array.from({ length: size${variant} }, (_, col) => (row === col ? 0 : INF)),
)
for (let index = 0; index < size${variant} - 1; index += 1) {
    matrix${variant}[index][index + 1] = (index * ${(variant + 2)}) % 9 + 3
    matrix${variant}[index + 1][index] = (index * ${(variant + 4)}) % 7 + 4
}
console.log(floydWarshall${variant}(matrix${variant}))
`,
    }
  },
]

const javaBasicTemplates: TemplateFactory[] = [
  (variant) => {
    const limit = 5 + (variant % 5)
    return {
      title: `Java counting starter ${variant}`,
      code: `
import java.util.stream.IntStream;

public class CountingStarter${variant} {
    public static void main(String[] args) {
        IntStream.rangeClosed(1, ${limit}).forEach(value -> System.out.print(value + " "));
        System.out.println();
    }
}
`,
    }
  },
  (variant) => {
    const size = 5 + (variant % 4)
    return {
      title: `Java array sum ${variant}`,
      code: `
import java.util.Arrays;

public class ArraySum${variant} {
    public static void main(String[] args) {
        int[] values = new int[] { ${joinNumbers(buildNumericSequence(variant + 3, size, 4))} };
        int total = Arrays.stream(values).sum();
        System.out.println(total);
    }
}
`,
    }
  },
  (variant) => {
    const wordLiteral = `dojo${variant}`
    return {
      title: `Java reverse string ${variant}`,
      code: `
public class ReverseString${variant} {
    public static void main(String[] args) {
        String word = "${wordLiteral}";
        StringBuilder builder = new StringBuilder(word);
        System.out.println(builder.reverse().toString());
    }
}
`,
    }
  },
  (variant) => {
    return {
      title: `Java squares table ${variant}`,
      code: `
import java.util.List;

public class SquaresTable${variant} {
    public static void main(String[] args) {
        List<Integer> squares = java.util.stream.IntStream.rangeClosed(1, 5)
            .map(value -> value * value)
            .boxed()
            .toList();
        System.out.println(squares);
    }
}
`,
    }
  },
]

const javaAdvancedTemplates: TemplateFactory[] = [
  (variant) => {
    const values = buildNumericSequence(variant + 4, 10, 3)
    values.sort((a, b) => a - b)
    const target = values[(variant * 4) % values.length]
    return {
      title: `Java Binary Search Arena ${variant}`,
      code: `
import java.util.Arrays;

public class BinarySearchArena${variant} {
    public static int binarySearch(int[] data, int target) {
        int left = 0;
        int right = data.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (data[mid] == target) {
                return mid;
            }
            if (data[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        int[] data = new int[] { ${joinNumbers(values)} };
        System.out.println(binarySearch(data, ${target}));
    }
}
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 4)
    return {
      title: `Java BFS Circuit ${variant}`,
      code: `
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

public class BfsCircuit${variant} {
    public static List<Integer> breadthFirst(List<List<Integer>> graph, int start) {
        List<Integer> order = new ArrayList<>();
        int[] distance = new int[graph.size()];
        Arrays.fill(distance, -1);
        Deque<Integer> queue = new ArrayDeque<>();
        queue.addLast(start);
        distance[start] = 0;
        while (!queue.isEmpty()) {
            int node = queue.removeFirst();
            order.add(node);
            for (int neighbour : graph.get(node)) {
                if (distance[neighbour] == -1) {
                    distance[neighbour] = distance[node] + 1;
                    queue.addLast(neighbour);
                }
            }
        }
        return order;
    }

    public static void main(String[] args) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int index = 0; index < ${nodes}; index++) {
            graph.add(new ArrayList<>());
        }
        for (int index = 0; index < ${nodes}; index++) {
            graph.get(index).add((index + 1) % ${nodes});
            graph.get(index).add((index + ${(variant % 3) + 2}) % ${nodes});
        }
        System.out.println(breadthFirst(graph, 0));
    }
}
`,
    }
  },
  (variant) => {
    const nodes = 7 + (variant % 5)
    return {
      title: `Java DFS Arena ${variant}`,
      code: `
import java.util.ArrayList;
import java.util.List;

public class DfsArena${variant} {
    private static void dfs(int node, List<List<Integer>> graph, List<Integer> order) {
        order.add(node);
        for (int neighbour : graph.get(node)) {
            dfs(neighbour, graph, order);
        }
    }

    public static void main(String[] args) {
        List<List<Integer>> tree = new ArrayList<>();
        for (int index = 0; index < ${nodes}; index++) {
            tree.add(new ArrayList<>());
        }
        for (int index = 1; index < ${nodes}; index++) {
            int parent = (index - 1) / 2;
            tree.get(parent).add(index);
        }
        List<Integer> order = new ArrayList<>();
        dfs(0, tree, order);
        System.out.println(order);
    }
}
`,
    }
  },
  (variant) => {
    const nodes = 5 + (variant % 3)
    return {
      title: `Java Dijkstra Duel ${variant}`,
      code: `
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

public class DijkstraDuel${variant} {
    private static record Edge(int to, int weight) {}

    public static int[] dijkstra(List<List<Edge>> graph, int start) {
        int[] distance = new int[graph.size()];
        Arrays.fill(distance, Integer.MAX_VALUE / 2);
        distance[start] = 0;
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));
        queue.add(new int[] {0, start});
        while (!queue.isEmpty()) {
            int[] entry = queue.poll();
            int dist = entry[0];
            int node = entry[1];
            if (dist != distance[node]) {
                continue;
            }
            for (Edge edge : graph.get(node)) {
                int next = edge.to();
                int nextDist = dist + edge.weight();
                if (nextDist < distance[next]) {
                    distance[next] = nextDist;
                    queue.add(new int[] {nextDist, next});
                }
            }
        }
        return distance;
    }

    public static void main(String[] args) {
        List<List<Edge>> graph = new ArrayList<>();
        for (int index = 0; index < ${nodes}; index++) {
            graph.add(new ArrayList<>());
        }
        for (int index = 0; index < ${nodes}; index++) {
            graph.get(index).add(new Edge((index + 1) % ${nodes}, (index * ${(variant + 3)}) % 9 + 3));
            graph.get(index).add(new Edge((index + ${(variant % 2) + 2}) % ${nodes}, (index * ${(variant + 5)}) % 11 + 4));
        }
        System.out.println(Arrays.toString(dijkstra(graph, 0)));
    }
}
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 4)
    return {
      title: `Java Topological Clash ${variant}`,
      code: `
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

public class TopologicalClash${variant} {
    public static List<Integer> topological(List<List<Integer>> graph) {
        int[] indegree = new int[graph.size()];
        for (List<Integer> neighbours : graph) {
            for (int node : neighbours) {
                indegree[node]++;
            }
        }
        Deque<Integer> queue = new ArrayDeque<>();
        for (int index = 0; index < indegree.length; index++) {
            if (indegree[index] == 0) {
                queue.addLast(index);
            }
        }
        List<Integer> order = new ArrayList<>();
        while (!queue.isEmpty()) {
            int node = queue.removeFirst();
            order.add(node);
            for (int neighbour : graph.get(node)) {
                indegree[neighbour]--;
                if (indegree[neighbour] == 0) {
                    queue.addLast(neighbour);
                }
            }
        }
        return order;
    }

    public static void main(String[] args) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int index = 0; index < ${nodes}; index++) {
            graph.add(new ArrayList<>());
        }
        for (int index = 0; index < ${nodes} - 1; index++) {
            graph.get(index).add(index + 1);
            if (index + 2 < ${nodes}) {
                graph.get(index).add(index + 2);
            }
        }
        System.out.println(topological(graph));
    }
}
`,
    }
  },
  (variant) => {
    const size = 8 + (variant % 5)
    return {
      title: `Java Union Find Rally ${variant}`,
      code: `
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class UnionFindRally${variant} {
    private static class DisjointSet {
        private final int[] parent;
        private final int[] rank;

        DisjointSet(int size) {
            parent = new int[size];
            rank = new int[size];
            for (int index = 0; index < size; index++) {
                parent[index] = index;
            }
        }

        int find(int node) {
            if (parent[node] != node) {
                parent[node] = find(parent[node]);
            }
            return parent[node];
        }

        void unite(int a, int b) {
            int rootA = find(a);
            int rootB = find(b);
            if (rootA == rootB) {
                return;
            }
            if (rank[rootA] < rank[rootB]) {
                int temp = rootA;
                rootA = rootB;
                rootB = temp;
            }
            parent[rootB] = rootA;
            if (rank[rootA] == rank[rootB]) {
                rank[rootA]++;
            }
        }
    }

    public static void main(String[] args) {
        DisjointSet dsu = new DisjointSet(${size});
        for (int index = 0; index < ${size - 1}; index++) {
            dsu.unite(index, (index + ${(variant % 3) + 1}) % ${size});
        }
        Map<Integer, List<Integer>> groups = new HashMap<>();
        for (int index = 0; index < ${size}; index++) {
            int leader = dsu.find(index);
            groups.computeIfAbsent(leader, ignored -> new ArrayList<>()).add(index);
        }
        System.out.println(groups);
    }
}
`,
    }
  },
  (variant) => {
    const values = buildNumericSequence(variant + 10, 12, 4)
    values.sort((a, b) => a - b)
    return {
      title: `Java Segment Tree Clash ${variant}`,
      code: `
import java.util.Arrays;

public class SegmentTreeClash${variant} {
    private final int size;
    private final int[] tree;

    public SegmentTreeClash${variant}(int[] values) {
        int n = 1;
        while (n < values.length) {
            n <<= 1;
        }
        size = n;
        tree = new int[2 * size];
        Arrays.fill(tree, 0);
        System.arraycopy(values, 0, tree, size, values.length);
        for (int index = size - 1; index >= 1; index--) {
            tree[index] = tree[index << 1] + tree[(index << 1) | 1];
        }
    }

    public void update(int position, int value) {
        int node = position + size;
        tree[node] = value;
        node >>= 1;
        while (node >= 1) {
            tree[node] = tree[node << 1] + tree[(node << 1) | 1];
            node >>= 1;
        }
    }

    public int rangeQuery(int left, int right) {
        int result = 0;
        int l = left + size;
        int r = right + size;
        while (l <= r) {
            if ((l & 1) == 1) {
                result += tree[l++];
            }
            if ((r & 1) == 0) {
                result += tree[r--];
            }
            l >>= 1;
            r >>= 1;
        }
        return result;
    }

    public static void main(String[] args) {
        int[] values = new int[] { ${joinNumbers(values)} };
        SegmentTreeClash${variant} tree = new SegmentTreeClash${variant}(values);
        System.out.println(tree.rangeQuery(1, ${Math.min(5 + (variant % 4), values.length - 1)}));
        tree.update(${variant % values.length}, ${values[(variant * 2) % values.length] + 4});
        System.out.println(tree.rangeQuery(0, ${Math.min(6 + (variant % 3), values.length - 1)}));
    }
}
`,
    }
  },
  (variant) => {
    const text = `tournament${variant}`
    const pattern = `ment${variant % 5}`
    return {
      title: `Java KMP Encounter ${variant}`,
      code: `
import java.util.ArrayList;
import java.util.List;

public class KmpEncounter${variant} {
    private static int[] prefixFunction(String pattern) {
        int[] prefix = new int[pattern.length()];
        int j = 0;
        for (int index = 1; index < pattern.length(); index++) {
            while (j > 0 && pattern.charAt(index) != pattern.charAt(j)) {
                j = prefix[j - 1];
            }
            if (pattern.charAt(index) == pattern.charAt(j)) {
                j++;
            }
            prefix[index] = j;
        }
        return prefix;
    }

    private static List<Integer> kmp(String text, String pattern) {
        int[] prefix = prefixFunction(pattern);
        List<Integer> result = new ArrayList<>();
        int j = 0;
        for (int index = 0; index < text.length(); index++) {
            while (j > 0 && text.charAt(index) != pattern.charAt(j)) {
                j = prefix[j - 1];
            }
            if (text.charAt(index) == pattern.charAt(j)) {
                j++;
            }
            if (j == pattern.length()) {
                result.add(index - j + 1);
                j = prefix[j - 1];
            }
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println(kmp("${text.repeat(2)}", "${pattern}"));
    }
}
`,
    }
  },
  (variant) => {
    const capacity = 12 + (variant % 5)
    return {
      title: `Java Knapsack Siege ${variant}`,
      code: `
import java.util.Arrays;

public class KnapsackSiege${variant} {
    public static int knapsack(int[] weights, int[] values, int capacity) {
        int[] dp = new int[capacity + 1];
        for (int index = 0; index < weights.length; index++) {
            int weight = weights[index];
            int value = values[index];
            for (int remaining = capacity; remaining >= weight; remaining--) {
                dp[remaining] = Math.max(dp[remaining], dp[remaining - weight] + value);
            }
        }
        return dp[capacity];
    }

    public static void main(String[] args) {
        int[] weights = new int[] { ${joinNumbers(buildNumericSequence(variant + 5, 6, 3).map((value) => (value % 7) + 2))} };
        int[] values = new int[] { ${joinNumbers(buildNumericSequence(variant + 7, 6, 5).map((value) => (value % 11) + 4))} };
        System.out.println(knapsack(weights, values, ${capacity}));
    }
}
`,
    }
  },
  (variant) => {
    const nodes = 4 + (variant % 3)
    return {
      title: `Java Floyd Warshall Masters ${variant}`,
      code: `
import java.util.Arrays;

public class FloydWarshallMasters${variant} {
    private static final int INF = 1_000_000_000;

    public static int[][] floydWarshall(int[][] distance) {
        int size = distance.length;
        for (int middle = 0; middle < size; middle++) {
            for (int left = 0; left < size; left++) {
                for (int right = 0; right < size; right++) {
                    int candidate = distance[left][middle] + distance[middle][right];
                    if (candidate < distance[left][right]) {
                        distance[left][right] = candidate;
                    }
                }
            }
        }
        return distance;
    }

    public static void main(String[] args) {
        int size = ${nodes};
        int[][] distance = new int[size][size];
        for (int row = 0; row < size; row++) {
            Arrays.fill(distance[row], INF);
            distance[row][row] = 0;
        }
        for (int index = 0; index < size - 1; index++) {
            distance[index][index + 1] = (index * ${(variant + 2)}) % 9 + 3;
            distance[index + 1][index] = (index * ${(variant + 4)}) % 7 + 4;
        }
        int[][] result = floydWarshall(distance);
        for (int[] row : result) {
            System.out.println(Arrays.toString(row));
        }
    }
}
`,
    }
  },
]

const cBasicTemplates: TemplateFactory[] = [
  (variant) => {
    const limit = 5 + (variant % 5)
    return {
      title: `C counting warmup ${variant}`,
      code: `
#include <stdio.h>

int main(void) {
    for (int value = 1; value <= ${limit}; ++value) {
        printf("%d ", value);
    }
    printf("\n");
    return 0;
}
`,
    }
  },
  (variant) => {
    const size = 4 + (variant % 4)
    return {
      title: `C array sum ${variant}`,
      code: `
#include <stdio.h>

int main(void) {
    int values[] = { ${joinNumbers(buildNumericSequence(variant + 4, size, 3))} };
    int length = sizeof(values) / sizeof(values[0]);
    int total = 0;
    for (int index = 0; index < length; ++index) {
        total += values[index];
    }
    printf("%d\n", total);
    return 0;
}
`,
    }
  },
  (variant) => {
    const number = 3 + (variant % 5)
    return {
      title: `C factorial basics ${variant}`,
      code: `
#include <stdio.h>

long long factorial_${variant}(int value) {
    long long result = 1;
    for (int index = 2; index <= value; ++index) {
        result *= index;
    }
    return result;
}

int main(void) {
    printf("%lld\n", factorial_${variant}(${number}));
    return 0;
}
`,
    }
  },
  (variant) => {
    const text = `dojo${variant}`
    return {
      title: `C reverse string ${variant}`,
      code: `
#include <stdio.h>
#include <string.h>

int main(void) {
    char word[] = "${text}";
    int left = 0;
    int right = (int)strlen(word) - 1;
    while (left < right) {
        char temp = word[left];
        word[left] = word[right];
        word[right] = temp;
        left++;
        right--;
    }
    printf("%s\n", word);
    return 0;
}
`,
    }
  },
]

const cAdvancedTemplates: TemplateFactory[] = [
  (variant) => {
    const values = buildNumericSequence(variant + 3, 10, 4)
    values.sort((a, b) => a - b)
    const target = values[(variant * 5) % values.length]
    return {
      title: `C Binary Search Arena ${variant}`,
      code: `
#include <stdio.h>

int binary_search_${variant}(int data[], int size, int target) {
    int left = 0;
    int right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (data[mid] == target) {
            return mid;
        }
        if (data[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

int main(void) {
    int data[] = { ${joinNumbers(values)} };
    int size = sizeof(data) / sizeof(data[0]);
    printf("%d\n", binary_search_${variant}(data, size, ${target}));
    return 0;
}
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 4)
    return {
      title: `C BFS Arena ${variant}`,
      code: `
#include <stdio.h>

void breadth_first_${variant}(int graph[][${nodes}], int size, int start) {
    int queue[${nodes} * ${nodes}] = {0};
    int front = 0;
    int back = 0;
    int distance[${nodes}] = {0};
    for (int index = 0; index < size; ++index) {
        distance[index] = -1;
    }
    queue[back++] = start;
    distance[start] = 0;
    while (front < back) {
        int node = queue[front++];
        printf("%d ", node);
        for (int neighbour = 0; neighbour < size; ++neighbour) {
            if (graph[node][neighbour] && distance[neighbour] == -1) {
                distance[neighbour] = distance[node] + 1;
                queue[back++] = neighbour;
            }
        }
    }
    printf("\n");
}

int main(void) {
    int graph[${nodes}][${nodes}] = {0};
    for (int index = 0; index < ${nodes}; ++index) {
        graph[index][(index + 1) % ${nodes}] = 1;
        graph[index][(index + ${(variant % 3) + 2}) % ${nodes}] = 1;
    }
    breadth_first_${variant}(graph, ${nodes}, 0);
    return 0;
}
`,
    }
  },
  (variant) => {
    const size = 7 + (variant % 5)
    return {
      title: `C DFS Circuit ${variant}`,
      code: `
#include <stdio.h>

void depth_first_${variant}(int node, int tree[][${size}], int size) {
    printf("%d ", node);
    for (int next = 0; next < size; ++next) {
        if (tree[node][next]) {
            depth_first_${variant}(next, tree, size);
        }
    }
}

int main(void) {
    int tree[${size}][${size}] = {0};
    for (int index = 1; index < ${size}; ++index) {
        int parent = (index - 1) / 2;
        tree[parent][index] = 1;
    }
    depth_first_${variant}(0, tree, ${size});
    printf("\n");
    return 0;
}
`,
    }
  },
  (variant) => {
    const nodes = 5 + (variant % 3)
    return {
      title: `C Dijkstra League ${variant}`,
      code: `
#include <stdio.h>

#define INF 1000000000

void dijkstra_${variant}(int graph[][${nodes}], int size, int start, int distance[]) {
    int visited[${nodes}] = {0};
    for (int index = 0; index < size; ++index) {
        distance[index] = INF;
    }
    distance[start] = 0;
    for (int step = 0; step < size; ++step) {
        int best = INF;
        int candidate = -1;
        for (int node = 0; node < size; ++node) {
            if (!visited[node] && distance[node] < best) {
                best = distance[node];
                candidate = node;
            }
        }
        if (candidate == -1) {
            break;
        }
        visited[candidate] = 1;
        for (int neighbour = 0; neighbour < size; ++neighbour) {
            if (graph[candidate][neighbour]) {
                int next = distance[candidate] + graph[candidate][neighbour];
                if (next < distance[neighbour]) {
                    distance[neighbour] = next;
                }
            }
        }
    }
}

int main(void) {
    int graph[${nodes}][${nodes}] = {0};
    for (int index = 0; index < ${nodes}; ++index) {
        graph[index][(index + 1) % ${nodes}] = (index * ${(variant + 3)}) % 9 + 3;
        graph[index][(index + ${(variant % 2) + 2}) % ${nodes}] = (index * ${(variant + 5)}) % 11 + 4;
    }
    int distance[${nodes}] = {0};
    dijkstra_${variant}(graph, ${nodes}, 0, distance);
    for (int index = 0; index < ${nodes}; ++index) {
        printf("%d ", distance[index]);
    }
    printf("\n");
    return 0;
}
`,
    }
  },
  (variant) => {
    const nodes = 6 + (variant % 4)
    return {
      title: `C Topological Fight ${variant}`,
      code: `
#include <stdio.h>

void topological_${variant}(int graph[][${nodes}], int size) {
    int indegree[${nodes}] = {0};
    int queue[${nodes}] = {0};
    int front = 0;
    int back = 0;
    for (int node = 0; node < size; ++node) {
        for (int neighbour = 0; neighbour < size; ++neighbour) {
            if (graph[node][neighbour]) {
                indegree[neighbour]++;
            }
        }
    }
    for (int node = 0; node < size; ++node) {
        if (indegree[node] == 0) {
            queue[back++] = node;
        }
    }
    while (front < back) {
        int node = queue[front++];
        printf("%d ", node);
        for (int neighbour = 0; neighbour < size; ++neighbour) {
            if (graph[node][neighbour]) {
                indegree[neighbour]--;
                if (indegree[neighbour] == 0) {
                    queue[back++] = neighbour;
                }
            }
        }
    }
    printf("\n");
}

int main(void) {
    int graph[${nodes}][${nodes}] = {0};
    for (int index = 0; index < ${nodes} - 1; ++index) {
        graph[index][index + 1] = 1;
        if (index + 2 < ${nodes}) {
            graph[index][index + 2] = 1;
        }
    }
    topological_${variant}(graph, ${nodes});
    return 0;
}
`,
    }
  },
  (variant) => {
    const size = 8 + (variant % 5)
    return {
      title: `C Union Find Duel ${variant}`,
      code: `
#include <stdio.h>

int find_${variant}(int parent[], int node) {
    if (parent[node] != node) {
        parent[node] = find_${variant}(parent, parent[node]);
    }
    return parent[node];
}

void unite_${variant}(int parent[], int rank[], int a, int b) {
    int rootA = find_${variant}(parent, a);
    int rootB = find_${variant}(parent, b);
    if (rootA == rootB) {
        return;
    }
    if (rank[rootA] < rank[rootB]) {
        int temp = rootA;
        rootA = rootB;
        rootB = temp;
    }
    parent[rootB] = rootA;
    if (rank[rootA] == rank[rootB]) {
        rank[rootA]++;
    }
}

int main(void) {
    int parent[${size}];
    int rank[${size}] = {0};
    for (int index = 0; index < ${size}; ++index) {
        parent[index] = index;
    }
    for (int index = 0; index < ${size} - 1; ++index) {
        unite_${variant}(parent, rank, index, (index + ${(variant % 3) + 1}) % ${size});
    }
    for (int index = 0; index < ${size}; ++index) {
        printf("%d:%d ", index, find_${variant}(parent, index));
    }
    printf("\n");
    return 0;
}
`,
    }
  },
  (variant) => {
    const values = buildNumericSequence(variant + 9, 12, 4)
    values.sort((a, b) => a - b)
    return {
      title: `C Fenwick Battle ${variant}`,
      code: `
#include <stdio.h>

void fenwick_add_${variant}(int tree[], int size, int index, int value) {
    index += 1;
    while (index <= size) {
        tree[index] += value;
        index += index & -index;
    }
}

int fenwick_sum_${variant}(int tree[], int index) {
    index += 1;
    int result = 0;
    while (index > 0) {
        result += tree[index];
        index -= index & -index;
    }
    return result;
}

int main(void) {
    int values[] = { ${joinNumbers(values)} };
    int size = sizeof(values) / sizeof(values[0]);
    int tree[${values.length + 2}] = {0};
    for (int index = 0; index < size; ++index) {
        fenwick_add_${variant}(tree, size, index, values[index]);
    }
    printf("%d\n", fenwick_sum_${variant}(tree, ${Math.min(5 + (variant % 4), values.length - 1)}));
    fenwick_add_${variant}(tree, size, ${variant % values.length}, ${(variant % 7) + 5});
    printf("%d\n", fenwick_sum_${variant}(tree, ${Math.min(7 + (variant % 3), values.length - 1)}));
    return 0;
}
`,
    }
  },
  (variant) => {
    const text = `kmparena${variant}`
    const pattern = `arena${variant % 5}`
    return {
      title: `C KMP League ${variant}`,
      code: `
#include <stdio.h>
#include <string.h>

void prefix_function_${variant}(const char *pattern, int prefix[]) {
    int length = (int)strlen(pattern);
    prefix[0] = 0;
    int j = 0;
    for (int index = 1; index < length; ++index) {
        while (j > 0 && pattern[index] != pattern[j]) {
            j = prefix[j - 1];
        }
        if (pattern[index] == pattern[j]) {
            j++;
        }
        prefix[index] = j;
    }
}

void kmp_search_${variant}(const char *text, const char *pattern) {
    int n = (int)strlen(pattern);
    int prefix[256] = {0};
    prefix_function_${variant}(pattern, prefix);
    int j = 0;
    for (int index = 0; text[index] != '\0'; ++index) {
        while (j > 0 && text[index] != pattern[j]) {
            j = prefix[j - 1];
        }
        if (text[index] == pattern[j]) {
            j++;
        }
        if (j == n) {
            printf("%d ", index - j + 1);
            j = prefix[j - 1];
        }
    }
    printf("\n");
}

int main(void) {
    const char *text = "${text.repeat(2)}";
    const char *pattern = "${pattern}";
    kmp_search_${variant}(text, pattern);
    return 0;
}
`,
    }
  },
  (variant) => {
    const capacity = 12 + (variant % 5)
    return {
      title: `C Knapsack Run ${variant}`,
      code: `
#include <stdio.h>

int knapsack_${variant}(int weights[], int values[], int items, int capacity) {
    int dp[128] = {0};
    for (int index = 0; index < items; ++index) {
        for (int remaining = capacity; remaining >= weights[index]; --remaining) {
            int candidate = dp[remaining - weights[index]] + values[index];
            if (candidate > dp[remaining]) {
                dp[remaining] = candidate;
            }
        }
    }
    return dp[capacity];
}

int main(void) {
    int weights[] = { ${joinNumbers(buildNumericSequence(variant + 5, 6, 3).map((value) => (value % 7) + 2))} };
    int values[] = { ${joinNumbers(buildNumericSequence(variant + 7, 6, 5).map((value) => (value % 11) + 4))} };
    int items = sizeof(weights) / sizeof(weights[0]);
    printf("%d\n", knapsack_${variant}(weights, values, items, ${capacity}));
    return 0;
}
`,
    }
  },
  (variant) => {
    const nodes = 4 + (variant % 3)
    return {
      title: `C Floyd Warshall Duel ${variant}`,
      code: `
#include <stdio.h>

#define INF 1000000000

void floyd_warshall_${variant}(int distance[][${nodes}], int size) {
    for (int middle = 0; middle < size; ++middle) {
        for (int left = 0; left < size; ++left) {
            for (int right = 0; right < size; ++right) {
                int candidate = distance[left][middle] + distance[middle][right];
                if (candidate < distance[left][right]) {
                    distance[left][right] = candidate;
                }
            }
        }
    }
}

int main(void) {
    int distance[${nodes}][${nodes}] = {0};
    for (int row = 0; row < ${nodes}; ++row) {
        for (int col = 0; col < ${nodes}; ++col) {
            distance[row][col] = (row == col) ? 0 : INF;
        }
    }
    for (int index = 0; index < ${nodes} - 1; ++index) {
        distance[index][index + 1] = (index * ${(variant + 2)}) % 9 + 3;
        distance[index + 1][index] = (index * ${(variant + 4)}) % 7 + 4;
    }
    floyd_warshall_${variant}(distance, ${nodes});
    for (int row = 0; row < ${nodes}; ++row) {
        for (int col = 0; col < ${nodes}; ++col) {
            int value = distance[row][col];
            printf("%d ", value >= INF ? -1 : value);
        }
        printf("\n");
    }
    return 0;
}
`,
    }
  },
]

export const SNIPPETS: Record<LanguageKey, Snippet[]> = {
  javascript: buildLanguageSnippets("javascript", javascriptBasicTemplates, javascriptAdvancedTemplates),
  python: buildLanguageSnippets("python", pythonBasicTemplates, pythonAdvancedTemplates),
  java: buildLanguageSnippets("java", javaBasicTemplates, javaAdvancedTemplates),
  cpp: buildLanguageSnippets("cpp", cppBasicTemplates, cppAdvancedTemplates),
  c: buildLanguageSnippets("c", cBasicTemplates, cAdvancedTemplates),
}

export const getRandomSnippet = (language: LanguageKey, excludeId?: string): Snippet => {
  const pool = SNIPPETS[language]
  if (!pool.length) {
    throw new Error(`No snippets configured for language: ${language}`)
  }
  if (pool.length === 1) {
    return pool[0]
  }
  let candidate = pool[Math.floor(Math.random() * pool.length)]
  if (excludeId) {
    for (let attempts = 0; attempts < 5 && candidate.id === excludeId; attempts += 1) {
      candidate = pool[Math.floor(Math.random() * pool.length)]
    }
  }
  return candidate
}
