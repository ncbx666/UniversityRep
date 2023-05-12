#include<iostream>
#include<vector>
#include<cstring>
using namespace std;

const int MAX_VERTICES = 20001; // ������������ ����� ������

vector<int> graph[MAX_VERTICES]; // ������ ���������
int n, m, s, t;                  // n - ����� ������, m - ����� �����, s - �����, t - ����
int parent[MAX_VERTICES];        // ��� �������� ������� � DFS ������

// �������� ����� ����� ��������� from � to
void add_edge(int from, int to) {
    graph[from].push_back(to);
    graph[to].push_back(from);
}

// ����� ������������� ���� � �����
bool dfs(int v) {
    if (v == t) return true; // ���� �������� �����, �� ���������� true
    for (int i = 0; i < int(graph[v].size()); ++i) {
        int to = graph[v][i]; //����� ������� �������
        if (parent[to] == -1) { // ���� ������� to ��� �� ��������
            parent[to] = v;        // �������� v ��� ������
            if (dfs(to)) return true; // ���� ������� ����� ������������� ����, �� ���������� true
        }
    }
    return false; // ������������� ���� �� ������
}

// ����� ������������ ����� � �����
int max_flow() {
    int result = 0;
    while (true) {
        memset(parent, -1, sizeof(parent)); // �������� ������ �������
        parent[s] = -2;                      // �������� ��������� �������
        if (!dfs(s)) break;                  // ���� ������������� ���� �� ������, �� ������� �� �����
        for (int v = t; v != s; v = parent[v]) { // �������� �� �������������� ���� � �������� ������
            int u = parent[v];
            for (int i = 0; i < graph[u].size(); ++i) {
                if (graph[u][i] == v) {
                    graph[u][i] = MAX_VERTICES; // �������� ����� ��� ���������
                    break;
                }
            }
            graph[v].push_back(u); // ��������� ����� �� v � u � ���������� ������������ 1
        }
        result++; // ����������� ������������ �����
    }
    return result;
}

int main() {
    n = 6;
    m = 7;
    s = 0;
    t = 5;
    add_edge(0, 1);
    add_edge(0, 2);
    add_edge(1, 3);
    add_edge(1, 4);
    add_edge(2, 3);
    add_edge(3, 5);
    add_edge(4, 5);
    cout << "max flow = "<< max_flow(); // ����� ������������� ������
    return 0;
}











//
//using namespace std;
//
//const int MAX_VERTICES = 10001; // ������������ ����� ������
//
//int capacity[MAX_VERTICES][MAX_VERTICES]; // ���������� ����������� �����
//vector<int> graph[MAX_VERTICES];          // ������ ���������
//int n=6, m=7, s=0, t=5;  // n - ����� ������, m - ����� �����, s - �����, t - ����
//
//int parent[MAX_VERTICES];                 // ��� �������� ������� � DFS ������
//int bottleneck[MAX_VERTICES];             // ��� �������� ����������� �������� �� ����
//
//// �������� ����� ����� ��������� from � to � ���������� ������������ cap
//void add_edge(int from, int to) {
//    graph[from].push_back(to);
//    graph[to].push_back(from);
//    capacity[from][to] = 1;
//    capacity[to][from] = 0; // ��������� �������� �����
//}
//
//// ����� ������������� ���� � ����� � ���������� �������� �� ���� ����
//bool dfs(int v) {
//    if (v == t) return true; // ���� �������� �����, �� ���������� true
//    for (int i = 0; i < int(graph[v].size()); i++) {
//        int to = graph[v][i];
//        if (capacity[v][to] > 0 && parent[to] == -1) { // ���� ������� to ��� �� �������� � ���� ������������ �����
//            parent[to] = v;        // �������� v ��� ������
//            bottleneck[to] = min(bottleneck[v], capacity[v][to]); // ��������� ���������� ��������
//            if (dfs(to)) { // ���� ������� ����� ������������� ����, �� ��������� ���������� ����������� ����� �� ���� ����
//                capacity[v][to] -= bottleneck[to];
//                capacity[to][v] += bottleneck[to];
//                return true;
//            }
//        }
//    }
//    return false; // ������������� ���� �� ������
//}
//
//// ����� ������������ ����� � �����
//int max_flow() {
//    int result = 0;
//    while (true) {
//        memset(parent, -1, sizeof(parent)); // �������� ������ �������
//        memset(bottleneck, 0, sizeof(bottleneck)); // �������� ������ ���������� ��������
//        bottleneck[s] = INT_MAX; // ���������� �������� ��� ������ ����� �������������
//        parent[s] = -2;           // �������� ��������� �������
//        if (!dfs(s)) break;       // ���� ������������� ���� �� ������, �� ������� �� �����
//        result += bottleneck[t]; // ����������� ������������ ����� �� �������� ����������� �������� ������������� ������
//    }
//    return result;
//}
//
//int main() {
//
//    for (int i = 0; i < m; i++) {
//        int from, to;
//        cin >> from >> to;
//        add_edge(from, to); // ��������� ����� ����� 1
//    }
//    cout << max_flow(); // ����� ������������� ������
//    return 0;
//}
//
