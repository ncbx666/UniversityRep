#include<iostream>
#include<vector>
#include<cstring>
using namespace std;

const int MAX_VERTICES = 20001; // максимальное число вершин

vector<int> graph[MAX_VERTICES]; // список смежности
int n, m, s, t;                  // n - число вершин, m - число ребер, s - исток, t - сток
int parent[MAX_VERTICES];        // для хранения предков в DFS обходе

// добавить ребро между вершинами from и to
void add_edge(int from, int to) {
    graph[from].push_back(to);
    graph[to].push_back(from);
}

// найти увеличивающий путь в графе
bool dfs(int v) {
    if (v == t) return true; // если достигли стока, то возвращаем true
    for (int i = 0; i < int(graph[v].size()); ++i) {
        int to = graph[v][i]; //берем текущую вершину
        if (parent[to] == -1) { // если вершина to еще не посещена
            parent[to] = v;        // помечаем v как предка
            if (dfs(to)) return true; // если удалось найти увеличивающий путь, то возвращаем true
        }
    }
    return false; // увеличивающий путь не найден
}

// найти максимальный поток в графе
int max_flow() {
    int result = 0;
    while (true) {
        memset(parent, -1, sizeof(parent)); // обнулить массив предков
        parent[s] = -2;                      // пометить начальную вершину
        if (!dfs(s)) break;                  // если увеличивающий путь не найден, то выходим из цикла
        for (int v = t; v != s; v = parent[v]) { // проходим по увеличивающему пути и изменяем потоки
            int u = parent[v];
            for (int i = 0; i < graph[u].size(); ++i) {
                if (graph[u][i] == v) {
                    graph[u][i] = MAX_VERTICES; // помечаем ребро как удаленное
                    break;
                }
            }
            graph[v].push_back(u); // добавляем ребро от v к u с пропускной способностью 1
        }
        result++; // увеличиваем максимальный поток
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
    cout << "max flow = "<< max_flow(); // вывод максимального потока
    return 0;
}











//
//using namespace std;
//
//const int MAX_VERTICES = 10001; // максимальное число вершин
//
//int capacity[MAX_VERTICES][MAX_VERTICES]; // пропускные способности ребер
//vector<int> graph[MAX_VERTICES];          // список смежности
//int n=6, m=7, s=0, t=5;  // n - число вершин, m - число ребер, s - исток, t - сток
//
//int parent[MAX_VERTICES];                 // для хранения предков в DFS обходе
//int bottleneck[MAX_VERTICES];             // для хранения бутылочного горлышка на пути
//
//// добавить ребро между вершинами from и to с пропускной способностью cap
//void add_edge(int from, int to) {
//    graph[from].push_back(to);
//    graph[to].push_back(from);
//    capacity[from][to] = 1;
//    capacity[to][from] = 0; // добавляем обратное ребро
//}
//
//// найти увеличивающий путь в графе и бутылочное горлышко на этом пути
//bool dfs(int v) {
//    if (v == t) return true; // если достигли стока, то возвращаем true
//    for (int i = 0; i < int(graph[v].size()); i++) {
//        int to = graph[v][i];
//        if (capacity[v][to] > 0 && parent[to] == -1) { // если вершина to еще не посещена и есть ненасыщенное ребро
//            parent[to] = v;        // помечаем v как предка
//            bottleneck[to] = min(bottleneck[v], capacity[v][to]); // обновляем бутылочное горлышко
//            if (dfs(to)) { // если удалось найти увеличивающий путь, то обновляем пропускные способности ребер на этом пути
//                capacity[v][to] -= bottleneck[to];
//                capacity[to][v] += bottleneck[to];
//                return true;
//            }
//        }
//    }
//    return false; // увеличивающий путь не найден
//}
//
//// найти максимальный поток в графе
//int max_flow() {
//    int result = 0;
//    while (true) {
//        memset(parent, -1, sizeof(parent)); // обнулить массив предков
//        memset(bottleneck, 0, sizeof(bottleneck)); // обнулить массив бутылочных горлышек
//        bottleneck[s] = INT_MAX; // бутылочное горлышко для истока равно бесконечности
//        parent[s] = -2;           // пометить начальную вершину
//        if (!dfs(s)) break;       // если увеличивающий путь не найден, то выходим из цикла
//        result += bottleneck[t]; // увеличиваем максимальный поток на значение бутылочного горлышка накопившегося потока
//    }
//    return result;
//}
//
//int main() {
//
//    for (int i = 0; i < m; i++) {
//        int from, to;
//        cin >> from >> to;
//        add_edge(from, to); // добавляем ребро весом 1
//    }
//    cout << max_flow(); // вывод максимального потока
//    return 0;
//}
//
