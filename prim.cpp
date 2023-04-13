#include<iostream>
#include<vector>
#include<string>
#include<fstream>
using namespace std;
bool findd(vector<int> a,int key) { 
	bool flag = 1;
	for (int i = 0; i < a.size(); i++) {
		if (a[i] == key) {
			flag = 0;
		}
	}
	return flag;
}
int main(){
	int dlina;
	cout << "enter the size of graph";
	cin >> dlina;
	int** ms = new int* [dlina];
	for (int i = 0; i < dlina; i++) {
		ms[i] = new int[dlina];
	}
	ifstream f;
	f.open("myGraph.txt");
	for (int i = 0; i < dlina; i++) {
		for (int j = 0; j < dlina; j++) {
			int b;
			f >> b;
			ms[i][j] = b;
		}
	}
	f.close();
	
	for (int i = 0; i < dlina; i++) {
		for (int j = i + 1; j < dlina; j++) {
				if(ms[i][j]==0){
				ms[i][j] = 1e10; // убрали все нули
			}
		}
	}
	int minimum = 0;
	vector<int> vert;
	for (int i = 0; i < dlina; i++) {
		vert.push_back(i);
	}
	vector<int> used = {0};
	vert.erase(remove(vert.begin(), vert.end(), 0), vert.end()); // удалили нулевую вершину, так как она в использованных
	while (vert.size() != 0) {
		int minim = 1e9;
		int vertic;
		for (int i = 0; i < used.size(); i++) { // прохожусь по всем использованным вершинам
			int k = used[i]; // берем текущую использованную вершину
			for (int j = k; j < dlina; j++) { // перебираем все вершины, инцидентные данной и ищем минимальный вес ребра
				if (findd(used, j)) { // если вершина в использованных, то тогда не перебираем, чтобы не было повторяющихся ребер.
					if (ms[k][j] < minim) {
						minim = ms[k][j];
						vertic = j; // сохраняем индекс вершины с минимальным весом ребра
					}
				}
			}
		}
		minimum += minim; // увеличиваем длину минимального пути
		used.push_back(vertic); // добавляем вершину в использованные
		vert.erase(remove(vert.begin(), vert.end(), vertic), vert.end()); //удаляем использованную вершину из нашего исходного графа
	}
	cout << "the shortest weight=" << minimum; 
	return 0;
}