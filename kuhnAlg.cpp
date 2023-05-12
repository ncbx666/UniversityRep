#include<iostream>
#include<vector>
using namespace std;

// ������� ������ � ������ ���� ��������������

const int n = 6, k = 6;

// ���������� ������ ��������� ��� ������ ������� �����
vector < vector<int> > g(n);

// ������ ��� ������ �� ������ ����. ��� ������ ���������� � ���, ���� �� � ������� �� ������ ���� ������� ����� �� �������������
vector<int> match(k, -1);

// ������ ��� ������ �� ����� ����
vector<bool> used(n, false);

// �������, ������� ��������� ���� �� ������������� ���� �� ������� v � ������������ �� ������� �� ������ ����
bool try_kuhn(int v) {
	if (used[v])  return false; // ���� �� �����, ��� �� �������� ���� �� ���� ������� � � ��� �� ����������, �� �� ���� ������ 
	used[v] = true; // ����������, ��� ������� ������������
	for (int i = 0; i < g[v].size(); ++i) { // ���������� ���� ������� v
		int to = g[v][i]; // ���������� ������� ������� �������
		if (match[to] == -1 || try_kuhn(match[to])) { // ���� ������� ������� ������� �� ��������, ��� ������ ����� �������������� ����, ������� ������������� � ������������ �������
			match[to] = v; // �������� ������� �� ������ ���� ������ � v
			return true; // ���������� �����
		}
	}
	return false; // ���� ����� �������������� ���� �� �������, ���������� ����
}

int main() {
	// ���������� ������ ��������� ��� ������ ������� �����
	g[0] = { 0,1,2,3,4,5 };
	g[1] = { 0,1 };
	g[2] = { 0,2 };
	g[3] = { 0,2,3,4,5 };
	g[4] = { 0 };
	g[5] = { 5 };

	match.assign(k, -1); // ���������� ����� �� �������
	for (int v = 0; v < n; ++v) { // ���������� ������� �� ����� ����
		if (try_kuhn(v)) { //���� ����� ������������� ����, �� ���������� ��� ����� ������
			used.assign(n, false);
		}
	}

	for (int i = 0; i < k; ++i) // ������� ��� ���� ������, ������� �� ����� � �������������
		if (match[i] != -1)
			cout<< match[i] + 1 <<' '<< i + 1 << '\n';

	return 0;
}