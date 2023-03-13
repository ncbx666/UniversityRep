#include<iostream>
#include<string>
#include<vector>
#include<random>
#include<fstream>
#include<map>
using namespace std;
void copy(vector<int>& a, vector<int>& b) {
	for (int i = 0; i < b.size(); i++) {
		a.push_back(b[i]);
	}
}//копирование массива в другой
int ind(int z, vector<int>& a) {
	int ans;
	for (int i = 0; i < a.size(); i++) {
		if (z == a[i]) {
			ans = i;
			break;
		}
	}
	return ans;
} //поиск индекса элемента в массиве
void bubble(vector<int> &m)
{
	for (int i =0; i <m.size(); i++)
		for (int j = 0; j < m.size()-1; j++)
		{
			if (m[j] < m[j + 1])
			{
				int foo = m[j];
				m[j] = m[j + 1];
				m[j + 1] = foo;
			}
		}
}//сортировка массива по убыванию
int index_alph(char a, string& s) {
	int i = 0;
	while (s[i] != '\0') {
		if (s[i] == a) break;
		i++;
	}
	return i / 2; //чтобы не было разницы больша€ или маленька€ буква
} //поиск индекса символа в алфавите
int find(char a, string& s) {
	int i = 0;
	int p = 0;
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == a) {
			p = 1;
			break;
		}
	}
	if (p == 1) {
		return 1;
	}
	else {
		return 0;
	}
}// поиск символа в строке
int len(string& s) {
	int i = 0;
	while (s[i] != '\0') {
		i++;
	}
	return i;
}//длина строки
int main() {
	setlocale(LC_ALL, "RUS");
	vector<int> num;
	vector<int> num_copy;
	for(int i=0;i<33;i++){
		num.push_back(0);
	}
	unsigned char a;
	string alph = "јаЅб¬в√гƒд≈е®Є∆ж«з»и…й кЋлћмЌнќоѕп–р—с“т”у‘ф’х÷ц„чЎшўщЏъџы№ьЁэёюя€"; //алфавит
	string text; //сюда записываем строку
	ifstream f;
	f.open("idiot.txt");
	while (!f.eof()) { //чтение файла
		getline(f, text);
		for (int i = 0; i < len(text); i++) {
			if (find(text[i],alph) == 1) { //проверка на нахождение символа в русском алфавите
				num[index_alph(text[i], alph)] += 1; //подсчет букв
			}
		}
	}
	f.close();
	copy(num_copy, num);
	bubble(num_copy);
	for (int i = 0; i < num_copy.size(); i++) {
		cout << alph[ind(num_copy[i],num)*2] << ' '; //ответ
	}cout << endl;
	return 0;
}