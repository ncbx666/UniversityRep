#include<iostream>
#include<string>
#include<fstream>
using namespace std;
bool findstr(string answer, string stroka) {
	bool flag = 0;
	int N = stroka.size();
	if(answer.size()<=N){
	for (int i=0; i < N- answer.size() ; ++i) {
		int k = 0;
		for (int j = 0; j < answer.size(); ++j) {
			if (stroka[i + j] == answer[j]) {
				++k;
				if (k == 3) {
					flag = 1;
				}
			}
		}
		}
	}
	return flag;
}
int main() {
	ifstream str;
	str.open("stroka.txt");
	string answer;
	getline(str, answer);
	bool flagg=0;
	while (!str.eof()) {
		string tempstr;
		getline(str, tempstr);
		if (findstr(answer, tempstr)) {
			flagg = 1;
		}
	}
	cout << flagg;
	
	return 0;
}