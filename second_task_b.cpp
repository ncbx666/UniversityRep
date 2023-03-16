#include<iostream>
#include<string>
#include<vector>
#include<random>
#include<fstream>
#include<map>
using namespace std;
string getRandstring(string& s, int n) {
	string ans = "";
	int l = rand()%(s.size()-n);
	int r = l + n - 1;
	for (int i = l; i < r; i++) {
		ans += s[i];
	}
	return ans;
}
int len(string& s) {
	int i = 0;
	while (s[i] != '\0') {
		i++;
	}
	return i;
}//длина строки
int hashfoo(string& line) {
	long sum = 0;
	for (int i = 0; i < len(line); i++) {
		if ((int)(line[i])<=122 and (int)line[i]>=97){
			sum += (int)line[i];
		}
		else if ((int)(line[i]) >= 65 and (int)(line[i]) <= 90) {
			sum += (int)line[i];
		}
	}
	return sum;
}
int main() {
	srand(time(NULL));
	int k;
	cout << "type the len of string";
	cin >> k;
	int n;
	cout << "type the number of iterations";
	cin >> n;
	ifstream f,fil;
	string str,s;
	f.open("idiot.txt");
	if (f.is_open())
	{
		while (getline(f, str))
		{
			s += str;
		}
	}
	f.close();
	int c = 0;
	fil.open("idiot.txt");
	for (int i = 0; i < n; i++) {
		string t = getRandstring(s, k);
		string temp = getRandstring(s, k);
		if (hashfoo(t)==hashfoo(temp)) {
			c++;
		}
	}
	fil.close();
	cout << c;
	return 0;
}