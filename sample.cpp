#include<iostream>
#include<string>
#include<vector>
#include<fstream>
using namespace std;
int len(string& s) {
	int i = 0;
	while (s[i] != '\0') {
		i++;
	}
	return i;
}
int main() {
	setlocale(LC_ALL, "RUS");
	int ans=1;
	unsigned char a;
	string text;
	ifstream f;
	getline(f, text);
	f.open("idiot.txt");
	while(!f.eof()){
		f >> text;
		for (int i = 0; i < len(text); i++) {
			a=text[i];
			ans *= (int)a;
		}
		
	}
	cout << ans << "\n";
	f.close();
	return 0;
}
