#include<string>
#include<cstring>
#include<fstream>
#include<vector>
#include<iostream>
using namespace std;

vector<int> zFoo(string s) {
	int n = (int)s.length();
	vector<int> z(n);
	for (int i = 1, l = 0, r = 0; i < n; ++i) {
		if (i <= r)
			z[i] = min(r - i + 1, z[i - l]);
		while (i + z[i] < n && s[z[i]] == s[i + z[i]])
			++z[i];
		if (i + z[i] - 1 > r)
			l = i, r = i + z[i] - 1;
	}
	return z;
}

int search(string p,string s) {
	string t = p + '¹' + s;
	vector<int> zT;
	zT = zFoo(t);
	for (int i = p.size() + 1; i < t.size(); ++i) {
		if (zT[i] == p.size()) {
			return i-p.size()-1;
		}
	}
}

int main() {
	string stroka = "aabc aaabc abcabc";
	string ans = "abc";
	int n;
	n = search(ans, stroka);
	cout << n;
	return 0;
}