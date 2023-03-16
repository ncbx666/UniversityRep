#include<iostream>
#include<string>
#include<fstream>
#include<map>
using namespace std;
void cipher(string& s) {
    string t = "";
    for (int i = 0; i < s.size(); i++) {
        t += char(((int)s[i] + 2));
    }
    s = t;
}
void uncipher(string& s) {
    string t = "";
    for (int i = 0; i < s.size(); i++) {
        t += char(((int)s[i] - 2));
    }
    s = t;
}

int main() {
    ifstream f;
    string s;
    f.open("11.txt");
    while (!f.eof()) {
        string t;
        getline(f, t);
        s += t;
    }
    f.close();
    cipher(s);
    uncipher(s);
    return 0;
}
